import numpy as np
from copy import deepcopy
import random

from abc import ABC
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.layers import Flatten, Dense, Activation, Input, Concatenate
from keras import backend as K

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from collections import deque

from math import sin, cos, tanh
from pal.environments import CoopPendulum


class Controller(ABC):
    """ Parent class for all controllers """
    def __init__(self, first_player: bool):
        self.first_player = first_player  # Defines whether it calculates u1 or u2. So if it is player one or not

    def u(self, t, x):
        return 0

    def new_exp(self, exp):
        pass

    def get_other_pred(self, t, x):  # regular pal knows nothing -> return 0
        return 0.


class StaticController(Controller):
    """ A pal that has a static nonlinear behavior """
    def u(self, t, x):
        assert len(x) > 1
        x1 = np.ndarray.item(x[0])
        x2 = np.ndarray.item(x[1])
        ret = sin(x1) - tanh(x2)  # Rule, how the pal computes the output
        return ret


class StaticControllerV2(Controller):
    """ A pal that has a static nonlinear behavior (u not 0 at x1 = 0) """
    def u(self, t, x):
        assert len(x) > 1
        x1 = np.ndarray.item(x[0])
        x2 = np.ndarray.item(x[1])
        ret = cos(x1) - tanh(x2)  # Rule, how the pal computes the output
        return ret


class StaticControllerCustomFunction(Controller):

    def __init__(self, first_player: bool, fn=None):
        super().__init__(first_player)
        if fn is None:
            print("No custom function defined. Will return 0")
            self.function = lambda x1, x2: 0
        else:
            self.function = fn

    def u(self, t, x):
        return self.function(x[0], x[1])


class StaticZeroController(Controller):
    """ A pal that always returns 0 """
    def u(self, t, x):
        return 0


class StaticNNController(Controller):
    """ A pal with a static NN as a control law """
    def __init__(self, first_player, neural_net):
        super(StaticNNController, self).__init__(first_player)
        self.neural_net = neural_net
        self.input_shape = tuple([1 if v is None else v for v in self.neural_net.input_shape])

    def u(self, t, x):
        """ Calculate the output for a given input """
        inputs = np.reshape(x, self.input_shape)
        #  inputs to policy NN that calculates the control variable
        u_self = np.ndarray.item(self.neural_net.predict(inputs))  # calculate the output of the policy NN
        return u_self


class ChangingController(Controller):
    """ Controller that changes behavior abruptly at change_time """
    def __init__(self, first_player, change_time=10.):
        super(ChangingController, self).__init__(first_player)
        self.change_time = change_time

    def u(self, t, x):
        assert len(x) > 1
        x1 = np.ndarray.item(x[0])
        x2 = np.ndarray.item(x[1])
        if t < self.change_time:
            ret = sin(x1) - tanh(x2)  # Rule, how the pal computes the output before changing
        else:
            ret = - 0.2 * x1 + 0.8 * x2  # Rule, ... after changing
        return ret


class ChangingControllerCustomFunction(ChangingController):
    def __init__(self, first_player: bool, change_time, fns: tuple=(None,)):
        super().__init__(first_player, change_time)
        self.change_time = change_time
        self.functions = fns
        for fn in self.functions:
            if fn is None:
                print("No custom function defined.")
                raise NotImplementedError

    def u(self, t, x):
        i = int(t/self.change_time) % len(self.functions)
        return self.functions[i](x[0], x[1])


class ContinuallyChangingControllerCustomFunction(ChangingControllerCustomFunction):
    def u(self, t, x):
        change_parameter = t/self.change_time
        i = int(change_parameter) % len(self.functions)
        j = (int(change_parameter)+1) % len(self.functions)
        weight_of_next = change_parameter - int(change_parameter)
        u = (1-weight_of_next)*self.functions[i](x[0], x[1]) + weight_of_next*self.functions[j](x[0], x[1])
        return u


class LearningStack:
    """ A stack that saves the experiences of a PartnerApproximatingLearner and provides methods for using the experience """
    def __init__(self, memory_span=10., use_cer=False, batch_percent=20.):  # hyper-parameter
        self._stack = deque(maxlen=int(1e6))  # much lower size is calculated from memory span after 2 experiences
        self.memory_span = memory_span
        self.use_cer = use_cer
        self.batch_percent = batch_percent
        self.last_number_exp = 0  # Amount of experiences the last time pick_random was called (for CER)

    def add(self, experience):
        """ Bring new experience to the stack """
        self._stack.append(experience)
        if len(self._stack) == 3:  # calculate LearningStack size from memory_span
            dt = self._stack[2][0] - self._stack[1][0]
            self._stack = deque(self._stack, maxlen=int(self.memory_span/dt))

    def pick_random(self):
        """ Picks X percent of the learning stack at random to train the model on """
        stack = self._stack
        number_exp = len(stack)
        n = int((self.batch_percent / 100) * self._stack.maxlen)  # Number of experiences to put in one batch
        n = min(n, number_exp)  # except if that would be bigger than the number of experiences available
        number_new = number_exp - self.last_number_exp  # number of new experiences since last training
        n = max(number_new, n)  # For small batch percent and batch size small make sure at least the new ones are used
        # CER: Always add newest exp as well to mitigate problems resulting from big max_stack_size (Zhang, Sutton 2018)
        if self.use_cer and n < number_exp and number_new < number_exp:  # With CER
            exp_indices_picked = random.sample(range(0, number_exp-number_new), n - number_new)  # pick random old exp
            exp_indices_picked += list(range(number_exp-number_new, number_exp))  # pick all of the new experiences
            batch = [stack[i] for i in exp_indices_picked]
            random.shuffle(batch)  # shuffle to not get a mini batch in training that only contains only new exp
        else:  # Without CER (or in case it makes no difference f.e. n = number_exp)
            exp_indices_picked = random.sample(range(0, number_exp), n)
            batch = [stack[i] for i in exp_indices_picked]  # is already shuffled
        self.last_number_exp = number_exp
        return batch

    def get_all_experiences(self):
        return self._stack


class PartnerApproximatingLearner(Controller):
    """ A pal that learns how the other pal behaves and adapts to that behavior """

    def __init__(self, first_player: bool, stop_ident_time=1e9, do_rl=False, learning_rate=0.01, activation_fcn='relu',
                 learn_time_delta=0.2, rl_time_delta=0.1, epochs=2, fit_batch_size=20, learn_stack=LearningStack(),
                 real_env=CoopPendulum(), rl_memory_span=50, wolf=0., win_lr_reduction=1, wolf_stop_rl=False):
        """ Sets various parameters, configures the ident, actor and critic NN and compiles the agent"""
        super(PartnerApproximatingLearner, self).__init__(first_player)  # Call to __init__ of parent class Controller
        self.learn_stack = learn_stack  # Controller specific LearningStack in which to save the experiences
        self.loosing_lr = learning_rate
        self.rl_lr = .001  # hyper-parameter
        self.win_lr_reduction = win_lr_reduction
        self.wolf = wolf
        self.wolf_stop_rl = wolf_stop_rl
        seed = np.random.randint(0, int(1e6)) + int(first_player)*100  # -> first player gets different seed than second

        # Configure neural network for identification:
        num_hidden_layer_ident = 3
        num_neurons_per_layer_ident = 16
        act_space_shape = real_env.action_space.shape
        obs_space_shape = real_env.observation_space.shape
        ident_nn = Sequential()
        ident_nn.add(Dense(num_neurons_per_layer_ident, kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=seed)
                           , input_shape=obs_space_shape))
        for i in range(num_hidden_layer_ident-1):  # Add the layers to the identification NN
            ident_nn.add(Dense(num_neurons_per_layer_ident, kernel_initializer=RandomUniform(minval=-1, maxval=1,
                                                                                             seed=seed + i)))
            ident_nn.add(Activation(activation_fcn))
        ident_nn.add(Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-0.0001, maxval=0.0001,
                                                                                seed=seed + 9)))
        ident_nn.add(Activation('linear'))
        opt = Adam(lr=learning_rate)  # hyper-parameter
        ident_nn.compile(optimizer=opt, loss='mse')  # hyper-parameter

        # Use the neural network inside a NNController for easy evaluation of the output:
        self.ident_ctrl = StaticNNController(first_player=(not self.first_player), neural_net=ident_nn)

        # Set other identification parameters
        self.ident_time_delta = learn_time_delta  # simulation time between training the other_model with experience
        self.last_ident_time = 0  # last time ident NN was trained
        self.epochs = epochs  # number of training epochs when its time to identify again
        self.fit_batch_size = fit_batch_size  # size of mini batch that the batch is split into for training by Keras
        self.stop_ident_time = stop_ident_time  # Time at which no training should occur anymore. Used for testing
        self.do_rl = do_rl
        if do_rl:
            self.rl_env = deepcopy(real_env)
            self.last_rl_time = -1
            self.rl_time_delta = rl_time_delta
            self.rl_env.set_ctrl_other(self.ident_ctrl)
            try:
                self.u_limit = self.rl_env.action_space_u1 if first_player else self.rl_env.action_space_u2
            except AttributeError:  # rl_env does not have individual limits
                self.u_limit = self.rl_env.action_space

            # Configure the Neural Networks of the RL-agent
            # 1. Actor:
            rl_num_hidden_layer_actor = 3
            rl_num_neurons_per_layer_actor = 16
            rl_actor = Sequential()  # Actor is a Sequential Neural Network (MLP)
            rl_actor.add(Flatten(input_shape=(1,) + obs_space_shape))
            for i in range(rl_num_hidden_layer_actor):  # Add the layers to the actor NN
                rl_actor.add(Dense(rl_num_neurons_per_layer_actor,
                                   kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=seed+10+i)))
                rl_actor.add(Activation(activation_fcn))
            rl_actor.add(Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=seed+19)))
            rl_actor.add(Activation('linear'))

            # 2. Critic:
            rl_num_hidden_layer_critic = 3
            rl_num_neurons_per_layer_critic = 32
            action_input = Input(shape=act_space_shape, name='action_input')
            observation_input = Input(shape=(1,) + obs_space_shape, name='observation_input')
            flattened_observation = Flatten()(observation_input)
            rl_critic_nn = Concatenate()([action_input, flattened_observation])
            for i in range(rl_num_hidden_layer_critic):
                rl_critic_nn = Dense(rl_num_neurons_per_layer_critic,
                                     kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=seed+20+i))(rl_critic_nn)
                rl_critic_nn = Activation(activation_fcn)(rl_critic_nn)
            rl_critic_nn = Dense(1, kernel_initializer=RandomUniform(minval=-1, maxval=1, seed=seed+29))(rl_critic_nn)
            rl_critic_nn = Activation('linear')(rl_critic_nn)
            rl_critic = Model(inputs=[action_input, observation_input], outputs=rl_critic_nn)

            # 3. Set training parameters for the Agent and compile it
            rl_frames_per_train = 200
            rl_mem_size = int(rl_memory_span * (round(1/self.rl_time_delta) * rl_frames_per_train))
            rl_memory = SequentialMemory(limit=rl_mem_size, window_length=1)
            random_process = OrnsteinUhlenbeckProcess(size=act_space_shape[0], theta=.15, mu=0., sigma=.3)
            self.rl_agent = DDPGAgent(nb_actions=act_space_shape[0], actor=rl_actor, critic=rl_critic,
                                      critic_action_input=action_input, memory=rl_memory, nb_steps_warmup_critic=100,
                                      nb_steps_warmup_actor=100, random_process=random_process, gamma=.99,
                                      target_model_update=1e-3)
            self.rl_agent.compile(Adam(lr=self.rl_lr, clipnorm=1.), metrics=['mae'])
            self.rl_actor_ctrl = StaticNNController(first_player=self.first_player, neural_net=rl_actor)

    def ident_other(self):
        """ Updates Identification of the other pal """
        batch = self.learn_stack.pick_random()  # get a batch from the LearningStack
        batch_t, batch_x, batch_u = zip(*batch)
        batch_u1, batch_u2 = zip(*batch_u)
        inputs = np.asarray(batch_x)
        if self.first_player:   # -> player 2 has to be identified
            outputs = np.reshape(np.array(batch_u2), (-1, 1))
        else:                   # -> player 1 has to be identified
            outputs = np.reshape(np.array(batch_u1), (-1, 1))
        self.ident_ctrl.neural_net.fit(inputs, outputs, batch_size=self.fit_batch_size, epochs=self.epochs, verbose=0,
                                       shuffle=True, validation_split=0.)

    def u(self, t, x) -> float:
        """ Calculates the control variable u of the learning pal (only for the "real" environment)
            The action inside the internal simulation is calculated by the actor NN and clipped by the env """
        if self.do_rl:
            u_self = self.rl_actor_ctrl.u(t, x)
            u_self = min(max(u_self, np.ndarray.item(self.u_limit.low)), np.ndarray.item(self.u_limit.high))
        else:
            u_self = 0.
        return u_self

    def get_other_pred(self, t, x):
        """ Returns the expected output of the other pal for the given input """
        u_other_pred = self.ident_ctrl.u(t, x)
        return u_other_pred

    def calc_error_on_learning_stack(self):
        stack = self.learn_stack.get_all_experiences()
        stack_t, stack_x, stack_u = zip(*stack)
        stack_u1, stack_u2 = zip(*stack_u)
        predictions = list()
        for i in range(len(stack_x)):
            predictions.append(self.get_other_pred(stack_t[i], stack_x[i]))

        assert len(predictions) == len(stack_u1)
        assert len(predictions) == len(stack_u2)

        if self.first_player:   # Predicting Player 2
            error = [(stack_u2[j] - predictions[j])**2 for j in range(len(predictions))]
        else:                   # Predicting Player 1
            error = [(stack_u1[j] - predictions[j]) ** 2 for j in range(len(predictions))]

        mse = sum(error) / len(error)
        return mse

    def new_exp(self, exp):
        """ Saves the new experience (time, state, control variables) on the stack and
            triggers rl/ident if enough time passed """
        self.learn_stack.add(exp[0:3])
        t_now = exp[0]
        winning = False
        if len(exp) > 3:  # if "real" reward is supplied: check if within winning limits
            winning = exp[3] > self.wolf  # hyper-parameter

        if self.do_rl and round(t_now - self.last_rl_time, 5) >= self.rl_time_delta:  # enough time passed since last RL
            if winning:
                K.set_value(self.rl_agent.actor_optimizer.lr, self.rl_lr/self.win_lr_reduction)
                K.set_value(self.rl_agent.critic.optimizer.optimizer.lr, self.rl_lr / self.win_lr_reduction)
            else:
                K.set_value(self.rl_agent.actor_optimizer.lr, self.rl_lr)
                K.set_value(self.rl_agent.critic.optimizer.optimizer.lr, self.rl_lr)
            if not (self.wolf_stop_rl and winning):
                self.improve_policy()
                self.last_rl_time = t_now

        if round(t_now - self.last_ident_time, 5) >= self.ident_time_delta and t_now < self.stop_ident_time:
            if winning:
                K.set_value(self.ident_ctrl.neural_net.optimizer.lr, self.loosing_lr/self.win_lr_reduction)
            else:
                K.set_value(self.ident_ctrl.neural_net.optimizer.lr, self.loosing_lr)
            self.ident_other()  # train my model of the other pal on data from my LearningStack
            self.last_ident_time = t_now

    def improve_policy(self):
        """ Does an episode of RL to improve critic and actor of the rl_agent """
        self.rl_agent.fit(self.rl_env, nb_steps=200, visualize=False, verbose=0, nb_max_episode_steps=200)
