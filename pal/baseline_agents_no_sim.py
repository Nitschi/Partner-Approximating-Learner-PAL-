import numpy as np
import warnings
from copy import deepcopy
from collections import deque
from datetime import datetime

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.layers import Flatten, Dense, Activation, Input, Concatenate
from keras.callbacks import History

from gym.spaces import Box

from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.agents.ddpg import DDPGAgent
from rl.core import Agent
from rl.util import clone_optimizer
from rl.callbacks import CallbackList, TrainEpisodeLogger, TrainIntervalLogger, Visualizer

from pal.environments import CoopPendulum
from pal.open_plot import open_plot


class CoopDDPG(Agent):  # Two Agents, who can not measure the output of the other (Based on Keras-rl agent impl.)


    def forward(self, observation):
        raise NotImplementedError

    def backward(self, reward, terminal):
        raise NotImplementedError

    def load_weights(self, filepath):
        raise NotImplementedError

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError

    @property
    def layers(self):
        raise NotImplementedError

    def __init__(self, nb_actions, actor1, actor2, critic1, critic2, critic_action_input1, critic_action_input2,
                 memory1, memory2,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process1=None, random_process2=None, custom_model_objects={}, target_model_update=.001,
                 **kwargs):

        super(CoopDDPG, self).__init__()

        self.agent1 = DDPGAgent(nb_actions, actor1, critic1, critic_action_input1, memory1, gamma, batch_size,
                                nb_steps_warmup_critic, nb_steps_warmup_actor, train_interval, memory_interval,
                                delta_range, delta_clip, random_process1, custom_model_objects, target_model_update,
                                **kwargs)
        self.agent2 = DDPGAgent(nb_actions, actor2, critic2, critic_action_input2, memory2, gamma, batch_size,
                                nb_steps_warmup_critic, nb_steps_warmup_actor, train_interval, memory_interval,
                                delta_range, delta_clip, random_process2, custom_model_objects, target_model_update,
                                **kwargs)

    def compile(self, optimizer, metrics=[]):
        self.agent1.compile(clone_optimizer(optimizer), deepcopy(metrics))
        self.agent2.compile(clone_optimizer(optimizer), deepcopy(metrics))

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):

        if not (self.agent1.compiled and self.agent2.compiled):
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.agent1.training = True
        self.agent2.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self.agent1._on_train_begin()
        self.agent2._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.agent1.step = np.int16(0)
        self.agent2.step = np.int16(0)
        observation = None
        episode_reward1 = None
        episode_reward2 = None
        episode_step = None
        did_abort = False
        try:
            while self.agent1.step < nb_steps:  # not individual for now
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward1 = np.float32(0)
                    episode_reward2 = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.agent1.reset_states()
                    self.agent2.reset_states()
                    observation = deepcopy(env.reset())
                    if self.agent1.processor is not None:  # not individual for now
                        observation = self.agent1.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.agent1.processor is not None:  # not individual for now. action is not from agent anyway
                            action = self.agent1.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.agent1.processor is not None:
                            observation, reward, done, info = self.agent1.processor.process_step(observation, reward,
                                                                                                 done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. '
                                'You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.agent1.processor is not None:
                                observation = self.agent1.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward1 is not None
                assert episode_reward2 is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action1 = self.agent1.forward(observation)
                action2 = self.agent2.forward(observation)
                if self.agent1.processor is not None:
                    action1 = self.agent1.processor.process_action(action1)
                if self.agent2.processor is not None:
                    action2 = self.agent2.processor.process_action(action2)
                action = (np.ndarray.item(action1), np.ndarray.item(action2))
                reward1 = np.float32(0)
                reward2 = np.float32(0)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)  # Use only one of the actions? added actions?
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.agent1.processor is not None:
                        observation, r, done, info = self.agent1.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward1 += info["r1"]
                    reward2 += info["r2"]
                    reward += info["r1"] + info["r2"]
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics1 = self.agent1.backward(reward1, terminal=done)
                metrics2 = self.agent2.backward(reward2, terminal=done)
                episode_reward1 += reward1
                episode_reward2 += reward2

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics1,  # not individual for now
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.agent1.step += 1
                self.agent2.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.agent1.forward(observation)
                    self.agent2.forward(observation)
                    self.agent1.backward(0., terminal=False)
                    self.agent2.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward1 + episode_reward2,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.agent1.step,  # not individual for now
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward1 = None
                    episode_reward2 = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self.agent1._on_train_end()
        self.agent2._on_train_end()


        return history


class CoopActionOtherDDPG(Agent):  # Two Agents, who can measure the output of the other (Based on Keras-rl agent impl.)

    def forward(self, observation):
        raise NotImplementedError

    def backward(self, reward, terminal):
        raise NotImplementedError

    def load_weights(self, filepath):
        raise NotImplementedError

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError

    @property
    def layers(self):
        raise NotImplementedError

    def __init__(self, nb_actions, actor1, actor2, critic1, critic2, critic_action_input1, critic_action_input2,
                 memory1, memory2,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process1=None, random_process2=None, custom_model_objects={}, target_model_update=.001,
                 **kwargs):

        super(CoopActionOtherDDPG, self).__init__()

        self.agent1 = DDPGAgent(nb_actions, actor1, critic1, critic_action_input1, memory1, gamma, batch_size,
                                nb_steps_warmup_critic, nb_steps_warmup_actor, train_interval, memory_interval,
                                delta_range, delta_clip, random_process1, custom_model_objects, target_model_update,
                                **kwargs)
        self.agent2 = DDPGAgent(nb_actions, actor2, critic2, critic_action_input2, memory2, gamma, batch_size,
                                nb_steps_warmup_critic, nb_steps_warmup_actor, train_interval, memory_interval,
                                delta_range, delta_clip, random_process2, custom_model_objects, target_model_update,
                                **kwargs)

    def compile(self, optimizer, metrics=[]):
        self.agent1.compile(clone_optimizer(optimizer), deepcopy(metrics))
        self.agent2.compile(clone_optimizer(optimizer), deepcopy(metrics))

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not (self.agent1.compiled and self.agent2.compiled):
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        assert self.processor is None  # Removed processors here for simplification. Not needed anyway
        assert nb_max_start_steps == 0  # Removed here for simplification. Not needed anyway
        assert action_repetition == 1  # Removed here for simplification. Not needed anyway

        self.agent1.training = True
        self.agent2.training = True

        experience_for_plotting = deque()

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self.agent1._on_train_begin()
        self.agent2._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.agent1.step = np.int16(0)
        self.agent2.step = np.int16(0)
        observation1 = observation2 = None
        episode_reward1 = None
        episode_reward2 = None
        episode_step = None
        did_abort = False
        try:
            while self.agent1.step < nb_steps:  # not individual for now
                if observation1 is None or observation2 is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward1 = np.float32(0)
                    episode_reward2 = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.agent1.reset_states()
                    self.agent2.reset_states()
                    obs = env.reset()
                    observation1 = deepcopy(obs) + (0.,)
                    observation2 = deepcopy(obs) + (0.,)

                # At this point, we expect to be fully initialized.
                assert episode_reward1 is not None
                assert episode_reward2 is not None
                assert episode_step is not None
                assert observation1 is not None
                assert observation2 is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action1 = np.ndarray.item(self.agent1.forward(observation1))
                action2 = np.ndarray.item(self.agent2.forward(observation2))
                action = (action1, action2)
                reward1 = np.float32(0)
                reward2 = np.float32(0)
                accumulated_info = {}
                done = False

                callbacks.on_action_begin(action)  # Use only one of the actions? added actions?
                obs, r, done, info = env.step(action)
                if done:
                    raise AttributeError  # The episode was reset unexpectedly
                    # (see https://stackoverflow.com/questions/42787924/)

                observation1 = deepcopy(obs) + (info["u2_clipped"],)  # Add action other to the observation
                observation2 = deepcopy(obs) + (info["u1_clipped"],)
                for key, value in info.items():
                    if not np.isreal(value):
                        continue
                    if key not in accumulated_info:
                        accumulated_info[key] = np.zeros_like(value)
                    accumulated_info[key] += value
                callbacks.on_action_end(action)
                reward1 += info["r1"]
                reward2 += info["r2"]

                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics1 = self.agent1.backward(reward1, terminal=done)
                metrics2 = self.agent2.backward(reward2, terminal=done)
                episode_reward1 += reward1
                episode_reward2 += reward2

                step_logs = {
                    'action': action[0] + action[1],
                    'observation': observation1,
                    'reward': reward1 + reward2,
                    'metrics': metrics1,  # not individual for now
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.agent1.step += 1
                self.agent2.step += 1

                if len(obs) == 2:
                    experience_for_plotting.append((info["t"], obs, (info["u1_clipped"], info["u2_clipped"]), (0., 0.),
                                                    r, (info["r1"], info["r2"])))

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.agent1.forward(observation1)
                    self.agent2.forward(observation2)
                    self.agent1.backward(0., terminal=False)
                    self.agent2.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward1 + episode_reward2,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.agent1.step,  # not individual for now
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation1 = None
                    observation2 = None
                    episode_step = None
                    episode_reward1 = None
                    episode_reward2 = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self.agent1._on_train_end()
        self.agent2._on_train_end()

        return experience_for_plotting


class RealityLearnSimulation:

    def __init__(self, env, rl_lr, rl_memory_span):
        self.real_env = env
        act_space_shape = self.real_env.action_space.shape
        obs_space_shape = self.real_env.observation_space.shape

        # Configure the Neural Networks of the RL-agent
        # 1. Actors:
        rl_num_hidden_layer_actor = 3
        rl_num_neurons_per_layer_actor = 16
        rl_actor1 = Sequential()  # Actor1 is a Sequential Neural Network (MLP)
        rl_actor1.add(Flatten(input_shape=(1,) + obs_space_shape))
        for i in range(rl_num_hidden_layer_actor):  # Add the layers to the actor1 NN
            rl_actor1.add(Dense(rl_num_neurons_per_layer_actor, kernel_initializer=RandomUniform(minval=-1, maxval=1)))
            rl_actor1.add(Activation('relu'))
        rl_actor1.add(Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-1, maxval=1)))
        rl_actor1.add(Activation('linear'))

        rl_actor2 = Sequential()  # Actor2 is a Sequential Neural Network (MLP)
        rl_actor2.add(Flatten(input_shape=(1,) + obs_space_shape))
        for i in range(rl_num_hidden_layer_actor):  # Add the layers to the actor2 NN
            rl_actor2.add(Dense(rl_num_neurons_per_layer_actor, kernel_initializer=RandomUniform(minval=-1, maxval=1)))
            rl_actor2.add(Activation('relu'))
        rl_actor2.add(
            Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-1, maxval=1)))
        rl_actor2.add(Activation('linear'))

        # 2. Critics:
        rl_num_hidden_layer_critic = 3
        rl_num_neurons_per_layer_critic = 32

        action_input1 = Input(shape=act_space_shape, name='action_input')
        observation_input1 = Input(shape=(1,) + obs_space_shape, name='observation_input')
        flattened_observation1 = Flatten()(observation_input1)
        rl_critic_nn1 = Concatenate()([action_input1, flattened_observation1])
        for i in range(rl_num_hidden_layer_critic):
            rl_critic_nn1 = Dense(rl_num_neurons_per_layer_critic,
                                  kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn1)
            rl_critic_nn1 = Activation('relu')(rl_critic_nn1)
        rl_critic_nn1 = Dense(1, kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn1)
        rl_critic_nn1 = Activation('linear')(rl_critic_nn1)
        rl_critic1 = Model(inputs=[action_input1, observation_input1], outputs=rl_critic_nn1)

        action_input2 = Input(shape=act_space_shape, name='action_input')
        observation_input2 = Input(shape=(1,) + obs_space_shape, name='observation_input')
        flattened_observation2 = Flatten()(observation_input2)
        rl_critic_nn2 = Concatenate()([action_input2, flattened_observation2])
        for i in range(rl_num_hidden_layer_critic):
            rl_critic_nn2 = Dense(rl_num_neurons_per_layer_critic,
                                  kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn2)
            rl_critic_nn2 = Activation('relu')(rl_critic_nn2)
        rl_critic_nn2 = Dense(1, kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn2)
        rl_critic_nn2 = Activation('linear')(rl_critic_nn2)
        rl_critic2 = Model(inputs=[action_input2, observation_input2], outputs=rl_critic_nn2)

        # 3. Set training parameters for the Agent and compile it
        rl_mem_size = int(rl_memory_span * round(1 / self.real_env.dt))
        rl_memory1 = SequentialMemory(limit=rl_mem_size, window_length=1)
        rl_memory2 = SequentialMemory(limit=rl_mem_size, window_length=1)
        random_process1 = OrnsteinUhlenbeckProcess(size=act_space_shape[0], theta=.15, mu=0., sigma=.3)
        random_process2 = OrnsteinUhlenbeckProcess(size=act_space_shape[0], theta=.15, mu=0., sigma=.3)

        self.coop_agent = CoopDDPG(nb_actions=act_space_shape[0], actor1=rl_actor1, actor2=rl_actor2, critic1=rl_critic1,
                                   critic2=rl_critic2, critic_action_input1=action_input1,
                                   critic_action_input2=action_input2, memory1=rl_memory1, memory2=rl_memory2,
                                   nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, random_process1=random_process1,
                                   random_process2=random_process2, gamma=.99, target_model_update=1e-3)
        self.coop_agent.compile(Adam(lr=rl_lr, clipnorm=1.), metrics=['mae'])

    def simulate(self, duration):
        nb_steps = duration*round(1/self.real_env.dt)
        self.coop_agent.fit(self.real_env, nb_steps, verbose=0, visualize=False, nb_max_episode_steps=nb_steps)


class RealityLearnSimulationActionOther:

    def __init__(self, env, rl_lr, rl_memory_span):
        self.real_env = env
        act_space_shape = self.real_env.action_space.shape
        obs_space_shape = (self.real_env.observation_space.shape[0] + 1,)
        assert len(obs_space_shape) == 1

        # Configure the Neural Networks of the RL-agent
        # 1. Actors:
        rl_num_hidden_layer_actor = 3
        rl_num_neurons_per_layer_actor = 16
        rl_actor1 = Sequential()  # Actor1 is a Sequential Neural Network (MLP)
        rl_actor1.add(Flatten(input_shape=(1,) + obs_space_shape))
        for i in range(rl_num_hidden_layer_actor):  # Add the layers to the actor1 NN
            rl_actor1.add(Dense(rl_num_neurons_per_layer_actor, kernel_initializer=RandomUniform(minval=-1, maxval=1)))
            rl_actor1.add(Activation('relu'))
        rl_actor1.add(Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-1, maxval=1)))
        rl_actor1.add(Activation('linear'))

        rl_actor2 = Sequential()  # Actor2 is a Sequential Neural Network (MLP)
        rl_actor2.add(Flatten(input_shape=(1,) + obs_space_shape))
        for i in range(rl_num_hidden_layer_actor):  # Add the layers to the actor2 NN
            rl_actor2.add(Dense(rl_num_neurons_per_layer_actor, kernel_initializer=RandomUniform(minval=-1, maxval=1)))
            rl_actor2.add(Activation('relu'))
        rl_actor2.add(
            Dense(act_space_shape[0], kernel_initializer=RandomUniform(minval=-1, maxval=1)))
        rl_actor2.add(Activation('linear'))

        # 2. Critics:
        rl_num_hidden_layer_critic = 3
        rl_num_neurons_per_layer_critic = 32

        action_input1 = Input(shape=act_space_shape, name='action_input')
        observation_input1 = Input(shape=(1,) + obs_space_shape, name='observation_input')
        flattened_observation1 = Flatten()(observation_input1)
        rl_critic_nn1 = Concatenate()([action_input1, flattened_observation1])
        for i in range(rl_num_hidden_layer_critic):
            rl_critic_nn1 = Dense(rl_num_neurons_per_layer_critic,
                                  kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn1)
            rl_critic_nn1 = Activation('relu')(rl_critic_nn1)
        rl_critic_nn1 = Dense(1, kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn1)
        rl_critic_nn1 = Activation('linear')(rl_critic_nn1)
        rl_critic1 = Model(inputs=[action_input1, observation_input1], outputs=rl_critic_nn1)

        action_input2 = Input(shape=act_space_shape, name='action_input')
        observation_input2 = Input(shape=(1,) + obs_space_shape, name='observation_input')
        flattened_observation2 = Flatten()(observation_input2)
        rl_critic_nn2 = Concatenate()([action_input2, flattened_observation2])
        for i in range(rl_num_hidden_layer_critic):
            rl_critic_nn2 = Dense(rl_num_neurons_per_layer_critic,
                                  kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn2)
            rl_critic_nn2 = Activation('relu')(rl_critic_nn2)
        rl_critic_nn2 = Dense(1, kernel_initializer=RandomUniform(minval=-1, maxval=1))(rl_critic_nn2)
        rl_critic_nn2 = Activation('linear')(rl_critic_nn2)
        rl_critic2 = Model(inputs=[action_input2, observation_input2], outputs=rl_critic_nn2)

        # 3. Set training parameters for the Agent and compile it
        rl_mem_size = int(rl_memory_span * round(1 / self.real_env.dt))
        rl_memory1 = SequentialMemory(limit=rl_mem_size, window_length=1)
        rl_memory2 = SequentialMemory(limit=rl_mem_size, window_length=1)
        random_process1 = OrnsteinUhlenbeckProcess(size=act_space_shape[0], theta=.15, mu=0., sigma=.3)
        random_process2 = OrnsteinUhlenbeckProcess(size=act_space_shape[0], theta=.15, mu=0., sigma=.3)

        self.coop_agent = CoopActionOtherDDPG(nb_actions=act_space_shape[0], actor1=rl_actor1, actor2=rl_actor2, critic1=rl_critic1,
                                   critic2=rl_critic2, critic_action_input1=action_input1,
                                   critic_action_input2=action_input2, memory1=rl_memory1, memory2=rl_memory2,
                                   nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, random_process1=random_process1,
                                   random_process2=random_process2, gamma=.99, target_model_update=1e-3)
        self.coop_agent.compile(Adam(lr=rl_lr, clipnorm=1.), metrics=['mae'])

    def simulate(self, duration):
        nb_steps = duration*round(1/self.real_env.dt)
        return self.coop_agent.fit(self.real_env, nb_steps, verbose=0, visualize=False, nb_max_episode_steps=nb_steps)


def simulate_coop_ddpg(n=15000):
    print(datetime.now())

    env = CoopPendulum(max_torque=10, action_space_u1=Box(np.array([-5]), np.array([5]), dtype=np.float32),
                 action_space_u2=Box(np.array([-5]), np.array([5]), dtype=np.float32))

    rl_lrs = [0.01]
    rl_memory_spans = [30]
    num_combinations = len(rl_lrs) * len(rl_memory_spans)
    counter = 1
    for rl_lr in rl_lrs:
        rl_lr = round(rl_lr, 4)
        for rl_mem_span in rl_memory_spans:
            # sim = RealityLearnSimulation(env=env, rl_lr=.001, rl_memory_span=64)
            print("----------------------Combination {0} of {1}----------------------".format(counter, num_combinations))
            counter += 1

            sim = RealityLearnSimulationActionOther(env=env, rl_lr=rl_lr, rl_memory_span=rl_mem_span)  # u_other observable
            info = "rlsneg,rl_lr{0},rlmem{1},n{2}".format(rl_lr, rl_mem_span, n)
            print(info)
            experience_for_plotting = sim.simulate(n)


            plot_file_name = info
            # Convert experience_for_plotting to a numpy array
            exp = list(zip(*experience_for_plotting))  # each component is all t
            number_exp = len(exp[0])
            for i in range(len(exp)):
                exp[i] = np.reshape(exp[i], (number_exp, -1))
            experience = np.concatenate(tuple(exp), axis=1)

            if plot_file_name is not None:
                full_plot_file_name = plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S")
                try:
                    np.save("data/" + full_plot_file_name, experience)
                    open_plot("data/" + full_plot_file_name+".npy")
                except OSError:
                    print("Folder data does not exist. Plot/run data not saved to disk")
                try:
                    sim.coop_agent.agent1.critic.save("data/models/" + "critic1-" + full_plot_file_name)
                    sim.coop_agent.agent1.actor.save("data/models/" + "actor1-" + full_plot_file_name)
                    sim.coop_agent.agent2.critic.save("data/models/" + "critic2-" + full_plot_file_name)
                    sim.coop_agent.agent2.actor.save("data/models/" + "actor2-" + full_plot_file_name)
                except OSError:
                    print("Folder data/models does not exist. Actor and critic models not saved")
            t = experience[:, 0]
            r = experience[:, 7]
            sum_r = round(sum(r), 2)
            #print("Total reward:                {0}         per sec: {1}".format(sum_r, round(sum_r / t[-1], 2)))
            dt = t[2] - t[1]
            steps_per_second = int(round(1 / dt))
            if round(t[-1], 5) >= 10000:  # run was longer or equal 200s*m simulation time
                reward_100_200 = round(sum(r[5000 * steps_per_second:10000 * steps_per_second]), 2)
                print("r from 100->{0}:             {1}         per sec: {2}".format(10000, reward_100_200,
                                                                                     round(reward_100_200 / 5000),
                                                                                     2))
            if round(t[-1], 5) > 10000:  # run was longer than 200s*m simulation time
                reward_from200 = round(sum(r[10000 * steps_per_second:]), 2)
                print("r from {0}-> end:            {1}        per sec: {2}".format(10000, reward_from200, round(
                    reward_from200 / (t[-1] - 10000), 2)))
            if experience.shape[1] > 9:  # if individual rewards were saved as well
                r1, r2 = experience[:, 8], experience[:, 9]
