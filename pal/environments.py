import numpy as np
from copy import deepcopy
from rl.core import Env
from gym.spaces import Box
from gym import make as gym_make


class CoopPendulum(Env):
    """ An environment that assumes the other controller as being fixed that a controller can act on """
    def __init__(self, ctrl_other=None, max_torque=None, action_space_u1=None, action_space_u2=None, perfident = False):
        ENV_NAME = 'Pendulum-v0'

        # Get the environment and extract the number of actions.
        self.env = gym_make(ENV_NAME).env  # last .env enables episodes longer than 200 steps
        assert len(self.env.action_space.shape) == 1
        self.other_controller = ctrl_other  # Can be overwritten by calling set_ctrl_other() afterwards
        self.t = 0.
        self.dt = self.env.unwrapped.dt
        self.state = self.reset()
        self.action_space = self.env.action_space
        self.action_space_u1 = deepcopy(self.action_space) if action_space_u1 is None else action_space_u1  # limits u1
        self.action_space_u2 = deepcopy(self.action_space) if action_space_u2 is None else action_space_u2  # limits u2
        self.max_torque = self.env.unwrapped.max_torque if max_torque is None else max_torque  # limits the summed u
        high = np.array([np.pi, 1])
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)
        self.perfident = perfident  # if the limit of u_other is incorporated

    def step(self, u):
        """ Performs one step.
        Case1: "Real" System: u is a tuple of u1 and u2. -> reward for the virtual controller that both
        controllers form together
        Case2: Simulated System: u is only the one value a single agent calculated and influence of u_other is
        calculated with self.other_controller.u which is normally set to the ident_controller -> Only reward for the
        agent that is training right now
        """
        info = dict()
        if self.other_controller is None:  # "Real" Environment. u is tuple of u1 and u2
            assert len(u) == 2
            u1_unclipped, u2_unclipped = u[0], u[1]
            u1_clipped = min(max(u1_unclipped, np.ndarray.item(self.action_space_u1.low)), np.ndarray.item(self.action_space_u1.high))
            u2_clipped = min(max(u2_unclipped, np.ndarray.item(self.action_space_u2.low)), np.ndarray.item(self.action_space_u2.high))
            u_all_clipped = min(max(u1_clipped + u2_clipped, -self.max_torque), self.max_torque)
            r_ctrl1 = self.reward_ctrl1(self.state[0], self.state[1], u1_unclipped)
            r_ctrl2 = self.reward_ctrl2(self.state[0], self.state[1], u2_unclipped)
            r = r_ctrl1 + r_ctrl2
            info['r1'] = r_ctrl1
            info['r2'] = r_ctrl2
            info['u1_clipped'] = u1_clipped
            info['u2_clipped'] = u2_clipped
        else:  # This is a simulation inside one of the controllers
            assert len(u) == 1  # This is a simulation. u is only u_self. u_other is calculated from ident
            # This is a simulation. We do not know what the second ctrls limits are
            my_limit = self.action_space_u2 if self.other_controller.first_player else self.action_space_u1
            u_other = self.other_controller.u(0., self.state)
            if self.perfident:
                other_limit = self.action_space_u2 if not self.other_controller.first_player else self.action_space_u1
                u_other = min(max(u_other, np.ndarray.item(other_limit.low)), np.ndarray.item(other_limit.high))
            u_self_unclipped = u[0]
            u_self_clipped = min(max(u_self_unclipped, np.ndarray.item(my_limit.low)), np.ndarray.item(my_limit.high))
            u_all_clipped = min(max(u_other + u_self_clipped, -self.max_torque), self.max_torque)
            if self.other_controller.first_player:
                # r_ctrl1 = self.reward_ctrl1(self.state[0], self.state[1], u_other)  # Not used in simulation
                r_ctrl2 = self.reward_ctrl2(self.state[0], self.state[1], u_self_unclipped)
                r = r_ctrl2
            else:
                r_ctrl1 = self.reward_ctrl1(self.state[0], self.state[1], u_self_unclipped)
                # r_ctrl2 = self.reward_ctrl2(self.state[0], self.state[1], u_other)  # Not used in simulation
                r = r_ctrl1
        # Call to Pendulum implementation by OpenAI
        self.state, __, done, __ = self.env.step((u_all_clipped,))

        phi = self.angle_normalize(self.env.unwrapped.state[0])
        self.state = (phi, self.env.unwrapped.state[1])
        self.t += self.dt
        info['t'] = self.t
        return self.state, r, done, info

    def reset(self):
        self.env.reset()
        self.state = (self.env.unwrapped.state[0], self.env.unwrapped.state[1])
        self.t = 0.
        return self.state

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def configure(self, *args, **kwargs):
        raise NotImplementedError

    def set_ctrl_other(self, ctrl_other):
        self.other_controller = ctrl_other

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward_ctrl1(self, th, thdot, u):
        return - self.angle_normalize(th)**2 - .1 * thdot**2 - .001 * (u**2)

    def reward_ctrl2(self, th, thdot, u):
        return - self.angle_normalize(th)**2 - .1 * thdot**2 - .001 * (u**2)


class NegotiateCoopPendulum(CoopPendulum):
    """
    The other Has two optima: one at the right and one at the left at the same angle. They should agree on the one that
     both have """

    def __init__(self, max_torque=20, action_space_u1=Box(np.array([-10]), np.array([10]), dtype=np.float32),
                 action_space_u2=Box(np.array([-10]), np.array([10]), dtype=np.float32)):
        super().__init__(max_torque=max_torque, action_space_u1=action_space_u1, action_space_u2=action_space_u2)

    def configure(self, *args, **kwargs):
        pass

    def reward_ctrl1(self, th, thdot, u):  # Does not care which side
        if self.angle_normalize(th) < 0:
            return - 0.7*(abs(self.angle_normalize(th)) - np.pi/4) ** 2 - .1 * thdot ** 2 - .01 * (u ** 2)
        else:
            return - (abs(self.angle_normalize(th)) - np.pi / 4) ** 2 - .1 * thdot ** 2 - .01 * (u ** 2)

    def reward_ctrl2(self, th, thdot, u):  # Wants to be at positive 0.3047
        return - (self.angle_normalize(th) - np.pi/4) ** 2 - .1 * thdot ** 2 - .01 * (u ** 2)


class PendulumOpenAIdepricated(Env):
    """ An environment that assumes the other controller as being fixed that a controller can act on """
    def __init__(self, ctrl_other=None):
        ENV_NAME = 'Pendulum-v0'

        # Get the environment and extract the number of actions.
        self.env = gym_make(ENV_NAME)
        assert len(self.env.action_space.shape) == 1
        self.other_controller = ctrl_other  # Static Controller for now
        self.t = 0.
        self.dt = self.env.unwrapped.dt
        self.state = self.reset()
        self.action_space = self.env.action_space
        self.max_torque = self.env.unwrapped.max_torque
        high = np.array([np.pi, 1])
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def step(self, u_self):
        u_other = 0.
        if self.other_controller is not None:
            u_other = self.other_controller.u(0., self.state)  # self.last_u_self)
            # print("u_other: {0}, u_self: {1}".format(u_other, u_self))
        u_all = (u_other + sum(u_self),)
        self.state, r, done, info = self.env.step(u_all)
        phi = self.angle_normalize(self.env.unwrapped.state[0])
        self.state = (phi, self.env.unwrapped.state[1])
        self.t += self.dt
        return self.state, r, done, {"t": self.t}

    def reset(self):
        self.state = self.env.reset()
        self.state = (self.env.unwrapped.state[0], self.env.unwrapped.state[1])
        self.t = 0.
        return self.state

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def configure(self, *args, **kwargs):
        raise NotImplementedError

    def set_ctrl_other(self, ctrl_other):
        self.other_controller = ctrl_other

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
