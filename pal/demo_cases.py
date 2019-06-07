from pal.environments import CoopPendulum, NegotiateCoopPendulum
from pal.agents import PartnerApproximatingLearner, StaticControllerCustomFunction, LearningStack
from pal.simulation import Simulation
from pal.baseline_agents_no_sim import simulate_coop_ddpg
import numpy as np
from gym.spaces import Box


def demo_both_learn_swing_up(n=300, runs=1):
    memory_span = 20
    batch_percent = 10
    learn_time_delta = 0.05
    rl_mem_span = 10
    act_fcn = 'sigmoid'
    info = "bothswingup,mem{0},bp{1},ltd{2},n{3},rl_sp{4},act{5}".format(memory_span, batch_percent,
                                                                        learn_time_delta, n,
                                                                        rl_mem_span, act_fcn)
    for i in range(runs):
        print("------------------------------------------------------------")
        print("Run Number " + str(i + 1) + " of " + str(runs))
        print(info)
        env = CoopPendulum(max_torque=10, action_space_u1=Box(np.array([-5]), np.array([5]), dtype=np.float32),
                           action_space_u2=Box(np.array([-5]), np.array([5]), dtype=np.float32))
        learning_stack1 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)
        learning_stack2 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)

        ctrl1 = PartnerApproximatingLearner(learn_stack=learning_stack1, first_player=True, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=n, real_env=env,
                                            rl_memory_span=rl_mem_span)

        ctrl2 = PartnerApproximatingLearner(learn_stack=learning_stack2, first_player=False, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=n, real_env=env,
                                            rl_memory_span=rl_mem_span)

        simulation = Simulation(sys=env, ctrl_1=ctrl1, ctrl_2=ctrl2)
        simulation.evaluate(n, time_out_seconds=1e6, show_plot=True, progress=True, plot_file_name=info,
                            calc_error_on_learning_stack=True)


def demo_negotiate_pend(n=300, runs=1):
    memory_span = 100
    batch_percent = 2
    learn_time_delta = 0.05
    rl_mem_span = 10
    act_fcn = 'sigmoid'
    info = "negpend,mem{0},bp{1},ltd{2},n{3},rl_sp{4},act{5}".format(memory_span, batch_percent, learn_time_delta, n,
                                                                      rl_mem_span, act_fcn)
    for i in range(runs):
        print("------------------------------------------------------------")
        print("Run Number " + str(i+1) + " of " + str(runs))
        print(info)
        env = NegotiateCoopPendulum()
        learning_stack1 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)
        learning_stack2 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)

        ctrl1 = PartnerApproximatingLearner(learn_stack=learning_stack1, first_player=True, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=n, real_env=env,
                                            rl_memory_span=rl_mem_span)
        ctrl2 = PartnerApproximatingLearner(learn_stack=learning_stack2, first_player=False, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=n, real_env=env,
                                            rl_memory_span=rl_mem_span)

        simulation = Simulation(sys=env, ctrl_1=ctrl1, ctrl_2=ctrl2)
        simulation.evaluate_critic(n, time_out_seconds=1e6, show_plot=True, progress=True, plot_file_name=info)


def demo_both_learn_swing_up_no_ident(n=300, runs=1):
    memory_span = 1
    batch_percent = 10
    learn_time_delta = 0.05
    rl_mem_span = 10
    act_fcn = 'sigmoid'
    info = "bothswingup,mem{0},bp{1},ltd{2},n{3},rl_sp{4},act{5}_no_ident".format(memory_span, batch_percent,
                                                                                  learn_time_delta, n,
                                                                                  rl_mem_span, act_fcn)
    for i in range(runs):
        print("------------------------------------------------------------")
        print("Run Number " + str(i + 1) + " of " + str(runs))
        print(info)
        env = CoopPendulum(max_torque=100, action_space_u1=Box(np.array([-5]), np.array([5]), dtype=np.float32),
                           action_space_u2=Box(np.array([-5]), np.array([5]), dtype=np.float32))
        learning_stack1 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)
        learning_stack2 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)

        ctrl1 = PartnerApproximatingLearner(learn_stack=learning_stack1, first_player=True, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=-1, real_env=env,
                                            rl_memory_span=rl_mem_span)

        ctrl2 = PartnerApproximatingLearner(learn_stack=learning_stack2, first_player=False, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=-1, real_env=env,
                                            rl_memory_span=rl_mem_span)

        simulation = Simulation(sys=env, ctrl_1=ctrl1, ctrl_2=ctrl2)
        simulation.evaluate(n, time_out_seconds=1e6, show_plot=True, progress=True, plot_file_name=info,
                            calc_error_on_learning_stack=True)


def demo_reality_learner():
    simulate_coop_ddpg()


def rosenbrock_scaled(x1, x2):
    return ((1 - x1)**2 + 100*(x2 - x1**2)**2)/5000 - 2


def demo_other_rosenbrock(n=300, runs=1):  # The partner has a constant control law (the rosenbrock function)
    memory_span = 100
    batch_percent = 10
    learn_time_delta = 0.05
    rl_mem_span = 10
    act_fcn = 'sigmoid'
    info = "otherrosen,mem{0},bp{1},ltd{2},n{3},rl_sp{4},act{5}".format(memory_span, batch_percent, learn_time_delta, n,
                                                                   rl_mem_span, act_fcn)
    for i in range(runs):
        print("------------------------------------------------------------")
        print("Run Number " + str(i+1) + " of " + str(runs))
        print(info)
        env = CoopPendulum(max_torque=100, action_space_u1=Box(np.array([-5]), np.array([5]), dtype=np.float32),
                           action_space_u2=Box(np.array([-5]), np.array([5]), dtype=np.float32))
        learning_stack1 = LearningStack(memory_span=memory_span, use_cer=True, batch_percent=batch_percent)

        ctrl1 = PartnerApproximatingLearner(learn_stack=learning_stack1, first_player=True, do_rl=True, learning_rate=0.01,
                                            learn_time_delta=learn_time_delta, activation_fcn=act_fcn,
                                            epochs=4, fit_batch_size=20, stop_ident_time=n, real_env=env,
                                            rl_memory_span=rl_mem_span)

        ctrl2 = StaticControllerCustomFunction(first_player=False, fn=rosenbrock_scaled)

        simulation = Simulation(sys=env, ctrl_1=ctrl1, ctrl_2=ctrl2)
        simulation.evaluate(n, time_out_seconds=1e6, show_plot=True, progress=True, plot_file_name=info,
                            calc_error_on_learning_stack=True)
