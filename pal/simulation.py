from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from copy import deepcopy

from pal.agents import PartnerApproximatingLearner, StaticController


class Simulation:

    def __init__(self, sys, ctrl_1=PartnerApproximatingLearner(first_player=True),
                 ctrl_2=StaticController(first_player=False)):
        self.system = sys  # System which the controllers act on
        self.controller1 = ctrl_1  # Only identifies for now
        self.controller2 = ctrl_2  # Static Controller for now
        self.mse_c1 = [1e6]

    def simulate(self, duration, time_out_seconds, progress, calc_error_on_learning_stack=False):
        """ Simulates the system with controllers for a duration
         Returns a numpy array with t_plot,x_plot,u and u_pred for plotting or evaluation """
        steps = int(round(1/self.system.dt)*duration)  # calculate number of total steps
        time = 0.
        last_observation = self.system.reset()  # Get the initial observation by calling reset()
        experience_for_plotting = deque()
        last_percent = 0.  # used for progress indicator
        timer = datetime.now()  # start time to determine if simulation takes longer than time_out_seconds
        last_time_progress_printed = timer  # Used to calculate how long it will take
        for i in range(steps):

            # Calculate control signal, prediction and next step
            u1 = self.controller1.u(time, last_observation)
            u2 = self.controller2.u(time, last_observation)
            try:
                u1 = min(max(u1, np.ndarray.item(self.system.action_space_u1.low)),
                                 np.ndarray.item(self.system.action_space_u1.high))
                u2 = min(max(u2, np.ndarray.item(self.system.action_space_u2.low)),
                                 np.ndarray.item(self.system.action_space_u2.high))
            except AttributeError:
                print("No clipping performed")
                pass
            u2_pred_by_c1 = self.controller1.get_other_pred(time, last_observation)
            u1_pred_by_c2 = self.controller2.get_other_pred(time, last_observation)
            u_all = (u1, u2)  # get u's into the correct shape
            observation, r, done, info = self.system.step(u_all)
            time = info["t"]

            if progress:  # progress indication
                new_percent = round((time/duration)*100)
                if new_percent != last_percent:
                    last_percent = new_percent
                    time_elapsed = datetime.now() - last_time_progress_printed
                    last_time_progress_printed = datetime.now()
                    print("{0}% Done ----- {1} to go".format(new_percent, (100-new_percent)*time_elapsed))

            # Save results for training
            exp = (time, observation, u_all)

            # Extend results for training with the predictions for subsequent plotting
            plot_exp = deepcopy(exp)
            if 'r1' in info and 'r2' in info:  # rewards
                self.controller1.new_exp(exp + (info['r1'],))
                self.controller2.new_exp(exp + (info['r2'],))

                if calc_error_on_learning_stack:
                    mse_c1 = self.controller1.calc_error_on_learning_stack()
                    plot_exp += ((u1_pred_by_c2, u2_pred_by_c1), r, (info['r1'], info['r2']), (mse_c1,))
                else:
                    plot_exp += ((u1_pred_by_c2, u2_pred_by_c1), r, (info['r1'], info['r2']))

            else:
                self.controller1.new_exp(exp)
                self.controller2.new_exp(exp)
                plot_exp += ((u1_pred_by_c2, u2_pred_by_c1), r)

            experience_for_plotting.append(plot_exp)
            last_observation = observation

            if datetime.now() - timer > timedelta(seconds=time_out_seconds):  # to stop simulation that takes to long
                print("Simulation took too long: ", datetime.now() - timer)
                return experience_for_plotting, self.system.dt, True  # last value says if it took too long

        return experience_for_plotting, self.system.dt, False  # last value says if it took too long

    def evaluate(self, duration, show_plot=True, time_out_seconds=1e6, plot_generalization=False, progress=False
                 , plot_file_name=None, calc_error_on_learning_stack=False):
        """ Starts a simulation to print metrics and plot performance """

        timer = datetime.now()  # to time how long the simulation runs
        experience_for_plotting, dt, took_too_long = \
            self.simulate(duration, time_out_seconds=time_out_seconds, progress=progress,
                          calc_error_on_learning_stack=calc_error_on_learning_stack)

        time_taken = datetime.now() - timer
        print("time taken: ", time_taken)

        # Convert experience_for_plotting to a numpy array
        exp = list(zip(*experience_for_plotting))  # each component is all t
        number_exp = len(exp[0])
        for i in range(len(exp)):
            exp[i] = np.reshape(exp[i], (number_exp, -1))
        experience = np.concatenate(tuple(exp), axis=1)

        if took_too_long:
            if plot_file_name is not None:
                np.save("data/" + plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S") + "-took_too_long", experience)
                print(plot_file_name + " took too long")
            return timedelta(hours=1), 1e6, 1e6, -1e6  # set errors very high if episode was too long

        if plot_file_name is not None:
            full_plot_file_name = plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S")
            np.save("data/" + full_plot_file_name, experience)
            print(full_plot_file_name)

        t = experience[:, 0]
        x1 = experience[:, 1]
        x2 = experience[:, 2]
        u1 = experience[:, 3]
        u2 = experience[:, 4]
        u1_pred_by_c2 = experience[:, 5]  # Value that pal 2 predicted for the output of pal 1
        u2_pred_by_c1 = experience[:, 6]
        prediction_error_c1 = u2 - u2_pred_by_c1
        prediction_error_c2 = u1 - u1_pred_by_c2

        # evaluate
        start_time_index = int(5 / dt)
        total_error_from_5s = np.sum(np.absolute(prediction_error_c1[start_time_index:])) * dt
        total_error_from_10s = np.sum(np.absolute(prediction_error_c1[(start_time_index*2):])) * dt
        print("Total error u2_pred  (5s->) : ", total_error_from_5s)  # Error from 5 seconds onward summed up
        print("Total error u2_pred (10s->) : ", total_error_from_10s)

        if experience.shape[1] > 7:  # if summed reward was saved as well
            r = experience[:, 7]
            sum_r = round(sum(r), 2)
            print("Total reward:                {0}         per sec: {1}".format(sum_r, round(sum_r / t[-1], 2)))
            dt = t[2] - t[1]
            steps_per_second = int(round(1 / dt))
            if round(t[-1], 5) >= 200:  # run was longer or equal 200s simulation time
                reward_100_200 = round(sum(r[100 * steps_per_second:200 * steps_per_second]), 2)
                print("r from 100->200:             {0}         per sec: {1}".format(reward_100_200,
                                                                                     round(reward_100_200 / 100),
                                                                                     2))
            if round(t[-1], 5) > 200:  # run was longer than 200s simulation time
                reward_from200 = round(sum(r[200 * steps_per_second:]), 2)
                print("r from 200-> end:            {0}        per sec: {1}".format(reward_from200, round(
                    reward_from200 / (t[-1] - 200), 2)))
            if experience.shape[1] > 9:  # if individual rewards were saved as well
                r1, r2 = experience[:, 8], experience[:, 9]
                if experience.shape[1] > 10:
                    self.mse_c1 = experience[:, 10]

        # plot
        if show_plot:  # comment in lines to plot more information (see also in evaluate2 if nothing changes)
            plt.figure()
            plt.plot(t, x1, color='red', linestyle='solid', linewidth=0.5, label='x1')
            plt.plot(t, x2, color='green', linestyle='solid', linewidth=0.5, label='x2')
            # plt.plot(t, prediction_error_c2, color='magenta', linestyle='None', linewidth=0.5, label='error_u1')
            # plt.plot(t, prediction_error_c1, color='pink', linestyle='None', linewidth=0.5, label='error_u2')
            plt.plot(t, u2, color='blue', linestyle='None', linewidth=0.5, label='u2')
            # plt.plot(t, u1, color='cyan', linestyle='None', linewidth=1, label='u1')
            plt.plot(t, u2_pred_by_c1, color='black', linestyle='None', linewidth=0.5, label='u2_pred_by_c1')
            # plt.plot(t, u1_pred_by_c2, color='orange', linestyle='None', linewidth=0.5, label='u1_pred_by_c2')

            #if experience.shape[1] > 7:  # if summed reward was saved as well
                #plt.plot(t, r, color='magenta', linestyle='solid', linewidth=0.5, label='reward summed')
                #if experience.shape[1] > 9:
                    #plt.plot(t, r1, color='black', linestyle='None', linewidth=0.5, label='reward u1')
                    #plt.plot(t, r2, color='orange', linestyle='None', linewidth=0.5, label='reward u2')

                    #if experience.shape[1] > 10:
                        #plt.plot(t, self.mse_c1, color='green', linestyle='None', linewidth=0.5, label='mse_c1')

            plt.ylim((-5, 5))
            plt.xlabel('time')
            plt.ylabel('state')
            plt.legend()
            plt.tight_layout()

        # plot generalization
        # plot the outputs of u2_pred_by_c1 for different combinations of x1 and x2 values to see if they correspond to
        # the real u2
        if plot_generalization:
            from matplotlib.ticker import LinearLocator, FormatStrFormatter

            # Make data.
            X1 = np.linspace(-np.pi, np.pi, 30)  # range to plot
            X2 = np.linspace(-np.pi, np.pi, 30)
            X1, X2 = np.meshgrid(X1, X2)
            U1_pred = np.zeros((len(X1), len(X2)))
            U2_pred = np.zeros((len(X1), len(X2)))
            U_1 = np.zeros((len(X1), len(X2)))
            U_2 = np.zeros((len(X1), len(X2)))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):  # for every combination of x1 and x2 in the range:
                    x = np.asarray([X1[i, j], X2[i, j]])
                    U2_pred[i, j] = self.controller1.get_other_pred(t=0, x=x)
                    U1_pred[i, j] = self.controller2.get_other_pred(t=0, x=x)
                    U_2[i, j] = self.controller2.u(t=0, x=x)
                    U_1[i, j] = self.controller1.u(t=0, x=x)

            U2_error = np.abs(U_2 - U2_pred)
            U1_error = np.abs(U_1 - U1_pred)


            # Plot the surface for U1
            fig1 = plt.figure()
            ax1 = fig1.gca(projection='3d')
            surf1pred = ax1.plot_wireframe(X1, X2, U1_pred, linewidth=0.3, color='b', antialiased=True, label="u1_pred")
            surf1real = ax1.plot_wireframe(X1, X2, U_1, color='r', linewidth=0.3, antialiased=True, label="u1")
            # surf3 = ax.plot_wireframe(X1, X2, U1_error, color='g', linewidth=0.3, antialiased=True, label="|error|")
            # ax1.set_zlim(-2., 2.)
            ax1.zaxis.set_major_locator(LinearLocator(10))
            ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()

            # Plot the surface for U2
            fig2 = plt.figure()
            ax2 = fig2.gca(projection='3d')
            surf2pred = ax2.plot_wireframe(X1, X2, U2_pred, linewidth=0.3, color='b', antialiased=True, label="u2_pred")
            surf2real = ax2.plot_wireframe(X1, X2, U_2, color='r', linewidth=0.3, antialiased=True, label="u2")
            # surf3 = ax.plot_wireframe(X1, X2, U2_error, color='g', linewidth=0.3, antialiased=True, label="|error|")
            # ax2.set_zlim(-6., 6.)
            ax2.zaxis.set_major_locator(LinearLocator(10))
            ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()

            print(datetime.now())

        if show_plot or plot_generalization:
            plt.show()  # If multiple runs should be performed before plotting, please call show_plot at the end of main
        return time_taken, total_error_from_5s, total_error_from_10s, sum_r

    def evaluate2(self, duration, time_out_seconds=1e6, plot_file_name=None, progress=False, calc_error_on_learning_stack=False):
        """ Starts a simulation to print metrics and plot performance """

        timer = datetime.now()  # to time how long the simulation runs
        experience_for_plotting, dt, took_too_long = self.simulate(duration, time_out_seconds=time_out_seconds,
                                                                   progress=progress, calc_error_on_learning_stack=calc_error_on_learning_stack)

        time_taken = datetime.now() - timer
        print("time taken: ", time_taken)

        # Convert experience_for_plotting to a numpy array
        exp = list(zip(*experience_for_plotting))  # each component is all t
        number_exp = len(exp[0])
        for i in range(len(exp)):
            exp[i] = np.reshape(exp[i], (number_exp, -1))
        experience = np.concatenate(tuple(exp), axis=1)

        if took_too_long:
            rewards = dict()
            rewards["sum_r"] = -1e6
            rewards["reward_from60"] = -1e6
            rewards["reward_from200"] = -1e6
            if plot_file_name is not None:
                np.save("data/" + plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S") + "-took_too_long", experience)
                print(plot_file_name + " took too long")
            return timedelta(hours=1), rewards  # set errors very high if episode was too long

        if plot_file_name is not None:
            full_plot_file_name = plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S")
            np.save("data/" + full_plot_file_name, experience)
            print(full_plot_file_name)

        t = experience[:, 0]
        x1 = experience[:, 1]
        x2 = experience[:, 2]
        u1 = experience[:, 3]
        u2 = experience[:, 4]
        u1_pred_by_c2 = experience[:, 5]  # Value that pal 2 predicted for the output of pal 1
        u2_pred_by_c1 = experience[:, 6]
        prediction_error_c1 = u2 - u2_pred_by_c1
        prediction_error_c2 = u1 - u1_pred_by_c2

        # evaluate
        start_time_index = int(5 / dt)
        total_error_from_5s = np.sum(np.absolute(prediction_error_c1[start_time_index:])) * dt
        total_error_from_10s = np.sum(np.absolute(prediction_error_c1[(start_time_index*2):])) * dt
        print("Total error u2_pred  (5s->) : ", total_error_from_5s)  # Error from 5 seconds onward summed up
        print("Total error u2_pred (10s->) : ", total_error_from_10s)
        rewards = dict()

        if experience.shape[1] > 7:  # if summed reward was saved as well
            r = experience[:, 7]
            sum_r = round(sum(r), 2)
            rewards["sum_r"] = sum_r
            print("Total reward:                {0}         per sec: {1}".format(sum_r, round(sum_r / t[-1], 2)))
            dt = t[2] - t[1]
            steps_per_second = int(round(1 / dt))
            if round(t[-1], 5) >= 200:  # run was longer or equal 200s simulation time
                reward_100_200 = round(sum(r[100 * steps_per_second:200 * steps_per_second]), 2)
                print("r from 100->200:             {0}         per sec: {1}".format(reward_100_200,
                                                                                     round(reward_100_200 / 100),
                                                                                     2))
            if round(t[-1], 5) > 200:  # run was longer than 200s simulation time
                reward_from200 = round(sum(r[200 * steps_per_second:]), 2)
                print("r from 200-> end:            {0}        per sec: {1}".format(reward_from200, round(
                    reward_from200 / (t[-1] - 200), 2)))
                rewards["reward_from200"] = reward_from200
            if experience.shape[1] > 9:  # if individual rewards were saved as well
                r1, r2 = experience[:, 8], experience[:, 9]
            if t[-1] > 60:
                reward_from60 = round(sum(r[60 * steps_per_second:]), 2)
                rewards["reward_from60"] = reward_from60
                print("r from 60-> end:            {0}        per sec: {1}".format(reward_from60, round(
                    reward_from60 / (t[-1] - 60), 2)))

            print(datetime.now())

        return time_taken, rewards

    def evaluate_critic(self, duration, show_plot=True, time_out_seconds=1e6, plot_critics=False, progress=False,
                        plot_file_name=None, eval_point=np.pi/4, calc_error_on_learning_stack=False):
        """ Starts a simulation to print metrics and plot performance. Also check critic """

        time_taken, rewards = self.evaluate2(duration, time_out_seconds, plot_file_name=plot_file_name,
                                             progress=progress, calc_error_on_learning_stack=calc_error_on_learning_stack)

        if plot_file_name is not None:
            full_plot_file_name = plot_file_name + datetime.now().strftime("-%m-%d-%H-%M-%S")
            # compile model changing params, because keras-rl specific optimizer and loss cant be loaded with load_model
            self.controller1.rl_agent.critic.compile(loss='mean_squared_error', optimizer="sgd")
            self.controller2.rl_agent.critic.compile(loss='mean_squared_error', optimizer="sgd")
            self.controller1.rl_agent.critic.save("models/" + "critic1-" + full_plot_file_name)
            self.controller1.rl_agent.actor.save("models/" + "actor1-" + full_plot_file_name)
            self.controller2.rl_agent.critic.save("models/" + "critic2-" + full_plot_file_name)
            self.controller2.rl_agent.actor.save("models/" + "actor2-" + full_plot_file_name)

        if plot_critics:
            from matplotlib.ticker import LinearLocator, FormatStrFormatter

            # Make data.
            X1 = np.linspace(-np.pi, np.pi, 30)  # range to plot
            X2 = np.linspace(-np.pi, np.pi, 30)
            X1, X2 = np.meshgrid(X1, X2)
            U_1 = np.zeros((len(X1), len(X2)))
            U_2 = np.zeros((len(X1), len(X2)))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):  # for every combination of x1 and x2 in the range:
                    x = np.asarray([X1[i, j], X2[i, j]])
                    np.reshape(x, (1, 1, 2))

                    # U_2[i, j] = c2_critic.u(t=0, x=x)
                    # U_1[i, j] = c1_critic.u(t=0, x=x)

            # Plot the surface for U1
            fig1 = plt.figure()
            ax1 = fig1.gca(projection='3d')
            surf1real = ax1.plot_wireframe(X1, X2, U_1, color='r', linewidth=0.3, antialiased=True, label="u1")
            # ax1.set_zlim(-2., 2.)
            ax1.zaxis.set_major_locator(LinearLocator(10))
            ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()

            # Plot the surface for U2
            fig2 = plt.figure()
            ax2 = fig2.gca(projection='3d')
            surf2real = ax2.plot_wireframe(X1, X2, U_2, color='r', linewidth=0.3, antialiased=True, label="u2")
            # ax2.set_zlim(-6., 6.)
            ax2.zaxis.set_major_locator(LinearLocator(10))
            ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
        inputs = [np.asarray([[[eval_point, 0.]]]), np.asarray([[[- eval_point, 0.]]])]

        actions = np.linspace(-10., 10., num=250)

        for input in inputs:
            outputs1 = list()
            outputs2 = list()
            for action in actions:
                action_array = np.asarray([action])
                outputs1.append(np.asscalar(self.controller1.rl_agent.critic.predict([action_array, input])))

            print("C1: bei Input " + str(np.asscalar(input[0][0][0])) + " Value = " + str(max(outputs1)))
            #print(outputs1)

            for action in actions:
                action_array = np.asarray([action])
                outputs2.append(np.asscalar(self.controller2.rl_agent.critic.predict([action_array, input])))

            print("C2: bei Input " + str(np.asscalar(input[0][0][0])) + " Value = " + str(max(outputs2)))
            #print(outputs2)
            print("-----------------------")

        return time_taken, rewards
