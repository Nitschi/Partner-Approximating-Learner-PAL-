import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def open_plot(filename="data/example-both_swing_up.npy"):

    experience = np.load(filename)

    experience = experience[0::1]

    xlim = (-6, experience[-1, 0] + 6)
    #xlim = (1200, 1500 + 6)
    ref = 0.3


    ylim = np.pi

    show_x1 = 'solid'
    show_x2 = 'solid'
    show_u1 = 'solid'
    show_u2 = 'None'
    show_u1_pred = 'solid'
    show_u2_pred = 'None'
    show_error1 = 'solid'
    show_error2 = 'solid'
    show_r_sum = 'None'
    show_r1 = 'None'
    show_r2 = 'None'
    show_mse_c1 = 'None'

    t = experience[:, 0]
    x1 = experience[:, 1]
    x2 = experience[:, 2]
    u1 = experience[:, 3]
    u2 = experience[:, 4]
    u1_pred_by_c2 = experience[:, 5]  # Value that controller 2 predicted for the output of controller 1
    u2_pred_by_c1 = experience[:, 6]
    prediction_error_c1 = u2 - u2_pred_by_c1
    prediction_error_c2 = u1 - u1_pred_by_c2

    # plot

    fig0 = plt.figure()

    plt.plot(t, x2, color='blue', linestyle=show_x2, linewidth=1, label='Winkelgeschw. $\omega$')
    plt.plot(t, x1, color='red', linestyle=show_x1, linewidth=1, label='Winkel $\phi$')


    #plt.plot(t, prediction_error_c2, color='magenta', linestyle=show_error1, linewidth=1.5, label='error_u1')
    #plt.plot(t, prediction_error_c1, color='green', linestyle=show_error2, linewidth=4, label='$Prädiktionsfehler e_2$')
    #plt.plot(t, u2, color='blue', linestyle=show_u2, linewidth=2, label='$u_2$')
    plt.plot(t, u1, color='cyan', linestyle=show_u1, linewidth=1, label='u1')
    #plt.plot(t, u2_pred_by_c1, color='black', linestyle=show_u2_pred, linewidth=2, label='$\hat{u}_P$')
    plt.plot(t, u1_pred_by_c2, color='orange', linestyle=show_u1_pred, linewidth=0.5, label='u1_pred_by_c2')
    if experience.shape[1] > 7:  # if summed reward was saved as well
        r = experience[:, 7]
        #plt.plot(t, r, color='magenta', linestyle=show_r_sum, linewidth=1, label='reward summed')
        sum_r = round(sum(r), 2)  # cumulative reward
        print("Total reward:                {0}         per sec: {1}".format(sum_r, round(sum_r/t[-1], 2)))
        dt = t[2] - t[1]
        steps_per_second = int(round(1/dt))
        if t[-1] >= 300:  # run was longer or equal 200s simulation time
            reward_0_300 = round(sum(r[0:300*steps_per_second]), 2)
            print("r from 0->300:             {0}         per sec: {1}".format(reward_0_300,
                                                                                 round(reward_0_300/100), 2))
        if t[-1] > 200:  # run was longer than 200s simulation time
            reward_from200 = round(sum(r[200*steps_per_second:]), 2)
            print("r from 200-> end:            {0}        per sec: {1}".format(reward_from200,
                                                                                round(reward_from200/(t[-1] - 200), 2)))
        if experience.shape[1] > 9:  # if individual rewards were saved as well
            r1, r2 = experience[:, 8], experience[:, 9]
            #plt.plot(t, r1, color='black', linestyle=show_r1, linewidth=0.5, label='reward u1')
            #plt.plot(t, r2, color='orange', linestyle=show_r2, linewidth=0.5, label='reward u2')
    if experience.shape[1] > 10:
        mse_c1 = experience[:, 10]
        #plt.plot(t, mse_c1, color='orange', linestyle=show_mse_c1, linewidth=4, label='$RMSE on Ident.-Buffer e_1$')

    plt.ylim((-ylim, ylim))
    plt.xlim(xlim)
    plt.xlabel('t (in s)')
    plt.ylabel('control variable $u_P$ (in rad/$s^2$) | prediction $\hat{u}_P$ (in rad/$s^2$)')
    plt.legend()
    plt.tight_layout()

    print(datetime.now())
    print(filename)

    if __name__ == "__main__":
        plt.show()


if __name__ == "__main__":
    open_plot()
