from pal import demo_cases
from matplotlib import pyplot as plt

if __name__ == '__main__':
    demo_cases.demo_both_learn_swing_up(n=300)  # Two pals cooperating to swing the pendulum to phi=0
    # demo_cases.demo_both_learn_swing_up_no_ident(n=300)  # Without identification
    # demo_cases.demo_negotiate_pend(n=300)     # Different goals for the pals
    # demo_cases.demo_reality_learner(n=15000)  # Baseline DDPG Agents trying to cooperate

    plt.show()
