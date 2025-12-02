def Term_Cost(sim_state):

    # sim_state[2, :] > 0 なら cost=0, それ以外は大きな値
    import numpy as np
    z = sim_state[2, :]
    cost = np.where(z > 0, 0, 100000000000)
    return cost
