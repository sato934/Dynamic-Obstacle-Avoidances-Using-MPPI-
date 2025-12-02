import numpy as np

def Ang_Model(State, Ctrl, P):
    State = np.atleast_2d(State)
    if State.shape[0] != 12 and State.shape[1] == 12:
        State = State.T

    way_num = Ctrl.shape[1]
    times = P['dt']

    phi = State[6, :]
    theta = State[7, :]
    psi = State[8, :]

    # p, q, r を求める式
    O = np.zeros((3, 3, way_num))
    O[0, 0, :] = np.cos(theta)
    O[0, 1, :] = 0
    O[0, 2, :] = -np.cos(phi) * np.sin(theta)
    O[1, 0, :] = 0
    O[1, 1, :] = 1
    O[1, 2, :] = np.sin(phi)
    O[2, 0, :] = np.sin(theta)
    O[2, 1, :] = 0
    O[2, 2, :] = np.cos(phi) * np.cos(theta)
    pqr = Ctrl[1:4, :]

    dot_avwxyz = np.zeros((3, way_num))  # dot_angle_velocity_world_xyz
    for i in range(way_num):
        try:
            u = np.linalg.inv(O[:, :, i])
        except np.linalg.LinAlgError:
            u = np.zeros((3, 3))
        u = np.nan_to_num(u)
        dot_avwxyz[:, i] = u @ pqr[:, i]

    dot_phi = dot_avwxyz[0, :]
    dot_theta = dot_avwxyz[1, :]
    dot_psi = dot_avwxyz[2, :]

    phi = phi + dot_phi * times
    theta = theta + dot_theta * times
    psi = psi + dot_psi * times

    next_state = np.vstack([phi, theta, psi, dot_phi, dot_theta, dot_psi])
    return next_state

