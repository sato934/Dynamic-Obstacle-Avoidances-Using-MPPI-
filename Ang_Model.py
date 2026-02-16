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
    # バッチ行列逆変換で高速化 (ループ不要)
    O_batch = np.transpose(O, (2, 0, 1))  # (way_num, 3, 3)
    # 特異行列チェック: 行列式がゼロに近い場合はゼロ行列に置換
    dets = np.linalg.det(O_batch)
    valid = np.abs(dets) > 1e-10
    inv_batch = np.zeros_like(O_batch)
    if np.any(valid):
        inv_batch[valid] = np.linalg.inv(O_batch[valid])
    inv_batch = np.nan_to_num(inv_batch)
    # バッチ行列ベクトル積: (way_num, 3, 3) @ (way_num, 3, 1) -> (way_num, 3, 1)
    pqr_batch = pqr.T[:, :, np.newaxis]  # (way_num, 3, 1)
    dot_avwxyz = (inv_batch @ pqr_batch).squeeze(-1).T  # (3, way_num)

    dot_phi = dot_avwxyz[0, :]
    dot_theta = dot_avwxyz[1, :]
    dot_psi = dot_avwxyz[2, :]

    phi = phi + dot_phi * times
    theta = theta + dot_theta * times
    psi = psi + dot_psi * times

    next_state = np.vstack([phi, theta, psi, dot_phi, dot_theta, dot_psi])
    return next_state

