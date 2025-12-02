import numpy as np

def Pos_Model(State, Ctrl, P):
    """
    研究室ホームページ(以下HPと書く)の式を使う．http://158.217.166.203:3000/Drone/DynamicalModel
    この関数でやってること:
    HP の式(1) を計算するために必要なもん準備
    HP の式(1) から現（在の）加速度を求める
    速度変化＝現加速度×制御周期
    座標変化＝現速度×制御周期
    新速度=現速度+速度変化
    新座標=現座標+座標変化
    求めた新座標と新速度をnext_state に代入
    """
    State = np.atleast_2d(State)
    if State.shape[0] != 12 and State.shape[1] == 12:
        State = State.T
    # State shape: (12, N) or (12, 1)
    if State.shape[0] == 12 and State.shape[1] == 1:
        # 1サンプルだけの場合
        psi = State[8, 0]
        theta = State[7, 0]
        phi = State[6, 0]
        # 必要に応じてスカラーでRを計算
        # ... 既存のスカラー計算 ...
    elif State.shape[0] == 12:
        # 複数サンプルの場合
        psi = State[8, :]
        theta = State[7, :]
        phi = State[6, :]
        # 必要に応じてベクトルでRを計算
        # ... 既存のベクトル計算 ...
    else:
        raise ValueError('State shape is not compatible: {}'.format(State.shape))
    # 必要なパラメータをインプットしたり定義したり
    way_num = Ctrl.shape[1]
    times = P['dt']
    m = P['m'] * np.ones(way_num)
    g = P['g'] * np.ones(way_num)

    # 状態や制御入力の数字を抜き出す
    F = Ctrl[0, :]
    x = State[0, :]
    y = State[1, :]
    z = State[2, :]
    vx = State[3, :]
    vy = State[4, :]
    vz = State[5, :]
    phi = State[6, :]
    theta = State[7, :]
    psi = State[8, :]

    # HP の式(1) の計算のために，必要な数字を求めておく
    R = np.zeros((3, 3, way_num))
    R[0, 0, :] = np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta)
    R[0, 1, :] = -np.cos(phi) * np.sin(psi)
    R[0, 2, :] = np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)
    R[1, 0, :] = np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta)
    R[1, 1, :] = np.cos(phi) * np.cos(psi)
    R[1, 2, :] = np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)
    R[2, 0, :] = -np.cos(phi) * np.sin(theta)
    R[2, 1, :] = np.sin(phi)
    R[2, 2, :] = np.cos(phi) * np.cos(theta)

    K1 = np.vstack([np.zeros(way_num), np.zeros(way_num), -(m * g)])
    K2 = np.vstack([np.zeros(way_num), np.zeros(way_num), F])
    r_dd = np.zeros((3, way_num))

    # 式(1) 実行
    for i in range(way_num):
        r_dd[:, i] = (K1[:, i] + R[:, :, i] @ K2[:, i]) * (1 / m[i])

    # 加速度r_dd が分かったので，それを使って新座標と新速度を求める
    hv = r_dd * times
    h_xyz = np.vstack([vx, vy, vz]) * times

    vx = vx + hv[0, :]
    vy = vy + hv[1, :]
    vz = vz + hv[2, :]

    x = x + h_xyz[0, :]
    y = y + h_xyz[1, :]
    z = z + h_xyz[2, :]

    next_state = np.vstack([x, y, z, vx, vy, vz])
    return next_state
