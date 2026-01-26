import numpy as np
import random

def Load_Settings(i):
    # 乱数シード
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    P = {}
    P['seed'] = seed

    P['m'] = 1.3 
    P['g'] = 9.8
    
    P['dt'] = 0.2 #制御周期[s]　T＝1/s
    P['Trial_time'] = 40 #反復1回当たりの時間　単一20 マルチ40
    P['Trial_num'] = 3 #反復回数 3
    P['Horizon'] = 4 #評価区間 6 4
    P['K'] = 5000 #経路数（サンプル数）　1000→5000
    P['Temp'] = 0.02 #逆温度

    P['ls'] = np.logspace(0, 1, P['Trial_num'])
    vF = 0.1 #分散
    vav = 0.1
    wp = 4 #位置の重み　禁止点の重みとの関係に注意 4
    wv = 0 #速度の重み -np.inf
    wa = -np.inf #加速度の重み
    wav = -np.inf #角速度の重み
    wcf = -np.inf #衝突の重み
    wcav = -np.inf #動的障害物回避の重み

    P['wbp'] = 3 * 10**6 #禁止点の重み
    P['speed_rate'] = 0 #速度重視度合い 0:速度無視
    P['vll'] = 1 #下限　可変乱数分散は不要のため固定
    P['vlu'] = 1 #上限  
    P['random_sample_rate'] = 0
    P['bp_switch'] = 0 #禁止点の切り替え 0:オフ 1:オン
    P['check'] = 4 #ロック確認秒数．〇秒前までの経路見てロックかどうか判断
    P['initial_controll'] = np.array([[13], [0], [0], [0]])
    
    # 障害物コスト関数の選択 0:従来型  1:動的衝突回避型
    P['obstacle_cost_type'] = 1
    
    # 動的衝突回避型のパラメータ
    P['agent_radius'] = 0.3  # 機体の半径[m]　0.35
    P['force_sigma_obstacle'] = 0.15 # 障害物力の減衰パラメータ[m] (ギリギリまで小さく)  マルチ0.15 単一0.1
    P['force_factor_obstacle'] = 250.0  # 障害物力の係数 (範囲を極小化した分、強度を大幅に上げる) 250.0
    
    # Social Forceモデルのパラメータ
    P['lambda_importance'] = 0.4  # 位置vs速度の相対的重要度 2.0 
    P['gamma'] = 0.4  # 速度相互作用パラメータ
    P['n'] = 2  # 速度相互作用の指数
    P['n_prime'] = 3  # 角度相互作用の指数
    P['force_factor_social'] = 3.0  # 力の係数 (動的障害物への反発を強化) 3.0
    P['neighborhood_range'] = 7.0  # 近傍範囲[m] (狭い環境では短めに) 7.0
    
    # 逆二乗反発力のパラメータ
    P['wrep'] = 1.0  # 逆二乗反発力の係数（弱めに調整）値×10^4
    P['gamma_rep'] = 1.0  # 正則化パラメータ（距離の二乗と同程度に設定）
    
    # 障害物同士 のパラメータ (意味ない)
    P['overlap_distance'] = 0.5  # 重なり距離[m] (機体半径より大きく)
    P['force_factor_group_repulsion'] = 10.0  # グループ反発力の係数 (近距離での強い反発)
    
    # マルチエージェント用パラメータ
    P['safety_distance'] = 0.65  # エージェント間の最小安全距離[m] (agent_radius * 2 + マージン)
    P['force_factor_inter_agent'] = 25.0  # エージェント間の力の係数（大きいほど強く回避）15.0 25.0
    P['force_sigma_inter_agent'] = 0.7   # エージェント間の減衰パラメータ[m]（大きいほど遠くから回避）　0.4 0.7
    P['goal_wait_time'] = 3.0  # ゴール到着後の待機時間[s]（作業時間）
    P['goal_lock_distance'] = 2.0  # ロック取得距離[m]（これ以内でロック判定）1.0 1.5
    
    # 目標到達判定
    P['goal_threshold'] = 0.2  # 目標到達とみなす距離の閾値[m] 0.2　マルチ0.4
    P['near_goal_threshold'] = 1.0  # 目標近傍での速度緩和を開始する距離[m]
    
    # マハラノビス距離による衝突判定
    P['collision_risk'] = 0.1 # 許容リスク δ (1%)or(10%)

    # 速度依存の動的楕円パラメータ
    P['use_velocity_dependent_cov'] = True  # 速度依存の共分散行列を使用するか
    P['sigma_perpendicular'] = 0.1  # 進行方向に垂直な方向の標準偏差[m]
    P['sigma_parallel_coeff'] = 0.1  # 進行方向の標準偏差の速度係数[s]（速度1m/sあたり0.05m増加）
    P['sigma_parallel_min'] = 0.1  # 進行方向の最小標準偏差[m]

    P['var'] = np.array([vF, vav, vav, vav])
    P['var2'] = np.zeros(12)
    P['Q_f'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    P['weight'] = np.array([
        10**wp, 10**wp, 10**wp,
        10**wv, 10**wv, 10**wv,
        10**wa, 10**wa, 10**wa,
        10**wav, 10**wav, 10**wav
    ]).reshape(1, 12)
    P['weight_c'] = np.array([10**wcf, 10**wcav, 10**wcav, 10**wcav])
    P['Trial_size'] = int(P['Trial_time'] / P['dt'])
    P['Horizon_size'] = int(P['Horizon'] / P['dt'])
    P['Dataset_size'] = P['Trial_size'] * P['Trial_num']
    P['var_mat'] = np.diag(P['var'])
    P['var2'] = np.diag(P['var2'])
    P['Q_f'] = np.diag(P['Q_f'])
    P['Ctrl_dim'] = P['var'].shape[0]

    # 障害物・目標の設定
    if i == 0:  # 障害物なし
        P['Goal_state'] = np.array([0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-8, 8, -2, 8])
        # 静的障害物は無し（空配列）
        P['object'] = np.zeros((2, 0, 1))
    elif i == 1:  #コの字型
        P['object'] = np.array([
            [-3, -3, 3, 3, 2, 2, -2, -2],
            [7, 2, 2, 7, 7, 3, 3, 7]
        ])
        # 2次元なら3次元に変換
        if P['object'].ndim == 2:
            P['object'] = np.expand_dims(P['object'], axis=2)
        P['Goal_state'] = np.array([0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-8, 8, -2, 10])
    elif i == 2:  #直方体
        P['object'] = np.array([
            [-1, -1, 1, 1],
            [1, 4, 4, 1]
        ])
        # 2次元なら3次元に変換
        if P['object'].ndim == 2:
            P['object'] = np.expand_dims(P['object'], axis=2)
        P['Goal_state'] = np.array([0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-5, 5, -2, 7])
    elif i == 5:  #移動障害物（円形）: スタートと目標の間をランダムに移動   11/10追加
        # 目標位置（例）
        P['Goal_state'] = np.array([0, -4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-8, 8, -6, 6])

        # 静的障害物は無し（空配列）
        P['object'] = np.zeros((2, 0, 1))

        # 動的障害物の設定
        g = P['Goal_state'][:2, 0]  # ゴール座標
        s = np.array([0.0, 0.0])    # スタート座標
        r = 0.3  # 円の半径
        n_points = 16  # 円周上の点の数

        # 指定された範囲内でランダムな中間地点を生成
        n_waypoints = np.random.randint(2, 5)  
        waypoints = []
        for _ in range(n_waypoints):
            # X座標: -3から3の範囲
            x = np.random.uniform(-3, 3)
            # Y座標: 3から-3の範囲
            y = np.random.uniform(-3, 3)
            waypoints.append(np.array([x, y]))
        waypoints = np.array(waypoints)
        
        # 円周上の点を生成（時計回り）
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = g[0] + r * np.cos(theta)  # 初期位置はゴール付近
        y = g[1] + r * np.sin(theta)
        
        # 円形の動的障害物を表す配列（2 x n_points x 1）形式
        circle_points = np.stack([x, y], axis=0)
        P['dynamic'] = True
        P['dynamic_obj'] = np.expand_dims(circle_points, axis=2)
        
        # 各ウェイポイントでの所要時間（より遅い動きのために時間を長く設定）
        segment_times = np.random.uniform(20.0, 30.0, n_waypoints)  
        segment_times = segment_times * (P['Trial_time'] / segment_times.sum())
        
        # 時刻tに応じた位置・速度計算用の情報を保存
        P['dynamic_waypoints'] = waypoints
        P['dynamic_segment_times'] = segment_times
        P['dynamic_start_time'] = 0.0
        P['dynamic_end_time'] = float(P['Trial_time'])
        
        # 初期速度（最初のウェイポイントに向かう）
        first_vel = (waypoints[0] - g) / segment_times[0]
        P['dynamic_velocity'] = first_vel
        
        # 開始位置を変更
        P['Init_State'] = np.array([0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    elif i == 7:  # 静止障害物で囲む　
        P['Goal_state'] = np.array([0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-8, 8, -6, 6])

        # 静止障害物で囲む
        # 上の壁
        wall_top = np.array([
            [-5, -5, 5, 5],
            [6, 7, 7, 6]
        ])
        # 下の壁
        wall_bottom = np.array([
            [-6, -6, 6, 6],
            [-9, -8, -8, -9]
        ])
        # 左の壁
        wall_left = np.array([
            [-5, -5, -4, -4],
            [-8, 8, 8, -8]
        ])
        # 右の壁
        wall_right = np.array([
            [4, 4, 5, 5],
            [-8, 8, 8, -8]
        ])
        
        # 静的障害物を3次元配列に変換
        P['object'] = np.stack([wall_top, wall_bottom, wall_left, wall_right], axis=2)

        # 複数の動的円形障害物の設定
        g = P['Goal_state'][:2, 0]  # ゴール座標
        r = 0.3  # 円の半径
        n_points = 64  # 円周上の点の数

        num_circles = 15  # 円形障害物の数

        circle_list = []
        waypoints_list = []
        segment_times_list = []
        velocities = []

        # X方向にズレを持たせて重なりを避けるためのオフセット
        offsets = np.linspace(-3.0, 3.0, num_circles) #-2 2

        for idx in range(num_circles):
            # 各障害物ごとにランダムな切り返し回数を設定
            n_waypoints = np.random.randint(3, 4)
            
            # 初期位置を分散（ゴール付近から少しずつずらす）
            center = g + np.array([offsets[idx], 0.0])

            # 円周上の点を生成（時計回り）
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            circle_points = np.stack([x, y], axis=0)
            circle_list.append(circle_points)

            # 指定された範囲内でランダムな中間地点を生成（速度0.5～4.0 m/s）
            waypoints = []
            for _ in range(n_waypoints):
                xw = np.random.uniform(-3, 3) + offsets[idx] * 0.3 #-3 3 -4 4
                yw = np.random.uniform(-3, 3)
                waypoints.append(np.array([xw, yw]))
            waypoints = np.array(waypoints)
            waypoints_list.append(waypoints)

            # 各ウェイポイントでの所要時間（速度を0.5～4.0 m/sに制御）
            segment_times = np.random.uniform(1.5, 3.0, n_waypoints)
            segment_times = segment_times * (P['Trial_time'] / segment_times.sum())
            segment_times_list.append(segment_times)

            # 初期速度（最初のウェイポイントに向かう）
            first_vel = (waypoints[0] - center) / segment_times[0]
            velocities.append(first_vel)

        P['dynamic'] = True
        # 動的障害物点群を (2, n_points, num_circles) で格納
        P['dynamic_obj'] = np.stack(circle_list, axis=2)
        P['dynamic_waypoints'] = waypoints_list
        P['dynamic_segment_times'] = segment_times_list
        P['dynamic_start_time'] = 0.0
        P['dynamic_end_time'] = float(P['Trial_time'])
        P['dynamic_velocity'] = velocities
        
        # 開始位置を変更
        P['Init_State'] = np.array([0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    elif i == 8:  # 動的障害物のランダム初期位置　
        P['Goal_state'] = np.array([0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-8, 8, -6, 6])
        
        start_pos = np.array([0, 4])  # 開始位置

        # 静止障害物で囲む  
        # 上の壁
        wall_top = np.array([
            [-5, -5, 5, 5],
            [6, 7, 7, 6]
        ])
        # 下の壁
        wall_bottom = np.array([
            [-5, -5, 5, 5],
            [-8, -7, -7, -8]
        ])
        # 左の壁
        wall_left = np.array([
            [-5, -5, -4, -4],
            [-6, 6, 6, -6]
        ])
        # 右の壁
        wall_right = np.array([
            [4, 4, 5, 5],
            [-6, 6, 6, -6]
        ])
        
        # 静的障害物を3次元配列に変換
        P['object'] = np.stack([wall_top, wall_bottom, wall_left, wall_right], axis=2)

        # 複数の動的円形障害物の設定（初期位置をランダムに）
        g = P['Goal_state'][:2, 0]  # ゴール座標
        r = 0.3  # 円の半径
        n_points = 64  # 円周上の点の数

        num_circles = 15  # 円形障害物の数

        circle_list = []
        waypoints_list = []
        segment_times_list = []
        velocities = []

        for idx in range(num_circles):
            # 各障害物ごとにランダムな切り返し回数を設定
            n_waypoints = np.random.randint(3, 4)
            
            # 初期位置をランダムに設定（開始位置付近は避ける）
            while True:
                center = np.array([
                    np.random.uniform(-3.5, 3.5), # マルチ　all -4
                    np.random.uniform(-4, 4)  #　単一　-4 -4 -5 -5
                ])
                # 開始位置から2.5m以上離れていることを確認
                dist_to_start = np.linalg.norm(center - start_pos)
                if dist_to_start > 2.5:
                    break

            # 円周上の点を生成（時計回り）
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            circle_points = np.stack([x, y], axis=0)
            circle_list.append(circle_points)

            # 指定された範囲内でランダムな中間地点を生成
            waypoints = []
            for _ in range(n_waypoints):
                xw = np.random.uniform(-4, 4) #　マルチ　-3 -3 -4 -4
                yw = np.random.uniform(-4, 4) #　単一　-4 -4 -5 -5
                waypoints.append(np.array([xw, yw]))
            waypoints = np.array(waypoints)
            waypoints_list.append(waypoints)

            # 各ウェイポイントでの所要時間
            segment_times = np.random.uniform(1.5, 3.0, n_waypoints)  
            segment_times = segment_times * (P['Trial_time'] / segment_times.sum())
            segment_times_list.append(segment_times)

            # 初期速度（最初のウェイポイントに向かう）
            first_vel = (waypoints[0] - center) / segment_times[0]
            velocities.append(first_vel)

        P['dynamic'] = True
        # 動的障害物点群を (2, n_points, num_circles) で格納
        P['dynamic_obj'] = np.stack(circle_list, axis=2)
        P['dynamic_waypoints'] = waypoints_list
        P['dynamic_segment_times'] = segment_times_list
        P['dynamic_start_time'] = 0.0
        P['dynamic_end_time'] = float(P['Trial_time'])
        P['dynamic_velocity'] = velocities
        
        # 開始位置を変更
        P['Init_State'] = np.array([0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    
    # デフォルトの開始位置（i == 5, 7, 8以外）
    if i not in [5, 7, 8]:
        P['Init_State'] = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    
    P['State_dim'] = P['Init_State'].size
    P['var2'] = np.diag(P['var2'])
    P['Q_f'] = np.diag(P['Q_f'])
    P['Ctrl_dim'] = P['var'].shape[0]

    return P
