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
    P['Trial_num'] = 10 #反復回数 
    P['Horizon'] = 4 #評価区間 4
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
    P['agent_radius'] = 0.3  # 機体の半径[m]
    P['force_sigma_obstacle'] = 0.2 # 障害物力の減衰パラメータ[m] (ギリギリまで小さく)  マルチ0.2 単一0.1
    P['force_factor_obstacle'] = 250.0  # 障害物力の係数 (範囲を極小化した分、強度を大幅に上げる) マルチ250.0 単一500
    
    #3次元化のパラメータ
    P['wall_height'] = 3.0 # 壁の高さ
    P['min_height'] = 0.0 # 地面
    P['max_height'] = 3.0 # 天井
    
    # Social Forceモデルのパラメータ
    P['lambda_importance'] = 0.4  # 位置vs速度の相対的重要度 
    P['gamma'] = 0.4  # 速度相互作用パラメータ
    P['n'] = 2  # 速度相互作用の指数
    P['n_prime'] = 3  # 角度相互作用の指数
    P['force_factor_social'] = 3.0  # 力の係数 (動的障害物への反発を強化) 
    P['neighborhood_range'] = 7.0  # 近傍範囲[m] (狭い環境では短めに) 
    
    # 逆二乗反発力のパラメータ
    P['wrep'] = 1.0  # 逆二乗反発力の係数（弱めに調整）値×10^4
    P['gamma_rep'] = 1.0  # 正則化パラメータ（距離の二乗と同程度に設定）
    
    # 障害物同士 のパラメータ (意味ない)
    P['overlap_distance'] = 0.5  # 重なり距離[m] (機体半径より大きく)
    P['force_factor_group_repulsion'] = 10.0  # グループ反発力の係数 (近距離での強い反発)
    
    # マルチエージェント用パラメータ
    P['safety_distance'] = 0.6  # エージェント間の最小安全距離[m] (agent_radius * 2 以上)
    P['force_factor_inter_agent'] = 50.0  # エージェント間の力の係数（大きいほど強く回避）
    P['force_sigma_inter_agent'] = 0.7   # エージェント間の減衰パラメータ[m]（大きいほど遠くから回避）　
    P['goal_wait_time'] = 3.0  # ゴール到着後の待機時間[s]（作業時間）
    P['goal_lock_distance'] = 2.0  # ロック取得距離[m]（これ以内でロック判定）
    
    # 目標到達判定
    P['goal_threshold'] = 0.4  # 目標到達とみなす距離の閾値[m] 単一0.2　マルチ0.4
    P['near_goal_threshold'] = 1.0  # 目標近傍での速度緩和を開始する距離[m]
    
    # マハラノビス距離による衝突判定
    P['collision_risk'] = 0.1 # 許容リスク δ 1%

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
    if i == 7:  # 静止障害物で囲む（3D対応：壁は高さ3m、球体障害物）
        P['Goal_state'] = np.array([0, -5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-6, 6, -6, 6])

        # 静止障害物で囲む（3D化：z座標追加）
        # 左の壁
        wall_left_base = np.array([
            [-5, -5, -4, -4],
            [-8, 8, 8, -8]
        ])
        # 右の壁
        wall_right_base = np.array([
            [4, 4, 5, 5],
            [-8, 8, 8, -8]
        ])
        
        # 各壁にz=0を追加（底面座標）
        wall_left = np.vstack([wall_left_base, np.zeros(wall_left_base.shape[1])])
        wall_right = np.vstack([wall_right_base, np.zeros(wall_right_base.shape[1])])
        
        # 静的障害物を3次元配列に変換（左右の壁のみ）
        P['object'] = np.stack([wall_left, wall_right], axis=2)
        P['object_height'] = P['wall_height']  # 壁の高さ5mを記録

        # 複数の動的球形障害物の設定（3D対応）
        g = P['Goal_state'][:3, 0]  # ゴール座標（x,y,z）
        r = 0.3  # 球の半径
        n_points = 32  # 球面上の点の数（フィボナッチ螺旋法）

        num_spheres = 25  # 球形障害物の数 15

        sphere_list = []
        waypoints_list = []
        segment_times_list = []
        velocities = []

        # X方向にズレを持たせて重なりを避けるためのオフセット
        offsets = np.linspace(-3.0, 3.0, num_spheres)

        for idx in range(num_spheres):
            # 各障害物ごとにランダムな切り返し回数を設定
            n_waypoints = np.random.randint(3, 4)
            
            # 初期位置を分散（ゴール付近から少しずつずらす、z座標もランダム）
            z_init = np.random.uniform(0.5, 2.5)  # ランダムな初期高度（天井3mに対応）
            center = np.array([g[0] + offsets[idx], g[1], z_init])

            # 球面上の点を生成（フィボナッチ螺旋法）
            indices = np.arange(0, n_points, dtype=float) + 0.5
            phi = np.arccos(1 - 2*indices/n_points)
            theta = np.pi * (1 + 5**0.5) * indices
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            sphere_points = np.stack([x, y, z], axis=0)
            sphere_list.append(sphere_points)

            # 指定された範囲内でランダムな3D中間地点を生成（速度0.5～4.0 m/s）
            waypoints = []
            for _ in range(n_waypoints):
                xw = np.random.uniform(-3, 3) + offsets[idx] * 0.3
                yw = np.random.uniform(-3, 3)
                zw = np.random.uniform(0.5, 2.5)  # z座標もランダム（天井3mに対応）
                waypoints.append(np.array([xw, yw, zw]))
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
        # 動的障害物点群を (3, n_points, num_spheres) で格納
        P['dynamic_obj'] = np.stack(sphere_list, axis=2)
        P['dynamic_waypoints'] = waypoints_list
        P['dynamic_segment_times'] = segment_times_list
        P['dynamic_start_time'] = 0.0
        P['dynamic_end_time'] = float(P['Trial_time'])
        P['dynamic_velocity'] = velocities
        
        # 開始位置
        P['Init_State'] = np.array([0, 4, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    
    elif i == 8:  # 動的障害物のランダム初期位置（3D対応）
        P['Goal_state'] = np.array([0, -5, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        P['axis'] = np.array([-6, 6, -6, 6])
        
        start_pos = np.array([0, 4, 1.5])  # 開始位置（3D）
        
        # 静止障害物で囲む（i=7と同じ：左右の壁）
        # 左の壁
        wall_left_base = np.array([
            [-5, -5, -4, -4],
            [-10, 10, 10, -10]
        ])
        # 右の壁
        wall_right_base = np.array([
            [4, 4, 5, 5],
            [-10, 10, 10, -10]
        ])
        
        # 各壁にz=0を追加（底面座標）
        wall_left = np.vstack([wall_left_base, np.zeros(wall_left_base.shape[1])])
        wall_right = np.vstack([wall_right_base, np.zeros(wall_right_base.shape[1])])
        
        # 静的障害物を3次元配列に変換（左右の壁のみ）
        P['object'] = np.stack([wall_left, wall_right], axis=2)
        P['object_height'] = P['wall_height']  # 壁の高さを記録
        
        # 複数の動的球形障害物の設定（3D対応、初期位置ランダム）
        g = P['Goal_state'][:3, 0]  # ゴール座標（x,y,z）
        r = 0.3  # 球の半径
        n_points = 32  # 球面上の点の数（フィボナッチ螺旋法）

        num_spheres = 25  # 球形障害物の数

        sphere_list = []
        waypoints_list = []
        segment_times_list = []
        velocities = []

        for idx in range(num_spheres):
            # 各障害物ごとにランダムな切り返し回数を設定
            n_waypoints = np.random.randint(3, 4)
            
            # 初期位置をランダムに設定（開始位置付近は避ける）
            while True:
                center = np.array([
                    np.random.uniform(-3.5, 3.5),  # X座標
                    np.random.uniform(-4, 4),      # Y座標
                    np.random.uniform(0.5, 2.5)    # Z座標（天井3mに対応）
                ])
                # 開始位置から2.5m以上離れていることを確認
                dist_to_start = np.linalg.norm(center - start_pos)
                if dist_to_start > 2.5:
                    break

            # 球面上の点を生成（フィボナッチ螺旋法）
            indices = np.arange(0, n_points, dtype=float) + 0.5
            phi = np.arccos(1 - 2*indices/n_points)
            theta = np.pi * (1 + 5**0.5) * indices
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            sphere_points = np.stack([x, y, z], axis=0)
            sphere_list.append(sphere_points)

            # 指定された範囲内でランダムな3D中間地点を生成
            waypoints = []
            for _ in range(n_waypoints):
                xw = np.random.uniform(-4, 4)  # X座標
                yw = np.random.uniform(-4, 4)  # Y座標
                zw = np.random.uniform(0.5, 2.5)  # Z座標
                waypoints.append(np.array([xw, yw, zw]))
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
        # 動的障害物点群を (3, n_points, num_spheres) で格納
        P['dynamic_obj'] = np.stack(sphere_list, axis=2)
        P['dynamic_waypoints'] = waypoints_list
        P['dynamic_segment_times'] = segment_times_list
        P['dynamic_start_time'] = 0.0
        P['dynamic_end_time'] = float(P['Trial_time'])
        P['dynamic_velocity'] = velocities
        
        # 開始位置
        P['Init_State'] = np.array([0, 4, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    
    P['State_dim'] = P['Init_State'].size
    P['var2'] = np.diag(P['var2'])
    P['Q_f'] = np.diag(P['Q_f'])
    P['Ctrl_dim'] = P['var'].shape[0]

    return P
