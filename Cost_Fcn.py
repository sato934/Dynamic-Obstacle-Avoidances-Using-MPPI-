import numpy as np
from numba import njit, prange
from check import check

@njit(fastmath=True, parallel=True)
def compute_social_force_numba(
    agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y,  # (K,)
    obs_points_x, obs_points_y,  # (M, N) 各障害物の全点群
    obs_point_counts,             # (M,) 各障害物の有効点数
    cov_inv_00, cov_inv_01, cov_inv_10, cov_inv_11,
    kappa_threshold, force_factor, force_sigma,
    lambda_importance, gamma, n, n_prime, force_factor_social, 
    neighborhood_range, overlap_distance, force_factor_group, agent_radius
):
    n_samples = agent_pos_x.shape[0]
    n_obstacles = obs_points_x.shape[0]
    
    costs = np.zeros(n_samples, dtype=np.float64)

            # --- 1.　動的衝突確率チェックコスト計算 ---
    for k in prange(n_samples):
        c_x = agent_pos_x[k]
        c_y = agent_pos_y[k]
        v_x = agent_vel_x[k]
        v_y = agent_vel_y[k]
        
        cost_k = 0.0
        
        for o in range(n_obstacles):
            n_points = obs_point_counts[o]
            
            # 障害物の中心を計算
            o_center_x = 0.0
            o_center_y = 0.0
            for p in range(n_points):
                o_center_x += obs_points_x[o, p]
                o_center_y += obs_points_y[o, p]
            o_center_x /= n_points
            o_center_y /= n_points
            
            # マハラノビス距離は中心との距離で計算
            dx = c_x - o_center_x
            dy = c_y - o_center_y
            mahalanobis_sq = (dx * cov_inv_00 + dy * cov_inv_10) * dx + \
                             (dx * cov_inv_01 + dy * cov_inv_11) * dy
            
            if mahalanobis_sq < kappa_threshold:
                risk_violation = kappa_threshold - mahalanobis_sq
                cost_k += force_factor * risk_violation * 1e8

            # --- 2. 最短距離計算（全点との距離を計算） ---
            min_dist_sq = 1e10
            for p in range(n_points):
                dx_p = c_x - obs_points_x[o, p]
                dy_p = c_y - obs_points_y[o, p]
                dist_sq = dx_p*dx_p + dy_p*dy_p
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
            
            min_dist = np.sqrt(min_dist_sq)
            dist_surface = min_dist - agent_radius
            
            if dist_surface < 0:
                 cost_k += force_factor * 1e6
            else:
                 cost_k += force_factor * np.exp(-dist_surface / force_sigma) * 1e4

            # --- 3. Social Force（中心間距離を使用） ---
            dist_center = np.sqrt(dx*dx + dy*dy)
            # 近傍範囲チェック
            if dist_center > neighborhood_range:
                continue
                
            # 正規化方向ベクトル (diff_direction)
            if dist_center > 1e-6:
                n_x = dx / dist_center
                n_y = dy / dist_center
            else:
                n_x = 0.0; n_y = 0.0
            
            # 相対速度 (障害物の速度は0と仮定しているため agent_vel そのまま)
            dv_x = v_x
            dv_y = v_y
            
            # 相互作用ベクトル: lambda * vel_diff + diff_direction
            iv_x = lambda_importance * dv_x + n_x
            iv_y = lambda_importance * dv_y + n_y
            
            iv_len = np.sqrt(iv_x*iv_x + iv_y*iv_y)
            
            if iv_len > 1e-6:
                id_x = iv_x / iv_len
                id_y = iv_y / iv_len
            else:
                id_x = 0.0; id_y = 0.0
                
            # 内積と角度 (interaction_direction . diff_direction)
            dot_prod = id_x * n_x + id_y * n_y
            if dot_prod > 1.0: dot_prod = 1.0
            elif dot_prod < -1.0: dot_prod = -1.0
            theta = np.arccos(dot_prod)
            
            # クロス積の符号 (2D cross product: ax*by - ay*bx)
            cross_val = id_x * n_y - id_y * n_x
            theta_sign = 1.0 if cross_val >= 0 else -1.0
            
            # Force計算
            B = gamma * iv_len
            if B > 1e-6:
                term_common = -dist_center / B
                f_vel = -np.exp(term_common - (n_prime * B * theta)**2)
                f_ang = -theta_sign * np.exp(term_common - (n * B * theta)**2)
                
                force_mag = np.abs(f_vel) + np.abs(f_ang)
                cost_k += force_factor_social * force_mag * 1e4

            # --- 4. 障害物同士の反発力（意味ない） ---
            if dist_center < overlap_distance and dist_center > 1e-6:
                rep_mag = (overlap_distance - dist_center) / overlap_distance
                cost_k += force_factor_group * rep_mag * 1e4
                
        costs[k] = cost_k
        
    return costs

def Cost_Fcn(nstate, state, P, banned_point, t=None):
    # nstateを必ず2次元配列に変換（shape: [状態数, サンプル数]）
    nstate = np.atleast_2d(nstate)
    if nstate.shape[0] != P['Goal_state'].shape[0]:
        nstate = nstate.T
    # stateも必ず2次元配列に変換
    state = np.atleast_2d(state)
    if state.shape[0] != P['Goal_state'].shape[0]:
        state = state.T
    # 目標状態を複製
    g_state = np.tile(P['Goal_state'], (1, nstate.shape[1]))
    
    # 目標までの距離を計算
    goal_distance = np.sqrt(
        (g_state[0, :] - nstate[0, :]) ** 2 + 
        (g_state[1, :] - nstate[1, :]) ** 2 + 
        (g_state[2, :] - nstate[2, :]) ** 2
    )
    
    # 目標近傍判定の閾値（例：1.0m）
    near_goal_threshold = P.get('near_goal_threshold', 1.0)
    
    # 遠い場合：目標方向への速度を設定
    # 近い場合：速度は小さくてもよい（位置のコストを重視）
    speed_factor = np.where(goal_distance > near_goal_threshold, P['speed_rate'], P['speed_rate'] * 0.3)
    
    g_state[3, :] = (g_state[0, :] - nstate[0, :]) * speed_factor
    g_state[4, :] = (g_state[1, :] - nstate[1, :]) * speed_factor
    g_state[5, :] = (g_state[2, :] - nstate[2, :]) * speed_factor
    
    diff_state = nstate - g_state
    
    # 通常コスト（位置のコストを目標近傍で強化）
    Cost = np.zeros(nstate.shape[1])
    for i in range(12):
        a = diff_state[i, :]
        weight = P['weight'][0, i]
        # 位置のコスト（0,1,2番目）は目標近傍で増加
        if i < 3:
            weight = weight * np.where(goal_distance < near_goal_threshold, 2.0, 1.0)
        Cost = Cost + weight * a ** 2

    # 禁止点コスト
    banned_valid = ~np.isnan(banned_point).all(axis=0)
    banned_count = np.sum(banned_valid)
    if banned_count >= 1:
        for i, valid in enumerate(banned_valid):
            if not valid:
                continue
            diff_banned = nstate[0:3, :] - banned_point[:, i].reshape(3, 1)
            # P['K']はサンプル数
            sum_sq = np.sum(diff_banned ** 2, axis=0)
            Cost = Cost + P['wbp'] / sum_sq

    # 衝突・着地ペナルティ（切り替え式）
    obstacle_cost_type = P.get('obstacle_cost_type', 0)
    
    if obstacle_cost_type == 0:
        # 従来型
        fale = check(nstate, state, P, t)
        Cost = Cost + fale * 1e17
    else:
        # 動的衝突回避型
        obstacle_cost = calculate_social_force_cost(nstate, state, P, t)
        Cost = Cost + obstacle_cost
    
    return Cost

# === 呼び出し側の修正関数 ===
def calculate_social_force_cost(nstate, state, P, t):
    # nstate: (state_dim, n_samples)
    n_samples = nstate.shape[1]
    
    # パラメータ取得 (辞書アクセスはループ外で1回だけやる)
    agent_radius = P.get('agent_radius', 0.35)
    force_sigma = P.get('force_sigma_obstacle', 0.8)
    force_factor = P.get('force_factor_obstacle', 10.0)
    lambda_importance = P.get('lambda_importance', 2.0)
    gamma = P.get('gamma', 0.35)
    n = P.get('n', 2)
    n_prime = P.get('n_prime', 3)
    force_factor_social = P.get('force_factor_social', 2.1)
    neighborhood_range = P.get('neighborhood_range', 10.0)
    overlap_distance = P.get('overlap_distance', 0.5)
    force_factor_group = P.get('force_factor_group_repulsion', 5.0)
    collision_risk = P.get('collision_risk', 0.05)
    position_cov = P.get('position_covariance', np.eye(2) * 0.1**2)
    
    # κ の計算 
    # κ = -2 ln(η_c * δ / A_r)
    # η_c = 1/(2π * sqrt(det(Σ_c))) : 2次元正規分布の正規化定数
    det_cov = np.linalg.det(position_cov)
    eta_c = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    agent_area = P.get('agent_area', np.pi * agent_radius**2)
    kappa_threshold = -2 * np.log(eta_c * collision_risk / agent_area)
    
    # 逆行列計算 (Try-Exceptは重いのでNumbaの外で)
    try:
        cov_inv = np.linalg.inv(position_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(2) / 0.01

    # 動的障害物の準備
    dynamic_obj = P.get('current_dynamic_obj', P.get('dynamic_obj'))
    
    # 障害物がない場合
    if dynamic_obj is None:
        return np.zeros(n_samples)
    
    # 障害物の点群データを準備（2次元配列形式）
    # dynamic_obj shape: (2, points, n_obstacles) or (2, points)
    obs_points_list_x = []
    obs_points_list_y = []
    obs_points_count_list = []
    
    if dynamic_obj.ndim == 3:
        n_obstacles = dynamic_obj.shape[2]
        max_points = dynamic_obj.shape[1]
        for i in range(n_obstacles):
            pts = dynamic_obj[:, :, i]  # (2, n_points)
            if pts.size > 0:
                obs_points_list_x.append(pts[0, :])
                obs_points_list_y.append(pts[1, :])
                obs_points_count_list.append(pts.shape[1])
    else:
        # 1つの障害物の場合
        if dynamic_obj.size > 0:
            obs_points_list_x.append(dynamic_obj[0, :])
            obs_points_list_y.append(dynamic_obj[1, :])
            obs_points_count_list.append(dynamic_obj.shape[1])
            max_points = dynamic_obj.shape[1]
            
    if not obs_points_list_x:
        return np.zeros(n_samples)
    
    # 2次元配列に変換（各障害物の点群を行に配置）
    # パディングして全て同じ長さにする
    max_points = max(len(pts) for pts in obs_points_list_x)
    n_obstacles = len(obs_points_list_x)
    
    obs_points_x_arr = np.zeros((n_obstacles, max_points), dtype=np.float64)
    obs_points_y_arr = np.zeros((n_obstacles, max_points), dtype=np.float64)
    
    for i in range(n_obstacles):
        n_pts = len(obs_points_list_x[i])
        obs_points_x_arr[i, :n_pts] = obs_points_list_x[i]
        obs_points_y_arr[i, :n_pts] = obs_points_list_y[i]
    
    obs_points_count_arr = np.array(obs_points_count_list, dtype=np.int64)
    
    # Numba関数を実行
    cost = compute_social_force_numba(
        nstate[0, :], nstate[1, :], nstate[3, :], nstate[4, :],  # エージェント情報
        obs_points_x_arr, obs_points_y_arr, obs_points_count_arr,  # 点群データ（2次元）
        cov_inv[0,0], cov_inv[0,1], cov_inv[1,0], cov_inv[1,1],  # 逆行列要素
        kappa_threshold,
        force_factor, force_sigma,
        lambda_importance, gamma, n, n_prime, force_factor_social, neighborhood_range,
        overlap_distance, force_factor_group, agent_radius
    )
    
    return cost


