import numpy as np
from numba import njit, prange
from check import check

# === Numbaによる高速計算カーネル ===
@njit(fastmath=True, parallel=True)
def compute_social_force_numba(
    agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y,  # (K,)
    obs_center_x, obs_center_y,                          # (M,) 障害物の数
    cov_inv_00, cov_inv_01, cov_inv_10, cov_inv_11,      # 逆行列の要素
    kappa_threshold,
    force_factor, force_sigma,
    lambda_importance, gamma, n, n_prime, force_factor_social, neighborhood_range,
    overlap_distance, force_factor_group, agent_radius
):
    n_samples = agent_pos_x.shape[0]
    n_obstacles = obs_center_x.shape[0]
    
    costs = np.zeros(n_samples, dtype=np.float64)

    # 外側ループを並列化（サンプルごとの計算は独立）
    for k in prange(n_samples):
        c_x = agent_pos_x[k]
        c_y = agent_pos_y[k]
        v_x = agent_vel_x[k]
        v_y = agent_vel_y[k]
        
        cost_k = 0.0
        
        for o in range(n_obstacles):
            o_x = obs_center_x[o]
            o_y = obs_center_y[o]
            
            # --- 1. マハラノビス距離チェック ---
            dx = c_x - o_x
            dy = c_y - o_y
            
            # 行列演算を展開: d.T @ cov_inv @ d
            mahalanobis_sq = (dx * cov_inv_00 + dy * cov_inv_10) * dx + \
                             (dx * cov_inv_01 + dy * cov_inv_11) * dy
            
            if mahalanobis_sq < kappa_threshold:
                risk_violation = kappa_threshold - mahalanobis_sq
                cost_k += force_factor * risk_violation * 1e8

            # --- 2. 静的な距離コスト (最短距離ベース) ---
            # ここでは障害物を点(中心)として簡易計算する場合
            # ※元のコードの「点群との最短距離」を厳密にやるなら点群データが必要ですが、
            #   Social Force計算用としては中心距離を使うのが一般的です。
            dist_sq = dx*dx + dy*dy
            dist_center = np.sqrt(dist_sq)
            
            # 表面間距離
            dist_surface = dist_center - agent_radius
            
            if dist_surface < 0:
                 cost_k += force_factor * 1e6
            else:
                 cost_k += force_factor * np.exp(-dist_surface / force_sigma) * 1e4

            # --- 3. Social Force (動的相互作用) ---
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

            # --- 4. Group Repulsion Force ---
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
    g_state[3, :] = (g_state[0, :] - nstate[0, :]) * P['speed_rate']
    g_state[4, :] = (g_state[1, :] - nstate[1, :]) * P['speed_rate']
    g_state[5, :] = (g_state[2, :] - nstate[2, :]) * P['speed_rate']
    diff_state = nstate - g_state
    # 通常コスト
    Cost = np.zeros(nstate.shape[1])
    for i in range(12):
        a = diff_state[i, :]
        Cost = Cost + P['weight'][0, i] * a ** 2

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
    
    kappa_threshold = -2 * np.log(collision_risk)
    
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
    
    # 障害物の中心座標リストを作成 (Numbaに渡すため配列化)
    # dynamic_obj shape: (2, points, n_obstacles) or (2, points)
    obs_centers = []
    if dynamic_obj.ndim == 3:
        n_obstacles = dynamic_obj.shape[2]
        for i in range(n_obstacles):
            pts = dynamic_obj[:, :, i]
            if pts.size > 0:
                obs_centers.append(pts.mean(axis=1))
    else:
        # 1つの障害物の場合
        if dynamic_obj.size > 0:
            obs_centers.append(dynamic_obj.mean(axis=1))
            
    if not obs_centers:
        return np.zeros(n_samples)
        
    obs_centers_arr = np.array(obs_centers) # Shape: (M, 2)
    
    # Numba関数を実行
    cost = compute_social_force_numba(
        nstate[0, :], nstate[1, :], nstate[3, :], nstate[4, :],  # エージェント情報
        obs_centers_arr[:, 0], obs_centers_arr[:, 1],            # 障害物情報
        cov_inv[0,0], cov_inv[0,1], cov_inv[1,0], cov_inv[1,1],  # 逆行列要素
        kappa_threshold,
        force_factor, force_sigma,
        lambda_importance, gamma, n, n_prime, force_factor_social, neighborhood_range,
        overlap_distance, force_factor_group, agent_radius
    )
    
    return cost


