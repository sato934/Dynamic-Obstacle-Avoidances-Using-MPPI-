"""
集中型マルチエージェント用コスト関数 (v5 - 2D版)
- エージェント間距離コストを 2D (x, y) で計算
- 全機体が対称的に衝突回避を行う
"""
import numpy as np
from numba import njit, prange
from Cost_Fcn import Cost_Fcn

@njit(fastmath=True, parallel=True)
def compute_inter_agent_distance_cost_numba(
    positions_x, positions_y, 
    finished_flags, 
    n_agents, n_samples,
    safety_distance, force_factor, force_sigma
):
    """
    エージェント間の距離コストをNumbaで高速計算 (2D)
    """
    costs = np.zeros(n_samples, dtype=np.float64)
    
    # サンプルごとに並列化
    for k in prange(n_samples):
        cost_k = 0.0
        
        # 全機体ペアについて
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # どちらかがゴースト化済みなら衝突コストを計算しない
                if finished_flags[i] or finished_flags[j]:
                    continue 
                
                dx = positions_x[i, k] - positions_x[j, k]
                dy = positions_y[i, k] - positions_y[j, k]
                
                # 2D距離
                dist = np.sqrt(dx*dx + dy*dy)
                dist_surface = dist - safety_distance
                
                if dist_surface < 0:
                    # 衝突時はペナルティ大
                    cost_k += force_factor * 1e6
                else:
                    # 接近時は指数関数的な反発コスト
                    cost_k += force_factor * np.exp(-dist_surface / force_sigma) * 1e4
        
        costs[k] = cost_k
    
    return costs

def Cost_Fcn_Centralized(combined_nstate, combined_state, agents_data, banned_point, t=None):
    """
    集中型コスト関数 (2D)
    """
    n_agents = len(agents_data)
    state_dim = agents_data[0]['P']['State_dim'] 
    
    n_samples = combined_nstate.shape[1]
    total_cost = np.zeros(n_samples)
    
    # 1. 各機体の個別コスト(既存のCost_Fcnを使用)
    for idx, agent in enumerate(agents_data):
        agent_nstate = combined_nstate[idx*state_dim:(idx+1)*state_dim, :]
        agent_state = combined_state[idx*state_dim:(idx+1)*state_dim, :]
        agent_cost = Cost_Fcn(agent_nstate, agent_state, agent['P'], banned_point, t)
        total_cost += agent_cost
    
    # 2. エージェント間距離コスト(Numba高速化版)
    P = agents_data[0]['P']
    agent_radius = P.get('agent_radius', 0.35)
    force_factor = P.get('force_factor_inter_agent', 10.0)
    force_sigma = P.get('force_sigma_inter_agent', 0.8)
    safety_distance = agent_radius * 2.0
    
    # 全機体の位置を抽出 (n_agents, n_samples)
    positions_x = np.zeros((n_agents, n_samples))
    positions_y = np.zeros((n_agents, n_samples))
    
    for i in range(n_agents):
        # 状態ベクトルの 0, 1番目が x, y であると仮定
        positions_x[i, :] = combined_nstate[i*state_dim, :]       # x
        positions_y[i, :] = combined_nstate[i*state_dim+1, :]     # y
    
    # 終了済みフラグを抽出
    finished_flags = np.array([
        agent.get('_temp_finished_flag', 
                  agent.get('ghosted', False) or agent.get('collision_occurred', False))
        for agent in agents_data
    ], dtype=np.bool_)
    
    # Numba関数を実行
    inter_agent_cost = compute_inter_agent_distance_cost_numba(
        positions_x, positions_y,
        finished_flags,
        n_agents, n_samples,
        safety_distance, force_factor, force_sigma
    )
    
    total_cost += inter_agent_cost
    
    return total_cost