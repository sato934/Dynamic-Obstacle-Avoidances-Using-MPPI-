"""
集中型マルチエージェント用コスト関数
既存のCost_Fcnを使用し、エージェント間距離コストのみ追加
"""
import numpy as np
from numba import njit, prange
from Cost_Fcn import Cost_Fcn

@njit(fastmath=True, parallel=True)
def compute_queue_constraint_cost_numba(
    positions_x, positions_y,
    goal_x, goal_y,
    priorities,
    finished_flags,  # ゴール済みまたは衝突済みフラグ
    n_agents, n_samples,
    queue_penalty_factor,
    queue_margin  
):
    """
    順序制約コスト：優先度の高い機体がゴールに近いべき
    priority[i] < priority[j] なら、機体iは機体jよりゴールに近いべき
    ゴール済みまたは衝突済みの機体は順序制約の対象外
    """
    costs = np.zeros(n_samples, dtype=np.float64)
    
    for k in prange(n_samples):
        cost_k = 0.0
        
        # ゴールまでの距離
        distances = np.zeros(n_agents, dtype=np.float64)
        for i in range(n_agents):
            dx = positions_x[i, k] - goal_x
            dy = positions_y[i, k] - goal_y
            distances[i] = np.sqrt(dx*dx + dy*dy)
        
        # 優先度順にソート
        sorted_indices = np.argsort(priorities)
        
        # 追い越し禁止チェック（マージン付き）
        for idx in range(len(sorted_indices) - 1):
            i = sorted_indices[idx]      # 優先度の高い機体
            j = sorted_indices[idx + 1]  # 次の優先度の機体
            
            # どちらかが終了済み（ゴールまたは衝突）なら順序制約を適用しない
            if finished_flags[i] or finished_flags[j]:
                continue
            
            # 機体jは、機体iより「margin以上」遠くにいるべき
            required_distance = distances[i] + queue_margin
            if distances[j] < required_distance:
                violation = required_distance - distances[j]
                cost_k += queue_penalty_factor * violation * 1e3
        
        costs[k] = cost_k
    
    return costs

@njit(fastmath=True, parallel=True)
def compute_inter_agent_distance_cost_numba(
    positions_x, positions_y,  # (n_agents, n_samples)
    finished_flags,  # (n_agents,) ゴール済みまたは衝突済みフラグ
    n_agents, n_samples,
    safety_distance, force_factor, force_sigma
):
    """
    エージェント間の距離コストをNumbaで高速計算
    静的障害物の距離コストと同じ方式
    どちらかが終了済み（ゴールまたは衝突）の場合は衝突コストを計算しない
    """
    costs = np.zeros(n_samples, dtype=np.float64)
    
    # サンプルごとに並列化
    for k in prange(n_samples):
        cost_k = 0.0
        
        # 全機体ペアについて
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # どちらかが終了済み（ゴールまたは衝突）なら衝突コストを計算しない
                if finished_flags[i] or finished_flags[j]:
                    continue  # ゴースト化：衝突コスト = 0
                
                dx = positions_x[i, k] - positions_x[j, k]
                dy = positions_y[i, k] - positions_y[j, k]
                
                dist = np.sqrt(dx*dx + dy*dy)
                dist_surface = dist - safety_distance
                
                if dist_surface < 0:
                    cost_k += force_factor * 1e6
                else:
                    cost_k += force_factor * np.exp(-dist_surface / force_sigma) * 1e4
        
        costs[k] = cost_k
    
    return costs

def Cost_Fcn_Centralized(combined_nstate, combined_state, agents_data, banned_point, t=None):
    """
    集中型コスト関数
    = 各機体の個別コスト（既存のCost_Fcn） + エージェント間距離コスト
    """
    n_agents = len(agents_data)
    state_dim = 12
    n_samples = combined_nstate.shape[1]
    total_cost = np.zeros(n_samples)
    
    # 1. 各機体の個別コスト（既存のCost_Fcnを使用）
    for idx, agent in enumerate(agents_data):
        agent_nstate = combined_nstate[idx*state_dim:(idx+1)*state_dim, :]
        agent_state = combined_state[idx*state_dim:(idx+1)*state_dim, :]
        agent_cost = Cost_Fcn(agent_nstate, agent_state, agent['P'], banned_point, t)
        total_cost += agent_cost
    
    # 2. エージェント間距離コスト（Numba高速化版）
    P = agents_data[0]['P']
    agent_radius = P.get('agent_radius', 0.35)
    force_factor = P.get('force_factor_inter_agent', 10.0)
    force_sigma = P.get('force_sigma_inter_agent', 0.8)
    safety_distance = agent_radius * 2.0
    
    # 全機体の位置を抽出 (n_agents, n_samples)
    positions_x = np.zeros((n_agents, n_samples))
    positions_y = np.zeros((n_agents, n_samples))
    
    for i in range(n_agents):
        positions_x[i, :] = combined_nstate[i*state_dim, :]      # x座標
        positions_y[i, :] = combined_nstate[i*state_dim+1, :]    # y座標
    
    # 終了済みフラグを抽出（ゴール済みまたは衝突済み）
    finished_flags = np.array([
        agent.get('goal_reached', False) or agent.get('collision_occurred', False) 
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
    
    # 3. 順序制約コスト（共通ゴールの場合のみ）
    if P.get('enable_queue_constraint', True):
        # 共通ゴール座標
        goal_state = agents_data[0]['P']['Goal_state']
        goal_x = goal_state[0, 0]
        goal_y = goal_state[1, 0]
        
        # 各機体の優先度
        priorities = np.array([agent.get('priority', agent['id']) for agent in agents_data], dtype=np.int64)
        
        queue_penalty_factor = P.get('queue_penalty_factor', 50.0)
        queue_margin = P.get('queue_margin', 1.0)
        
        queue_cost = compute_queue_constraint_cost_numba(
            positions_x, positions_y,
            goal_x, goal_y,
            priorities,
            finished_flags,  # 終了済みフラグを渡す
            n_agents, n_samples,
            queue_penalty_factor,
            queue_margin
        )
        
        total_cost += queue_cost
    
    return total_cost
