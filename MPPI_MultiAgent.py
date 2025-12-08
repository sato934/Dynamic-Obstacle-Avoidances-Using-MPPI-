"""
集中型マルチエージェントMPPI（MPPI_GTをベース）
状態と制御入力を連結して最適化
"""
import numpy as np
import time
from Cost_Fcn_Centralized import Cost_Fcn_Centralized
from Sim_Model import Sim_Model
from Term_Cost import Term_Cost
from check import check

def MPPI_MultiAgent(agents_data, banned_point):
    """
    集中型マルチエージェントMPPI
    MPPI_GTと同じロジックで、状態と制御を連結
    """
    n_agents = len(agents_data)
    P = agents_data[0]['P']
    state_dim = P['State_dim']
    ctrl_dim = P['Ctrl_dim']
    
    # 各機体の初期化
    for agent in agents_data:
        P_agent = agent['P']
        agent['trial_state'] = np.zeros((state_dim, P_agent['Trial_size']))
        agent['seq_ctrl'] = np.zeros((ctrl_dim, 1, P_agent['Trial_size'] + P_agent['Horizon_size']))
        # 初期制御入力を設定
        agent['seq_ctrl'][0, 0, :] = P_agent['initial_controll'][0, 0]
        agent['seq_ctrl'][1, 0, :] = P_agent['initial_controll'][1, 0]
        agent['seq_ctrl'][2, 0, :] = P_agent['initial_controll'][2, 0]
        agent['seq_ctrl'][3, 0, :] = P_agent['initial_controll'][3, 0]
        agent['trial_state'][:, 0] = P_agent['Init_State'].flatten()
    
    # MPPIループ
    for step in range(P['Trial_size'] - 1):
        loop_start = time.time()
        current_time = step * P['dt']
        
        # 連結状態を作成
        combined_state = np.vstack([agent['trial_state'][:, step].reshape(-1, 1) for agent in agents_data])
        
        # コスト正規化（MPPI_GTと同じ）
        cost_init = Cost_Fcn_Centralized(combined_state, combined_state, agents_data, banned_point, current_time)
        combined_init = np.vstack([agent['P']['Init_State'] for agent in agents_data])
        cost_init_baseline = Cost_Fcn_Centralized(combined_init, combined_init, agents_data, banned_point, 0)
        cost_diff = cost_init / (cost_init_baseline + 1e-6)
        cost_diff = np.clip(cost_diff, P['vll'], P['vlu'])
        
        # ノイズ分散（MPPI_GTと同じ）
        noise_var = P['var'] * cost_diff
        if noise_var.ndim == 1:
            noise_var = noise_var.reshape(-1, 1)
        
        # トラジェクトリコスト計算
        trj_cost = np.zeros(P['K'])
        sim_state = np.tile(combined_state, (1, P['K']))  # (state_dim*n_agents, K)
        
        # ノイズ系列生成（全機体分）
        combined_ctrl_dim = ctrl_dim * n_agents
        noise_seq = np.random.randn(combined_ctrl_dim, P['K'], P['Horizon_size'])
        for i_noise in range(P['Horizon_size']):
            for idx in range(n_agents):
                ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
                noise_seq[ctrl_slice, :, i_noise] = noise_var * noise_seq[ctrl_slice, :, i_noise]
        
        # 制御入力シーケンス（全機体分）
        horizon_input = np.zeros((combined_ctrl_dim, P['K'], P['Horizon_size']))
        for idx, agent in enumerate(agents_data):
            for h in range(P['Horizon_size']):
                horizon_input[idx*ctrl_dim:(idx+1)*ctrl_dim, :, h] = \
                    np.tile(agent['seq_ctrl'][:, :, step + h], (1, P['K']))
        
        # ランダムサンプリング
        horizon_input[:, :int(P['random_sample_rate']*P['K']), :] = 0
        
        # シミュレーション
        for i_sim in range(P['Horizon_size']):
            next_sim_state_list = []
            for idx, agent in enumerate(agents_data):
                agent_state = sim_state[idx*state_dim:(idx+1)*state_dim, :]
                agent_ctrl = horizon_input[idx*ctrl_dim:(idx+1)*ctrl_dim, :, i_sim] + \
                            noise_seq[idx*ctrl_dim:(idx+1)*ctrl_dim, :, i_sim]
                next_agent_state = Sim_Model(agent_state, agent_ctrl, agent['P'])
                next_sim_state_list.append(next_agent_state)
            
            next_sim_state = np.vstack(next_sim_state_list)
            sim_time = current_time + i_sim * P['dt']
            trj_cost += Cost_Fcn_Centralized(next_sim_state, sim_state, agents_data, banned_point, sim_time)
            sim_state = next_sim_state
        
        # 終端コスト
        for idx, agent in enumerate(agents_data):
            agent_sim_state = sim_state[idx*state_dim:(idx+1)*state_dim, :]
            trj_cost += Term_Cost(agent_sim_state)
        
        # 重み計算（MPPI_GTと同じ）
        min_cost = np.min(trj_cost)
        norm_cost = np.sum(np.exp(-1 / P['Temp'] * (trj_cost - min_cost)))
        weight = np.exp(-1 / P['Temp'] * (trj_cost - min_cost)) / norm_cost
        
        # 制御更新（各機体ごと、終了していない機体のみ）
        for idx, agent in enumerate(agents_data):
            # 既に終了している機体はスキップ
            if agent.get('goal_reached', False) or agent.get('collision_occurred', False):
                continue
            
            ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
            for i_update in range(P['Horizon_size']):
                update = np.dot(noise_seq[ctrl_slice, :, i_update], weight)
                update = update.reshape(-1, 1)
                agent['seq_ctrl'][:, :, step + i_update] += update
        
        # 次状態計算（各機体ごと、終了していない機体のみ）
        for agent in agents_data:
            # 既に終了している機体はスキップ（状態を固定）
            if agent.get('goal_reached', False) or agent.get('collision_occurred', False):
                continue
            
            agent['trial_state'][:, step + 1] = Sim_Model(
                agent['trial_state'][:, step],
                agent['seq_ctrl'][:, :, step],
                agent['P']
            ).flatten()
        
        # 状態チェック
        all_finished = True
        for agent in agents_data:
            if not check_agent_status(agent, step, agents_data):
                all_finished = False
        
        print(f"Trial loop {step+1}/{P['Trial_size']-1} 経過時間: {time.time() - loop_start:.3f} 秒")
        
        if all_finished:
            print(f"全機体が終了（Step {step+1}）")
            break
    
    return agents_data

def check_agent_status(agent, step, agents_data):
    """
    機体の状態チェック（MPPI_GTと同じ + エージェント間衝突判定）
    """
    P = agent['P']
    trial_state = agent['trial_state']
    
    # 既に終了している場合
    if agent.get('goal_reached', False) or agent.get('collision_occurred', False):
        return True
    
    # 目標到達判定
    distance_to_goal = np.linalg.norm(trial_state[0:3, step+1] - P['Goal_state'][0:3, 0])
    goal_threshold = P.get('goal_threshold', 0.2)
    
    if distance_to_goal <= goal_threshold:
        print(f"Agent {agent['id']}: ゴール到達！")
        agent['goal_reached'] = True
        trial_state[:, step+2:] = trial_state[:, step+1].reshape(-1, 1)
        return True
    
    # 障害物との衝突判定
    current_time = (step + 1) * P['dt']
    fale = check(trial_state[:, step + 1], trial_state[:, step], P, current_time)
    
    if np.any(fale >= 1):
        print(f"Agent {agent['id']}: 障害物と衝突")
        agent['collision_occurred'] = True
        agent['collision_pos'] = trial_state[0:3, step].copy()
        print(f"  collision_pos (1ステップ前) = {agent['collision_pos']}")
        trial_state[:, step+2:] = 0
        return True
    
    # エージェント間の衝突判定
    safety_distance = P.get('safety_distance', 0.5)  # 機体間の最小安全距離
    my_pos = trial_state[0:3, step+1]
    
    for other_agent in agents_data:
        if other_agent['id'] == agent['id']:
            continue
        if other_agent.get('collision_occurred', False):
            continue  # 既に衝突した機体は無視
        
        other_pos = other_agent['trial_state'][0:3, step+1]
        distance = np.linalg.norm(my_pos - other_pos)
        
        if distance < safety_distance:
            print(f"Agent {agent['id']}: Agent {other_agent['id']}と衝突（距離: {distance:.2f}m）")
            agent['collision_occurred'] = True
            agent['collision_pos'] = trial_state[0:3, step].copy()
            print(f"  collision_pos (1ステップ前) = {agent['collision_pos']}")
            trial_state[:, step+2:] = 0
            return True
    
    return False
