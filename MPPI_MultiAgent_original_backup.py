"""
集中型マルチエージェントMPPI（MPPI_GTをベース）
状態と制御入力を連結して最適化
到着後の待機とゴースト化を実装
"""
import numpy as np
import time
from Cost_Fcn_MultiAgent import Cost_Fcn_Centralized
from Sim_Model import Sim_Model
from Term_Cost import Term_Cost
from check import check

# --- 到着後の待機とゴースト化を実装 ---
class GoalManager:
    
    # 状態定義（定数）
    ACTIVE = 0           # 通常移動中
    WAITING_IN_QUEUE = 1 # 待機列で待機中（ゴール占有中のため）
    AT_GOAL_WORKING = 2  # ゴール到着・作業中
    GHOSTED = 3          # ゴースト化済み（終了）
    
    def __init__(self, n_agents, goal_pos, wait_time=5.0, goal_threshold=0.2):
        self.n_agents = n_agents
        self.goal_pos = goal_pos  # 共通ゴール座標 (x, y)
        self.wait_time = wait_time
        self.goal_threshold = goal_threshold
        
        self.arrival_times = np.zeros(n_agents)  # 到着時刻
        self.states = np.zeros(n_agents, dtype=int) # 各エージェントの状態管理
        self.is_working = np.zeros(n_agents, dtype=bool)  # 作業中（待機中）フラグ
        self.goal_locked_by = -1  # 現在ゴールを占有しているエージェントID


def MPPI_MultiAgent(agents_data, banned_point):
    """
    集中型マルチエージェントMPPI
    """
    n_agents = len(agents_data)
    P = agents_data[0]['P']
    state_dim = P['State_dim']
    ctrl_dim = P['Ctrl_dim']
    
    # --- GoalManagerの初期化 ---
    common_goal_pos = agents_data[0]['P']['Goal_state'][0:2, 0]
    wait_time = P.get('goal_wait_time', 5.0)  # 待機時間（Load_Settingsで設定可能）
    goal_threshold = P.get('goal_threshold', 0.2)
    manager = GoalManager(n_agents, common_goal_pos, wait_time, goal_threshold)

    # 各機体の初期化
    for agent in agents_data:
        P_agent = agent['P']
        agent['trial_state'] = np.zeros((state_dim, P_agent['Trial_size']))
        agent['seq_ctrl'] = np.zeros((ctrl_dim, 1, P_agent['Trial_size'] + P_agent['Horizon_size']))
        agent['seq_ctrl'][0, 0, :] = P_agent['initial_controll'][0, 0]
        agent['seq_ctrl'][1, 0, :] = P_agent['initial_controll'][1, 0]
        agent['seq_ctrl'][2, 0, :] = P_agent['initial_controll'][2, 0]
        agent['seq_ctrl'][3, 0, :] = P_agent['initial_controll'][3, 0]
        agent['trial_state'][:, 0] = P_agent['Init_State'].flatten()
        agent['ghosted'] = False  # 完全透明化
    
    # MPPIループ
    for step in range(P['Trial_size'] - 1):
        loop_start = time.time()
        current_time = step * P['dt']
        
        # --- GoalManagerによる状態管理 ---
        for idx, agent in enumerate(agents_data):
            # 既に終了している機体はスキップ
            if agent.get('collision_occurred', False) or agent.get('ghosted', False):
                continue
            
            # 現在位置取得
            curr_pos = agent['trial_state'][0:2, step]
            dist = np.linalg.norm(curr_pos - manager.goal_pos)
            
            # 1. 到着判定 & ロック取得
            if not manager.is_working[idx]:
                # ゴールに近く、かつ「誰もロックしていない」場合
                if dist < manager.goal_threshold and manager.goal_locked_by == -1:
                    manager.goal_locked_by = idx
                    manager.is_working[idx] = True
                    manager.arrival_times[idx] = current_time
                    # ゴール位置に正確に配置（位置のみ、速度は0）
                    agent['trial_state'][0:2, step] = manager.goal_pos
                    agent['trial_state'][3:5, step] = 0.0  # xy速度を0に
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: ゴール到着・作業開始（ロック取得）")

            # 2. 作業中 & 時間経過判定
            if manager.is_working[idx]:
                elapsed = current_time - manager.arrival_times[idx]
                # 待機時間が経過したか？
                if elapsed >= manager.wait_time:
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: 作業完了・ゴースト化")
                    agent['ghosted'] = True  # 完全透明化
                    manager.goal_locked_by = -1  # ロック解放
                    manager.is_working[idx] = False

        # 連結状態を作成
        combined_state = np.vstack([agent['trial_state'][:, step].reshape(-1, 1) for agent in agents_data])
        
        # コスト正規化
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
        
        # 制御更新（待機中・終了機体を除く）
        for idx, agent in enumerate(agents_data):
            # 終了済み or ゴール作業中の機体は更新しない
            # 待機列の機体は待機位置へ移動するため制御更新は継続
            if agent.get('collision_occurred', False) or agent.get('ghosted', False):
                continue
            if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                continue
            
            ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
            for i_update in range(P['Horizon_size']):
                update = np.dot(noise_seq[ctrl_slice, :, i_update], weight)
                update = update.reshape(-1, 1)
                agent['seq_ctrl'][:, :, step + i_update] += update
        
        # 次状態計算（待機中・終了機体を除く）
        for idx, agent in enumerate(agents_data):
            # ゴースト化した機体は最後の位置で固定（0にならないように）
            if agent.get('ghosted', False):
                agent['trial_state'][:, step + 1] = agent['trial_state'][:, step].copy()
                continue
            
            # 衝突済みの機体はスキップ（既に0で埋められている）
            if agent.get('collision_occurred', False):
                continue
            
            # 待機中の機体はゴール位置で固定
            if manager.is_working[idx]:
                # ゴール位置に固定（x, y座標）
                agent['trial_state'][0:2, step + 1] = manager.goal_pos
                # z座標は現在の高度を維持
                agent['trial_state'][2, step + 1] = agent['trial_state'][2, step]
                # 速度を全て0に
                agent['trial_state'][3:6, step + 1] = 0.0
                # 角度・角速度は現在値を維持
                agent['trial_state'][6:, step + 1] = agent['trial_state'][6:, step]
                continue
            
            agent['trial_state'][:, step + 1] = Sim_Model(
                agent['trial_state'][:, step],
                agent['seq_ctrl'][:, :, step],
                agent['P']
            ).flatten()
        
        # 状態チェック
        all_finished = True
        for idx, agent in enumerate(agents_data):
            if not check_agent_status(agent, step, agents_data, manager, idx):
                all_finished = False
        
        print(f"Trial loop {step+1}/{P['Trial_size']-1} 経過時間: {time.time() - loop_start:.3f} 秒")
        
        if all_finished:
            print(f"全機体が終了（Step {step+1}）")
            break
    
    return agents_data

def check_agent_status(agent, step, agents_data, manager, agent_idx):
    """
    機体の状態チェック（衝突判定のみ）
    ※ ゴール到達判定は GoalManager が行う
    """
    P = agent['P']
    trial_state = agent['trial_state']
    
    # 既に終了している場合
    if agent.get('collision_occurred', False) or agent.get('ghosted', False):
        return True
    
    # 障害物との衝突判定
    current_time = (step + 1) * P['dt']
    fale = check(trial_state[:, step + 1], trial_state[:, step], P, current_time)
    
    if np.any(fale >= 1):
        print(f"Agent {agent['id']}: 障害物と衝突")
        agent['collision_occurred'] = True
        agent['collision_pos'] = trial_state[0:3, step].copy()
        trial_state[:, step+2:] = 0
        # ロックを持っていたら解放
        if manager.goal_locked_by == agent_idx:
            manager.goal_locked_by = -1
            manager.is_working[agent_idx] = False
        return True
    
    # エージェント間の衝突判定
    safety_distance = P.get('safety_distance', 0.7)
    my_pos = trial_state[0:2, step+1]
    
    for other_idx, other_agent in enumerate(agents_data):
        if other_agent['id'] == agent['id']:
            continue
        if other_agent.get('collision_occurred', False):
            continue
        
        # ゴースト化した機体との衝突は無視
        if other_agent.get('ghosted', False):
            continue
        
        # 待機中の機体との衝突は判定する（物理的に存在）
        
        other_pos = other_agent['trial_state'][0:2, step+1]
        distance = np.linalg.norm(my_pos - other_pos)
        
        if distance < safety_distance:
            print(f"Agent {agent['id']}: Agent {other_agent['id']}と衝突（距離: {distance:.2f}m）")
            agent['collision_occurred'] = True
            agent['collision_pos'] = trial_state[0:3, step].copy()
            trial_state[:, step+2:] = 0
            # ロックを持っていたら解放
            if manager.goal_locked_by == agent_idx:
                manager.goal_locked_by = -1
                manager.is_working[agent_idx] = False
            return True
    
    return False
