
"""集中型マルチエージェントMPPI (3D版 - エラー修正済み)
- 3D (x, y, z) で計算
- 型エラー対策: dtype=np.float64, .astype(np.float64) を維持
- 形状エラー対策: .reshape() による次元整合を維持
- Weightブースト & 制御リセット & 無敵判定 を維持
"""
import numpy as np
import time
from Cost_Fcn_MultiAgent import Cost_Fcn_Centralized
from Sim_Model import Sim_Model
from Term_Cost import Term_Cost
from check import check

# --- ゴール管理クラス ---
class GoalManager:
    ACTIVE = 0          # 通常移動中(ロック未取得) 
    AT_GOAL_WORKING = 1  # ゴール到着・作業中(ロック取得中)
    GHOSTED = 2          # ゴースト化済み(終了)
    
    def __init__(self, n_agents, goal_pos, wait_time=5.0, goal_threshold=0.2):
        self.n_agents = n_agents
        self.goal_pos = goal_pos # (x, y, z) 3D
        self.wait_time = wait_time
        self.goal_threshold = goal_threshold
        
        self.arrival_times = np.zeros(n_agents)
        self.lock_acquired_times = np.full(n_agents, -1.0)  # ロック取得時刻
        self.ghost_times = np.full(n_agents, -1.0)  # ゴースト化時刻
        self.states = np.zeros(n_agents, dtype=int)
        self.goal_locked_by = -1

# --------------------------------

def MPPI_MultiAgent(agents_data, banned_point):
    n_agents = len(agents_data)
    P = agents_data[0]['P']
    state_dim = P['State_dim']
    ctrl_dim = P['Ctrl_dim']
    
    # (x, y, z) 3次元取得
    common_goal_pos = agents_data[0]['P']['Goal_state'][0:3, 0]
    wait_time = P.get('goal_wait_time', 5.0)
    goal_threshold = P.get('goal_threshold', 0.2)
    lock_distance = P.get('goal_lock_distance', 1.0)
    
    manager = GoalManager(n_agents, common_goal_pos, wait_time, goal_threshold)

    # --- 各機体の初期化 ---
    for agent in agents_data:
        P_agent = agent['P']
        
        # ★【型エラー対策】float64を指定
        agent['trial_state'] = np.zeros((state_dim, P_agent['Trial_size']), dtype=np.float64)
        agent['seq_ctrl'] = np.zeros((ctrl_dim, 1, P_agent['Trial_size'] + P_agent['Horizon_size']), dtype=np.float64)
        
        for i in range(4):
            agent['seq_ctrl'][i, 0, :] = P_agent['initial_controll'][i, 0]
        
        agent['trial_state'][:, 0] = P_agent['Init_State'].flatten()
        agent['ghosted'] = False 
        
        if 'original_goal' not in agent:
            # 【3D】0:3
            agent['original_goal'] = P_agent['Goal_state'][0:3, 0].copy()
        if 'original_max_approach_speed' not in agent:
            agent['original_max_approach_speed'] = P_agent.get('max_approach_speed', 2.0)
        
        if 'original_weight' not in agent:
            agent['original_weight'] = P_agent['weight'].copy()
    
    # --- MPPIループ ---
    for step in range(P['Trial_size'] - 1):
        current_time = step * P['dt']
        loop_start = time.time()
        # =================================================================
        # 1. GoalManager & 意思決定
        # =================================================================
        
        # [Step A] 作業完了判定
        for idx, agent in enumerate(agents_data):
            if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                elapsed = current_time - manager.arrival_times[idx]
                if elapsed >= manager.wait_time:
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: 作業完了 -> ゴースト化")
                    agent['ghosted'] = True
                    manager.states[idx] = GoalManager.GHOSTED
                    manager.ghost_times[idx] = current_time  # ゴースト化時刻を記録
                    if manager.goal_locked_by == idx:
                        manager.goal_locked_by = -1

        # [Step B] ロック取得判定
        if manager.goal_locked_by == -1:
            active_distances = []
            for idx, agent in enumerate(agents_data):
                if (not agent.get('collision_occurred', False) and 
                    not agent.get('ghosted', False) and
                    manager.states[idx] == GoalManager.ACTIVE):
                    
                    # 【3D】0:3
                    curr_pos = agent['trial_state'][0:3, step]
                    dist = np.linalg.norm(curr_pos - agent['original_goal'])
                    active_distances.append((idx, dist, agent['id']))
            
            if active_distances:
                active_distances.sort(key=lambda x: x[1])
                nearest_idx, nearest_dist, nearest_id = active_distances[0]
                
                if nearest_dist < lock_distance:
                    manager.goal_locked_by = nearest_idx
                    manager.lock_acquired_times[nearest_idx] = current_time  # ロック取得時刻を記録
                    
                    # --- ★【修正】制御リセット (型・形状エラー対策) ---
                    init_u = agents_data[nearest_idx]['P']['initial_controll']
                    init_u = init_u.reshape(ctrl_dim, 1)
                    
                    horizon = agents_data[nearest_idx]['P']['Trial_size'] + agents_data[nearest_idx]['P']['Horizon_size']
                    reset_seq = np.tile(init_u, (1, 1, horizon))
                    
                    # ★重要: float64 にキャスト
                    agents_data[nearest_idx]['seq_ctrl'] = reset_seq.astype(np.float64)
                    
                    print(f"[Time {current_time:.1f}s] Agent {nearest_id}: ロック取得！制御リセット (距離 {nearest_dist:.2f}m)")

        # [Step C] 目標設定
        for idx, agent in enumerate(agents_data):
            if agent.get('collision_occurred', False) or agent.get('ghosted', False):
                continue
            
            # 【3D】0:3
            curr_pos = agent['trial_state'][0:3, step]
            orig_goal = agent['original_goal']
            dist_to_orig = np.linalg.norm(curr_pos - orig_goal)
            
            if manager.states[idx] == GoalManager.ACTIVE:
                if manager.goal_locked_by == idx and dist_to_orig < manager.goal_threshold:
                    manager.states[idx] = GoalManager.AT_GOAL_WORKING
                    manager.arrival_times[idx] = current_time
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: ゴール到達・作業開始")
                    
                    # ホバリング推力を計算（水平姿勢を仮定）
                    m = agent['P']['m']
                    g = agent['P']['g']
                    F_hover = m * g
                    
                    # 即座に位置・速度・制御入力を固定
                    agent['trial_state'][0:3, step] = manager.goal_pos
                    agent['trial_state'][3:6, step] = 0.0
                    agent['seq_ctrl'][:, :, step:] = F_hover
                    
                
                agent['P']['Goal_state'][0:3, 0] = orig_goal
            
            elif manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                agent['P']['Goal_state'][0:3, 0] = orig_goal
                
                # ホバリング推力を計算
                m = agent['P']['m']
                g = agent['P']['g']
                F_hover = m * g
                
                # 位置・速度を固定し、ホバリング推力を維持
                agent['trial_state'][0:3, step] = manager.goal_pos
                agent['trial_state'][3:6, step] = 0.0
                agent['seq_ctrl'][:, :, step:] = F_hover

        # [Step D] weight のブースト
        boost_factor = 10.0 
        for idx, agent in enumerate(agents_data):
            if manager.goal_locked_by == idx:
                agent['P']['weight'] = agent['original_weight'] * boost_factor
            else:
                agent['P']['weight'] = agent['original_weight']

        # =================================================================
        # 2. MPPI 計算
        # =================================================================
        
        combined_state = np.vstack([agent['trial_state'][:, step].reshape(-1, 1) for agent in agents_data])
        
        for idx, agent in enumerate(agents_data):
            agent['_temp_finished_flag'] = agent.get('ghosted', False) or agent.get('collision_occurred', False)
        
        cost_init = Cost_Fcn_Centralized(combined_state, combined_state, agents_data, banned_point, current_time)
        combined_init = np.vstack([agent['P']['Init_State'] for agent in agents_data])
        cost_init_baseline = Cost_Fcn_Centralized(combined_init, combined_init, agents_data, banned_point, 0)
        
        cost_diff = np.clip(cost_init / (cost_init_baseline + 1e-6), P['vll'], P['vlu'])
        noise_var = (P['var'] * cost_diff).reshape(-1, 1) if (P['var'] * cost_diff).ndim == 1 else P['var'] * cost_diff
        
        combined_ctrl_dim = ctrl_dim * n_agents
        noise_seq = np.random.randn(combined_ctrl_dim, P['K'], P['Horizon_size'])
        for i_noise in range(P['Horizon_size']):
            for idx in range(n_agents):
                ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
                noise_seq[ctrl_slice, :, i_noise] *= noise_var
        
        horizon_input = np.zeros((combined_ctrl_dim, P['K'], P['Horizon_size']))
        for idx, agent in enumerate(agents_data):
            for h in range(P['Horizon_size']):
                # 形状エラー対策　強制的に (ctrl_dim, 1)
                current_u = agent['seq_ctrl'][:, :, step + h].reshape(ctrl_dim, 1)
                horizon_input[idx*ctrl_dim:(idx+1)*ctrl_dim, :, h] = \
                    np.tile(current_u, (1, P['K']))
        
        horizon_input[:, :int(P['random_sample_rate']*P['K']), :] = 0
        
        trj_cost = np.zeros(P['K'])
        sim_state = np.tile(combined_state, (1, P['K']))
        
        for i_sim in range(P['Horizon_size']):
            next_sim_state_list = []
            
            for idx, agent in enumerate(agents_data):
                agent_state = sim_state[idx*state_dim:(idx+1)*state_dim, :]
                
                #  ゴースト機体は予測内でも完全凍結
                if agent.get('ghosted', False):
                    next_agent_state = agent_state.copy()
                    next_agent_state[0:3, :] = -9999.0  # 位置を遥か彼方へ (3D)
                    next_agent_state[3:6, :] = 0.0      # 速度ゼロ (vx,vy,vz)
                    next_sim_state_list.append(next_agent_state)
                
                # 作業中機体はその場(ゴール)に固定
                elif manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                    next_agent_state = agent_state.copy()
                    next_agent_state[0:3, :] = manager.goal_pos.reshape(3, 1)  # 3D
                    next_agent_state[3:6, :] = 0.0  # vx,vy,vz
                    next_sim_state_list.append(next_agent_state)
                
                # 通常機体: 普通にシミュレーション
                else:
                    agent_ctrl = horizon_input[idx*ctrl_dim:(idx+1)*ctrl_dim, :, i_sim] + \
                                noise_seq[idx*ctrl_dim:(idx+1)*ctrl_dim, :, i_sim]
                    next_agent_state = Sim_Model(agent_state, agent_ctrl, agent['P'])
                    next_sim_state_list.append(next_agent_state)
            
            # リストを結合して次の状態とする
            next_sim_state = np.vstack(next_sim_state_list)
            
            # コスト計算
            sim_time = current_time + i_sim * P['dt']
            trj_cost += Cost_Fcn_Centralized(next_sim_state, sim_state, agents_data, banned_point, sim_time)
            
            # 状態更新
            sim_state = next_sim_state
        
        for idx, agent in enumerate(agents_data):
            agent_sim_state = sim_state[idx*state_dim:(idx+1)*state_dim, :]
            trj_cost += Term_Cost(agent_sim_state)
        
        min_cost = np.min(trj_cost)
        norm_cost = np.sum(np.exp(-1 / P['Temp'] * (trj_cost - min_cost)))
        weight = np.exp(-1 / P['Temp'] * (trj_cost - min_cost)) / norm_cost
        
        for idx, agent in enumerate(agents_data):
            if agent.get('collision_occurred', False) or agent.get('ghosted', False):
                continue
            # AT_GOAL_WORKING状態ではホバリング推力を維持するため更新しない
            if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                continue
            
            ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
            for i_update in range(P['Horizon_size']):
                update = np.dot(noise_seq[ctrl_slice, :, i_update], weight)
                
                # ★【形状エラー対策】受け皿と同じ形に変形してから足す
                target_shape = agent['seq_ctrl'][:, :, step + i_update].shape
                agent['seq_ctrl'][:, :, step + i_update] += update.reshape(target_shape)
        
        # =================================================================
        # 3. 状態更新 & 判定
        # =================================================================
        
        all_finished = True
        for idx, agent in enumerate(agents_data):
            if agent.get('ghosted', False):
                # 【3D】Ghost退避
                if agent['trial_state'][0, step] > -9000:
                    agent['trial_state'][:, step + 1] = agent['trial_state'][:, step]
                    agent['trial_state'][0:3, step + 1] = np.array([-9999.0, -9999.0, -9999.0])
                else:
                    agent['trial_state'][:, step + 1] = agent['trial_state'][:, step]
                continue
            
            if agent.get('collision_occurred', False): continue
            all_finished = False
            
            if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                agent['trial_state'][:, step + 1] = agent['trial_state'][:, step].copy()
                # 【3D】位置・速度固定
                agent['trial_state'][0:3, step + 1] = manager.goal_pos
                agent['trial_state'][3:6, step + 1] = 0.0
            else:
                # ★Stateを (12, 1) に、Ctrlを (4, 1) に強制整形して渡す
                # これで (1, 4) が混ざって計算がおかしくなるのを防ぎます
                agent['trial_state'][:, step + 1] = Sim_Model(
                    agent['trial_state'][:, step].reshape(-1, 1),      # State: (12, 1)
                    agent['seq_ctrl'][:, :, step].reshape(ctrl_dim, 1), # Ctrl:  (4, 1) ★ここが重要
                    agent['P']
                ).flatten()
        
        for idx, agent in enumerate(agents_data):
            check_agent_status(agent, step, agents_data, manager, idx)
        
        for agent in agents_data:
            if '_temp_finished_flag' in agent: del agent['_temp_finished_flag']
        
        print(f"Trial loop {step+1}/{P['Trial_size']-1} 経過時間: {time.time() - loop_start:.3f} 秒")

        if all_finished:
            print(f"全機体が終了(Step {step+1})")
            break
    
    # 各エージェントにロック取得時刻とゴースト化時刻を記録
    for idx, agent in enumerate(agents_data):
        agent['lock_acquired_time'] = manager.lock_acquired_times[idx]
        agent['ghost_time'] = manager.ghost_times[idx]
    
    return agents_data

def check_agent_status(agent, step, agents_data, manager, agent_idx):
    """衝突判定 (3D)"""
    P = agent['P']
    trial_state = agent['trial_state']
    
    if agent.get('collision_occurred', False) or agent.get('ghosted', False):
        return True
    
    # ロック取得中かどうかのフラグ
    is_locked = (manager.goal_locked_by == agent_idx or 
                 manager.states[agent_idx] == GoalManager.AT_GOAL_WORKING)
    
    # 障害物衝突（壁との衝突はロック取得中でも常にチェック）
    current_time = (step + 1) * P['dt']
    fale = check(trial_state[:, step + 1], trial_state[:, step], P, current_time)
    
    if np.any(fale >= 1):
        if is_locked:
            # ロック取得中：壁に衝突した場合は壁手前に押し戻す（衝突扱いにしない）
            print(f"Agent {agent['id']}: ロック取得中に壁衝突検出 → 前ステップに押し戻し")
            trial_state[:, step + 1] = trial_state[:, step].copy()
            return False
        # 衝突種別を判定：障害物 vs 地面/天井
        pos_next = trial_state[0:3, step+1]
        min_height = P.get('min_height', 0.0)
        max_height = P.get('max_height', 5.0)
        agent_radius = P.get('agent_radius', 0.3)
        is_ground_ceiling = (pos_next[2] <= min_height + agent_radius) or \
                            (pos_next[2] >= max_height - agent_radius)
        
        if is_ground_ceiling:
            # 地面/天井衝突：直前の経路を遡って障害物衝突を探す
            obstacle_collision_found = False
            if 'object' in P and P['object'].size > 0:
                from check import check_wall_collision_batch
                wall_height = P.get('object_height', P.get('wall_height', 5.0))
                lookback = min(step, 20)
                for back_step in range(step, max(step - lookback, 0), -1):
                    pos_check = trial_state[0:3, back_step]
                    if 0 <= pos_check[2] <= wall_height:
                        for obj_idx in range(P['object'].shape[2]):
                            xv = P['object'][0, :, obj_idx]
                            yv = P['object'][1, :, obj_idx]
                            result = check_wall_collision_batch(
                                np.array([pos_check[0]]), np.array([pos_check[1]]),
                                xv, yv, agent_radius)
                            if np.any(result):
                                agent['collision_pos'] = pos_check.copy()
                                obstacle_collision_found = True
                                print(f"Agent {agent['id']}: 障害物衝突検出（step {back_step}）")
                                break
                    if obstacle_collision_found:
                        break
            
            if not obstacle_collision_found:
                agent['collision_pos'] = trial_state[0:3, step+1].copy()
                print(f"Agent {agent['id']}: 地面/天井と衝突")
        else:
            agent['collision_pos'] = trial_state[0:3, step].copy()
            print(f"Agent {agent['id']}: 障害物と衝突")
        
        agent['collision_occurred'] = True
        trial_state[:, step+2:] = 0
        if manager.goal_locked_by == agent_idx: manager.goal_locked_by = -1
        manager.states[agent_idx] = GoalManager.GHOSTED
        # リセット
        agent['P']['weight'] = agent.get('original_weight', agent['P']['weight'])
        return True
    
    # 機体間衝突（ロック取得中はスキップ）
    if is_locked:
        return False
    
    safety_distance = P.get('safety_distance', 0.7)
    my_pos = trial_state[0:3, step+1] # 3D
    
    for other_agent in agents_data:
        if other_agent['id'] == agent['id']: continue
        if other_agent.get('collision_occurred'): continue
        if other_agent.get('ghosted'): continue
        
        other_pos = other_agent['trial_state'][0:3, step+1] # 3D
        dist = np.linalg.norm(my_pos - other_pos)
        
        if dist < safety_distance:
            print(f"Agent {agent['id']}: Agent {other_agent['id']}と衝突 ({dist:.2f}m)")
            agent['collision_occurred'] = True
            agent['collision_pos'] = trial_state[0:3, step].copy()
            trial_state[:, step+2:] = 0
            if manager.goal_locked_by == agent_idx: manager.goal_locked_by = -1
            manager.states[agent_idx] = GoalManager.GHOSTED
            # リセット
            agent['P']['weight'] = agent.get('original_weight', agent['P']['weight'])
            return True
            
    return False