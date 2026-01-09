"""
集中型マルチエージェントMPPI（MPPI_GTをベース）
状態と制御入力を連結して最適化
【改善版 v3】待機列の優先順位管理
1. 待機中の機体は互いのコスト計算から除外（finished_flags扱い）
2. ゴールが空いたときは「最初に待機列に入った機体」が優先
3. 円周配置は維持
"""
import numpy as np
import time
from Cost_Fcn_MultiAgent import Cost_Fcn_Centralized
from Sim_Model import Sim_Model
from Term_Cost import Term_Cost
from check import check

# --- ゴール管理クラス ---
class GoalManager:
    """エージェントのゴール到着・待機・作業・ゴースト化を管理"""
    
    # 状態定義
    ACTIVE = 0           # 通常移動中
    WAITING_IN_QUEUE = 1 # 待機列で待機中（ゴール占有中のため）
    AT_GOAL_WORKING = 2  # ゴール到着・作業中
    GHOSTED = 3          # ゴースト化済み（終了）
    
    def __init__(self, n_agents, goal_pos, wait_time=5.0, goal_threshold=0.2):
        self.n_agents = n_agents
        self.goal_pos = goal_pos  # 共通ゴール座標 (x, y) ※これは不変
        self.wait_time = wait_time
        self.goal_threshold = goal_threshold
        
        self.arrival_times = np.zeros(n_agents)    # 到着時刻（作業開始時刻）
        self.states = np.zeros(n_agents, dtype=int)  # 各エージェントの状態
        self.wait_positions = np.zeros((n_agents, 2))  # 各エージェントの待機位置（固定）
        self.queue_entry_times = np.full(n_agents, np.inf)  # 待機列入り時刻（優先順位用）
        self.goal_locked_by = -1  # 現在ゴールを占有しているエージェントID (-1: 空き)
        
    def get_next_in_queue(self):
        """待機列の中で最も早く入った機体のインデックスを返す"""
        waiting_agents = np.where(self.states == self.WAITING_IN_QUEUE)[0]
        if len(waiting_agents) == 0:
            return -1
        # 最も早く待機列に入った機体
        earliest_idx = waiting_agents[np.argmin(self.queue_entry_times[waiting_agents])]
        return earliest_idx

# --------------------------------

def MPPI_MultiAgent(agents_data, banned_point):
    """
    集中型マルチエージェントMPPI（改善版 v3）
    """
    n_agents = len(agents_data)
    P = agents_data[0]['P']
    state_dim = P['State_dim']
    ctrl_dim = P['Ctrl_dim']
    
    # --- GoalManagerの初期化 ---
    common_goal_pos = agents_data[0]['P']['Goal_state'][0:2, 0]
    wait_time = P.get('goal_wait_time', 5.0)
    goal_threshold = P.get('goal_threshold', 0.2)
    # 待機距離を広めに確保（仮ロック距離より外側）
    wait_distance = P.get('goal_wait_distance', 3.5)
    
    manager = GoalManager(n_agents, common_goal_pos, wait_time, goal_threshold)

    # 各機体の初期化
    for agent in agents_data:
        P_agent = agent['P']
        agent['trial_state'] = np.zeros((state_dim, P_agent['Trial_size']))
        agent['seq_ctrl'] = np.zeros((ctrl_dim, 1, P_agent['Trial_size'] + P_agent['Horizon_size']))
        # 初期入力
        for i in range(4):
            agent['seq_ctrl'][i, 0, :] = P_agent['initial_controll'][i, 0]
        
        agent['trial_state'][:, 0] = P_agent['Init_State'].flatten()
        agent['ghosted'] = False  # 完全透明化フラグ
        
        # 元のゴール座標をバックアップ (x, y)
        if 'original_goal' not in agent:
            agent['original_goal'] = P_agent['Goal_state'][0:2, 0].copy()
        
        # 元のmax_approach_speedをバックアップ
        if 'original_max_approach_speed' not in agent:
            agent['original_max_approach_speed'] = P_agent.get('max_approach_speed', 2.0)
    
    # MPPIループ
    for step in range(P['Trial_size'] - 1):
        loop_start = time.time()
        current_time = step * P['dt']
        
        # --- GoalManager & ゴール座標の動的更新 ---
        for idx, agent in enumerate(agents_data):
            # 終了済みの機体はスキップ
            if agent.get('collision_occurred', False) or agent.get('ghosted', False):
                continue
            
            # 現在位置
            curr_pos = agent['trial_state'][0:2, step]
            
            # 本来のゴール座標（バックアップから参照）
            orig_goal = agent['original_goal']
            
            # 本来のゴールまでの距離（判定用）
            dist_to_orig = np.linalg.norm(curr_pos - orig_goal)
            
            # -------------------------------------------------
            # 状態遷移ロジック
            # -------------------------------------------------
            current_state = manager.states[idx]
            
            # ACTIVE状態: 通常移動中
            if current_state == GoalManager.ACTIVE:
                # --- 仮ロック取得 ---
                if manager.goal_locked_by == -1 and dist_to_orig < manager.goal_threshold * 3:
                    manager.goal_locked_by = idx
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: ロック取得（距離 {dist_to_orig:.2f}m）")
                
                # --- ゴール到着判定 ---
                if manager.goal_locked_by == idx and dist_to_orig < manager.goal_threshold:
                    manager.states[idx] = GoalManager.AT_GOAL_WORKING
                    manager.arrival_times[idx] = current_time
                    agent['P']['Goal_state'][0:2, 0] = orig_goal
                    agent['P']['max_approach_speed'] = agent['original_max_approach_speed']
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: ゴール到着・作業開始")
                
                # --- ロックを持っている場合は必ずゴールへ向かう ---
                elif manager.goal_locked_by == idx:
                    agent['P']['Goal_state'][0:2, 0] = orig_goal
                    agent['P']['max_approach_speed'] = agent['original_max_approach_speed']
                
                # --- 待機判定：ロックが他人によってされている場合のみ待機 ---
                elif manager.goal_locked_by != -1 and manager.goal_locked_by != idx and dist_to_orig < wait_distance + 0.5:
                    
                    # ゴールから自分へのベクトルを計算
                    diff = curr_pos - orig_goal
                    dist_vec = np.linalg.norm(diff)
                    
                    # ベクトルがゼロでなければ正規化（方向を計算）
                    if dist_vec > 1e-6:
                        direction = diff / dist_vec
                    else:
                        direction = np.array([1.0, 0.0])  # 重なっている場合の緊急回避方向
                    
                    # ゴールから wait_distance だけ離れた「手前」の位置を計算
                    # ※少しだけIDで角度をずらすと、完全に重なるのを防げます（任意）
                    angle_offset = (agent['id'] % 2 - 0.5) * 0.5 # -0.25rad か +0.25rad ずらす
                    
                    # 回転行列で少しずらす（重なり防止）
                    c, s = np.cos(angle_offset), np.sin(angle_offset)
                    rot_mat = np.array([[c, -s], [s, c]])
                    direction = np.dot(rot_mat, direction)
                    wait_pos = orig_goal + direction * wait_distance
                    
                    manager.wait_positions[idx] = wait_pos
                    manager.states[idx] = GoalManager.WAITING_IN_QUEUE
                    manager.queue_entry_times[idx] = current_time  # 待機列入り時刻を記録
                    
                    # 目標を待機位置へ
                    agent['P']['Goal_state'][0:2, 0] = wait_pos
                    
                    # 待機位置へ向かうときは速度を落とす（オーバーシュート防止）
                    agent['P']['max_approach_speed'] = agent['original_max_approach_speed']# *0.5
                    
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: 待機列に入る")
                
                else:
                    # 通常移動: 本来のゴールを目指す
                    agent['P']['Goal_state'][0:2, 0] = orig_goal
                    # 通常速度を維持
                    agent['P']['max_approach_speed'] = agent['original_max_approach_speed']
            
            # WAITING_IN_QUEUE状態: 待機列で待機中
            elif current_state == GoalManager.WAITING_IN_QUEUE:
                # ゴールが空いた場合
                if manager.goal_locked_by == -1:
                    # 自分が先頭（最優先）かチェック
                    next_idx = manager.get_next_in_queue()
                    
                    if next_idx == idx:
                        # --- 自分の番が来たら、まず「移動モード」に戻す ---
                        print(f"[Time {current_time:.1f}s] Agent {agent['id']}: 待機終了・ゴールへ移動開始")
                        
                        # ロックを仮取得しておく（他の機体が横取りしないように）
                        manager.goal_locked_by = idx
                        
                        # いきなり AT_GOAL_WORKING にせず、ACTIVE に戻す
                        manager.states[idx] = GoalManager.ACTIVE
                        
                        # 待機列入りの記録をリセット
                        manager.queue_entry_times[idx] = np.inf
                        
                        # 目標を本来のゴールへ
                        agent['P']['Goal_state'][0:2, 0] = orig_goal
                        # 速度制限解除
                        agent['P']['max_approach_speed'] = agent['original_max_approach_speed']
                        # 初期化
                        agent['seq_ctrl'][:, :, :] = 0.0
                    else:
                        # まだ自分の番ではない → 待機位置で待つ
                        agent['P']['Goal_state'][0:2, 0] = manager.wait_positions[idx]
                        #agent['P']['max_approach_speed'] = agent['original_max_approach_speed'] * 0.5
                        agent['P']['max_approach_speed'] = agent['original_max_approach_speed'] # 通常速度
                else:
                    # ゴール占有中 → 待機継続
                    agent['P']['Goal_state'][0:2, 0] = manager.wait_positions[idx]
                    #agent['P']['max_approach_speed'] = agent['original_max_approach_speed'] * 0.5
                    agent['P']['max_approach_speed'] = agent['original_max_approach_speed'] # 通常速度
            
            # AT_GOAL_WORKING状態: ゴールで作業中
            elif current_state == GoalManager.AT_GOAL_WORKING:
                elapsed = current_time - manager.arrival_times[idx]
                
                if elapsed >= manager.wait_time:
                    print(f"[Time {current_time:.1f}s] Agent {agent['id']}: 作業完了")
                    agent['ghosted'] = True
                    manager.states[idx] = GoalManager.GHOSTED
                    manager.goal_locked_by = -1  # ロック解放
                else:
                    # 作業継続: ゴール位置を維持
                    agent['P']['Goal_state'][0:2, 0] = orig_goal

        # --- 以下、通常のMPPI処理 ---

        # 連結状態を作成
        combined_state = np.vstack([agent['trial_state'][:, step].reshape(-1, 1) for agent in agents_data])
        
        # 【重要】finished_flagsの処理
        # - 待機中(WAITING)の機体同士は反発しない（デッドロック防止）
        # - ACTIVE機体は待機中の機体を避ける必要がある
        # → 解決策: 待機中の機体もfinished扱いにする（互いに反発しない）
        #   ただし、ACTIVE機体には別途待機位置を避けるコストが必要
        finished_flags_for_cost = np.array([
            agent.get('ghosted', False) or 
            agent.get('collision_occurred', False) or
            manager.states[idx] == GoalManager.WAITING_IN_QUEUE  # 待機中も除外（デッドロック防止）
            for idx, agent in enumerate(agents_data)
        ])
        
        # agents_dataに一時的にフラグを追加（Cost_Fcn_Centralized用）
        for idx, agent in enumerate(agents_data):
            agent['_temp_finished_flag'] = finished_flags_for_cost[idx]
        
        # コスト計算
        cost_init = Cost_Fcn_Centralized(combined_state, combined_state, agents_data, banned_point, current_time)
        combined_init = np.vstack([agent['P']['Init_State'] for agent in agents_data])
        cost_init_baseline = Cost_Fcn_Centralized(combined_init, combined_init, agents_data, banned_point, 0)
        
        # コスト差分と正規化
        cost_diff = cost_init / (cost_init_baseline + 1e-6)
        cost_diff = np.clip(cost_diff, P['vll'], P['vlu'])
        
        # ノイズ分散
        noise_var = P['var'] * cost_diff
        if noise_var.ndim == 1:
            noise_var = noise_var.reshape(-1, 1)
        
        # ノイズ生成と入力シーケンス作成
        combined_ctrl_dim = ctrl_dim * n_agents
        noise_seq = np.random.randn(combined_ctrl_dim, P['K'], P['Horizon_size'])
        horizon_input = np.zeros((combined_ctrl_dim, P['K'], P['Horizon_size']))
        
        for i_noise in range(P['Horizon_size']):
            for idx in range(n_agents):
                ctrl_slice = slice(idx*ctrl_dim, (idx+1)*ctrl_dim)
                noise_seq[ctrl_slice, :, i_noise] *= noise_var
        
        for idx, agent in enumerate(agents_data):
            for h in range(P['Horizon_size']):
                horizon_input[idx*ctrl_dim:(idx+1)*ctrl_dim, :, h] = \
                    np.tile(agent['seq_ctrl'][:, :, step + h], (1, P['K']))
        
        # ランダムサンプリング
        horizon_input[:, :int(P['random_sample_rate']*P['K']), :] = 0
        
        # 予測シミュレーション
        trj_cost = np.zeros(P['K'])
        sim_state = np.tile(combined_state, (1, P['K']))
        
        for i_sim in range(P['Horizon_size']):
            next_sim_state_list = []
            for idx, agent in enumerate(agents_data):
                agent_state = sim_state[idx*state_dim:(idx+1)*state_dim, :]
                
                # 　AT_GOAL_WORKING機体は予測でも位置固定
                if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                    # 現在の状態をコピー（位置固定）
                    next_agent_state = agent_state.copy()
                    # 位置をゴールに固定（全サンプル）
                    next_agent_state[0, :] = manager.goal_pos[0]
                    next_agent_state[1, :] = manager.goal_pos[1]
                    # 速度を0に（全サンプル）
                    next_agent_state[3, :] = 0.0
                    next_agent_state[4, :] = 0.0
                else:
                    # 通常の予測シミュレーション
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
        
        # 重み計算と制御更新
        min_cost = np.min(trj_cost)
        norm_cost = np.sum(np.exp(-1 / P['Temp'] * (trj_cost - min_cost)))
        weight = np.exp(-1 / P['Temp'] * (trj_cost - min_cost)) / norm_cost
        
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
        
        # --- ループ1: まず全機体の「次状態」を確定させる ---
        all_finished = True
        
        for idx, agent in enumerate(agents_data):
            # ゴースト化済みの機体処理（座標退避）
            if agent.get('ghosted', False):
                if agent['trial_state'][0, step] > -9000:
                    agent['trial_state'][:, step + 1] = agent['trial_state'][:, step]
                    agent['trial_state'][0:2, step + 1] = np.array([-9999.0, -9999.0])
                else:
                    agent['trial_state'][:, step + 1] = agent['trial_state'][:, step]
                continue
            
            if agent.get('collision_occurred', False):
                continue

            all_finished = False
            
            # 作業中は位置固定
            if manager.states[idx] == GoalManager.AT_GOAL_WORKING:
                agent['trial_state'][:, step + 1] = agent['trial_state'][:, step].copy()
                agent['trial_state'][0:2, step + 1] = manager.goal_pos
                agent['trial_state'][3:5, step + 1] = 0.0
            else:
                agent['trial_state'][:, step + 1] = Sim_Model(
                    agent['trial_state'][:, step], 
                    agent['seq_ctrl'][:, :, step], 
                    agent['P']
                ).flatten()

        # --- ループ2: 全員の座標が確定した後に、衝突判定を行う ---
        for idx, agent in enumerate(agents_data):
            # 衝突チェック（ここで初めて全員の step+1 が揃っているため正しく判定できる）
            check_agent_status(agent, step, agents_data, manager, idx)
        
        # 一時フラグを削除
        for agent in agents_data:
            if '_temp_finished_flag' in agent:
                del agent['_temp_finished_flag']
        
        print(f"Trial loop {step+1}/{P['Trial_size']-1} 経過時間: {time.time() - loop_start:.3f} 秒")
        
        if all_finished:
            print(f"全機体が終了（Step {step+1}）")
            break
    
    return agents_data

def check_agent_status(agent, step, agents_data, manager, agent_idx):
    """
    機体の状態チェック（衝突判定）
    戻り値: True なら終了済み
    """
    P = agent['P']
    trial_state = agent['trial_state']
    
    # 終了済み
    if agent.get('collision_occurred', False) or agent.get('ghosted', False):
        return True
    
    # 障害物衝突
    current_time = (step + 1) * P['dt']
    fale = check(trial_state[:, step + 1], trial_state[:, step], P, current_time)
    
    if np.any(fale >= 1):
        print(f"Agent {agent['id']}: 障害物と衝突")
        agent['collision_occurred'] = True
        agent['collision_pos'] = trial_state[0:3, step].copy()
        trial_state[:, step+2:] = 0
        # ロック解放と状態リセット
        if manager.goal_locked_by == agent_idx:
            manager.goal_locked_by = -1
        manager.states[agent_idx] = GoalManager.GHOSTED
        manager.queue_entry_times[agent_idx] = np.inf
        # 速度制限を元に戻す
        agent['P']['max_approach_speed'] = agent.get('original_max_approach_speed', 2.0)
        return True
    
    # エージェント間衝突
    safety_distance = P.get('safety_distance', 0.7)
    my_pos = trial_state[0:2, step+1]
    my_state = manager.states[agent_idx]
    
    for other_idx, other_agent in enumerate(agents_data):
        if other_agent['id'] == agent['id']:
            continue
        if other_agent.get('collision_occurred', False):
            continue
        # ゴーストは無視
        if other_agent.get('ghosted', False):
            continue
        # 作業中の機体も衝突判定から除外（ゴールに固定されているため）
        #if manager.states[other_idx] == GoalManager.AT_GOAL_WORKING:
        #    continue
            
        other_pos = other_agent['trial_state'][0:2, step+1]
        dist = np.linalg.norm(my_pos - other_pos)
        
        if dist < safety_distance:
            print(f"Agent {agent['id']}: Agent {other_agent['id']}と衝突 ({dist:.2f}m)")
            agent['collision_occurred'] = True
            agent['collision_pos'] = trial_state[0:3, step].copy()
            trial_state[:, step+2:] = 0
            # ロック解放と状態リセット
            if manager.goal_locked_by == agent_idx:
                manager.goal_locked_by = -1
            manager.states[agent_idx] = GoalManager.GHOSTED
            manager.queue_entry_times[agent_idx] = np.inf
            # 速度制限を元に戻す
            agent['P']['max_approach_speed'] = agent.get('original_max_approach_speed', 2.0)
            return True
            
    return False
