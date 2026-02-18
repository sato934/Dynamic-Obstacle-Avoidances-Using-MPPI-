import numpy as np
from datetime import datetime
from copy import deepcopy
from Load_Settings import Load_Settings
from MPPI_MultiAgent import MPPI_MultiAgent
from Graph_MultiAgent import Graph_MultiAgent
from Graph_MultiAgent_Analysis import (
    plot_inter_agent_distance,
    plot_velocity_profiles,
    plot_control_inputs
)

# --- 初期化 ---
print(datetime.now())

# シナリオとエージェント数の設定
scenario = 8  # シナリオ番号 固定
n_agents = 3

# 共通のパラメータ（障害物配置など）を先に生成
P_common = Load_Settings(scenario)

# 各機体のパラメータ設定
agents_data = []
for agent_id in range(n_agents):
    # 共通パラメータをディープコピー（障害物配置を完全に共有）
    P = deepcopy(P_common)
    np.random.seed(P['seed'] + agent_id)
    
    # 初期位置と目標位置を設定（3D対応）
    if n_agents == 3:
        # 横に並べる（x方向に分散、3D座標）
        init_positions = [
            np.array([0, 4, 0.5]), 
            np.array([-2, 4, 0.5]), 
            np.array([2, 4, 0.5])
        ]
        # 全機体が同じゴール座標に到達（3D）
        common_goal = np.array([0, -4, 2.5])
        goal_positions = [common_goal, common_goal, common_goal]
    else:
        # 円形配置（3D対応）
        angle_init = 2 * np.pi * agent_id / n_agents
        radius = 2.0
        init_positions = [np.array([
            radius * np.cos(angle_init), 
            4 + radius * np.sin(angle_init),
            0.5
        ])]
        common_goal = np.array([0, -5, 2.5])
        goal_positions = [common_goal]
    
    init_pos = init_positions[agent_id % len(init_positions)]
    goal_pos = goal_positions[agent_id % len(goal_positions)]
    
    P['Init_State'] = np.array([
        init_pos[0], init_pos[1], init_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    P['Goal_state'] = np.array([
        goal_pos[0], goal_pos[1], goal_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    agents_data.append({
        'id': agent_id + 1,  # 表示用ID（1, 2, 3）
        'P': P,
        'trial_state': None,
        'seq_ctrl': None,
        'collision_pos': None,
        'goal_reached': False,
        'collision_occurred': False
    })

# 禁止点（共通）
banned_point = np.full((3, 100), np.nan)

# 複数回試行
trial_results = []
for trial_idx in range(1):  # 反復回数
    t0 = datetime.now()
    print(f"Iteration: {trial_idx + 1}")
    
    # MPPI実行
    trial_agents_data = MPPI_MultiAgent(agents_data.copy(), banned_point)
    trial_results.append(trial_agents_data)
    
    t1 = datetime.now()
    print(f"Elapsed: {t1 - t0}")

# グラフ描画
Graph_MultiAgent(trial_results[0])

# マルチエージェント分析グラフの作成
print("\n=== マルチエージェント分析グラフを作成中 ===")

# 共有ゴール位置を設定
common_goal = np.array([0, -4, 2.5])
P_common['shared_goal_pos'] = common_goal

import os
multi_save_dir = 'Result_Multi_Animation'
if not os.path.exists(multi_save_dir):
    os.makedirs(multi_save_dir)

# 1. 機体間距離グラフ
print("\n--- 機体間距離グラフ ---")
plot_inter_agent_distance(
    trial_results[0], P_common,
    save_path=os.path.join(multi_save_dir, 'multi_agent_distance.png')
)

# 2. 速度プロファイルグラフ
print("\n--- 速度プロファイルグラフ ---")
plot_velocity_profiles(
    trial_results[0], P_common,
    save_path=os.path.join(multi_save_dir, 'multi_velocity_profiles.png')
)

# 3. 制御入力グラフ 使わないかも？
print("\n--- 制御入力グラフ ---")
plot_control_inputs(
    trial_results[0], P_common,
    save_path=os.path.join(multi_save_dir, 'multi_control_inputs.png')
)

print('\nFinish!!')
print(datetime.now())
