"""
集中型マルチエージェントMPPI制御システム
MPPI_GTをベースに、状態と制御入力を連結して最適化
"""
import numpy as np
from datetime import datetime
from Load_Settings import Load_Settings
from MPPI_MultiAgent import MPPI_MultiAgent
from Graph_MultiAgent import Graph_MultiAgent

# --- 初期化 ---
print(datetime.now())

# シナリオとエージェント数の設定
scenario = 7
n_agents = 3

# 各機体のパラメータ設定
agents_data = []
for agent_id in range(n_agents):
    P = Load_Settings(scenario)
    np.random.seed(P['seed'] + agent_id)
    
    # 初期位置と目標位置を設定（既存のスタート・ゴールをベースに分散）
    if n_agents == 3:
        # 横に並べる（x方向に分散）
        init_positions = [np.array([0, 4]), np.array([-2, 4]), np.array([2, 4])]
        goal_positions = [np.array([0, -4]), np.array([2, -4]), np.array([-2, -4])]
    else:
        # 円形配置
        angle_init = 2 * np.pi * agent_id / n_agents
        angle_goal = angle_init + np.pi
        radius = 2.0
        init_positions = [np.array([radius * np.cos(angle_init), 4 + radius * np.sin(angle_init)])]
        goal_positions = [np.array([radius * np.cos(angle_goal), -5 + radius * np.sin(angle_goal)])]
    
    init_pos = init_positions[agent_id % len(init_positions)]
    goal_pos = goal_positions[agent_id % len(goal_positions)]
    
    P['Init_State'] = np.array([
        init_pos[0], init_pos[1], 5, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    P['Goal_state'] = np.array([
        goal_pos[0], goal_pos[1], 5, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    agents_data.append({
        'id': agent_id,
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

print('\nFinish!!')
print(datetime.now())
