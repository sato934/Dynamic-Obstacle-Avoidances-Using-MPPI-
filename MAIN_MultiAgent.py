"""
集中型マルチエージェントMPPI制御システム
MPPI_GTをベースに、状態と制御入力を連結して最適化
"""
import numpy as np
from datetime import datetime
from copy import deepcopy
from Load_Settings import Load_Settings
from MPPI_MultiAgent import MPPI_MultiAgent
from Graph_MultiAgent import Graph_MultiAgent

# --- 初期化 ---
print(datetime.now())

# シナリオとエージェント数の設定
scenario = 7  # シナリオ番号 固定
n_agents = 3

# ★ 機体の優先順位を設定（0が最優先）
# None: 自動割り当て（0, 1, 2, ...）
# リスト: 任意の順序を指定
#   例: [0, 2, 1] → Agent 1が最優先、次にAgent 3、最後にAgent 2
#   例: [2, 0, 1] → Agent 2が最優先、次にAgent 1、最後にAgent 3
custom_priorities = None  # ここで優先順位を指定

# 優先順位の決定
if custom_priorities is None:
    priorities = list(range(n_agents))  # デフォルト: [0, 1, 2]
    print(f"優先順位: {priorities}")
else:
    priorities = custom_priorities
    print(f"優先順位: {priorities}")
    # 優先順位の検証
    if len(priorities) != n_agents:
        raise ValueError(f"優先順位の数（{len(priorities)}）が機体数（{n_agents}）と一致しません")
    if set(priorities) != set(range(n_agents)):
        raise ValueError(f"優先順位は0から{n_agents-1}の数字を重複なく含む必要があります")

# 共通のパラメータ（障害物配置など）を先に生成
P_common = Load_Settings(scenario)

# 各機体のパラメータ設定
agents_data = []
for agent_id in range(n_agents):
    # 共通パラメータをディープコピー（障害物配置を完全に共有）
    P = deepcopy(P_common)
    np.random.seed(P['seed'] + agent_id)
    
    # 初期位置と目標位置を設定
    if n_agents == 3:
        # 横に並べる（x方向に分散）
        init_positions = [np.array([0, 4]), np.array([-2, 4]), np.array([2, 4])]
        # 全機体が同じゴール座標に到達
        common_goal = np.array([0, -4])
        goal_positions = [common_goal, common_goal, common_goal]
    else:
        # 円形配置
        angle_init = 2 * np.pi * agent_id / n_agents
        radius = 2.0
        init_positions = [np.array([radius * np.cos(angle_init), 4 + radius * np.sin(angle_init)])]
        common_goal = np.array([0, -5])
        goal_positions = [common_goal]
    
    init_pos = init_positions[agent_id % len(init_positions)]
    goal_pos = goal_positions[agent_id % len(goal_positions)]
    
    P['Init_State'] = np.array([
        init_pos[0], init_pos[1], 5, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    P['Goal_state'] = np.array([
        goal_pos[0], goal_pos[1], 5, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1)
    
    agents_data.append({
        'id': agent_id + 1,  # 表示用ID（1, 2, 3）
        'P': P,
        'priority': priorities[agent_id],  # カスタム優先順位または自動割り当て
        'trial_state': None,
        'seq_ctrl': None,
        'collision_pos': None,
        'goal_reached': False,
        'collision_occurred': False
    })

# 優先順位の確認表示
print("\n=== 機体設定 ===")
for agent in agents_data:
    print(f"Agent {agent['id']}: 優先度={agent['priority']} ({'最優先' if agent['priority']==0 else f'{agent['priority']+1}番目'})")
print(f"到達順序: Agent {[a['id'] for a in sorted(agents_data, key=lambda x: x['priority'])]}")
print("================\n")

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
