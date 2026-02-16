import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False


def plot_inter_agent_distance(agents_history, P, save_path=None):
    """
    マルチエージェント間の距離時系列グラフを作成
    目標到達後は距離を0に設定
    """
    n_agents = len(agents_history)
    
    if n_agents < 2:
        print("エージェントが1機のみのため、機体間距離グラフはスキップします。")
        return
    
    # 最小ステップ数を取得（各エージェントで異なる可能性）
    num_steps = min(agent['trial_state'].shape[1] for agent in agents_history)
    dt = P['dt']
    time_array = np.arange(num_steps) * dt
    
    # 各エージェントの目標到達時刻を検出
    goal_pos = P.get('shared_goal_pos', P['Goal_state'][0:3, 0])
    goal_threshold = P.get('goal_threshold', 0.4)
    
    goal_reached_steps = []  # 各エージェントの目標到達ステップ
    for idx, agent in enumerate(agents_history):
        reached_step = None
        for step in range(num_steps):
            pos = agent['trial_state'][0:3, step]
            dist_to_goal = np.linalg.norm(pos - goal_pos)
            if dist_to_goal <= goal_threshold:
                reached_step = step
                print(f"Quadrotor {idx+1}: 目標到達 {step * dt:.2f}秒後（ステップ {step})")
                break
        
        if reached_step is None:
            reached_step = num_steps - 1  # 到達しなかった場合は最後
        goal_reached_steps.append(reached_step)
    
    # 全エージェントが到達するまで
    analysis_end_step = max(goal_reached_steps) + 1
    time_array = time_array[:analysis_end_step]
    
    # エージェント間距離を計算
    agent_pairs = []
    distance_data = {}
    
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            pair_name = f"Quadrotor {i+1}-{j+1}"
            agent_pairs.append((i, j, pair_name))
            distance_data[pair_name] = np.zeros(analysis_end_step)
    
    print("\n機体間距離を計算中...")
    for step in range(analysis_end_step):
        for i, j, pair_name in agent_pairs:
            # どちらかのエージェントが目標に到達していたら距離を0にする
            if step >= goal_reached_steps[i] or step >= goal_reached_steps[j]:
                distance_data[pair_name][step] = 0.0
            else:
                pos_i = agents_history[i]['trial_state'][0:3, step]
                pos_j = agents_history[j]['trial_state'][0:3, step]
                
                distance = np.linalg.norm(pos_i - pos_j)
                distance_data[pair_name][step] = distance
    
    # 安全距離
    safety_distance = P.get('safety_distance', 0.65)
    agent_radius = P.get('agent_radius', 0.3)
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 各ペアの距離をプロット（機体1=青、機体2=赤、機体3=緑）
    pair_colors = {
        'Quadrotor 1-2': 'purple',  # 青と赤の中間
        'Quadrotor 1-3': 'cyan',    # 青と緑の中間
        'Quadrotor 2-3': 'orange'   # 赤と緑の中間
    }
    for idx, (i, j, pair_name) in enumerate(agent_pairs):
        color = pair_colors.get(pair_name, 'gray')
        ax.plot(time_array, distance_data[pair_name], color=color, 
                linewidth=2, label=pair_name, alpha=0.8)
    
    # 衝突距離線（機体半径の2倍）
    collision_distance = 2 * agent_radius
    ax.axhline(y=collision_distance, color='red', linestyle='--', linewidth=2,
               label=f'Collision Distance (d = {collision_distance:.2f} m)')
    
    # グラフ装飾
    ax.set_xlabel('Time $t$ [s]', fontsize=18)
    ax.set_ylabel('Inter-Agent Distance $d$ [m]', fontsize=18)
    ax.set_title('Inter-Agent Distance Over Time', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, loc='best')
    
    # y軸範囲の調整
    max_distance = max(
        np.max(distances) for distances in distance_data.values()
    )
    ax.set_ylim(0, max_distance * 1.1, fontsize=18)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()


def plot_velocity_profiles(agents_history, P, save_path=None):
    """
    各エージェントの速度の時系列グラフを作成
    """
    n_agents = len(agents_history)
    num_steps = min(agent['trial_state'].shape[1] for agent in agents_history)
    dt = P['dt']
    time_array = np.arange(num_steps) * dt
    
    # 速度ノルムを計算
    velocity_data = {}
    
    print("\n速度を計算中...")
    for idx, agent in enumerate(agents_history):
        agent_name = f"Quadrotor {idx + 1}"
        velocity_norm = np.zeros(num_steps)
        
        for step in range(num_steps):
            vx = agent['trial_state'][3, step]
            vy = agent['trial_state'][4, step]
            vz = agent['trial_state'][5, step]
            velocity_norm[step] = np.sqrt(vx**2 + vy**2 + vz**2)
        
        velocity_data[agent_name] = velocity_norm
        
        # 統計情報
        max_vel = np.max(velocity_norm)
        avg_vel = np.mean(velocity_norm)
        print(f"{agent_name}: 最大速度 {max_vel:.2f} m/s, 平均速度 {avg_vel:.2f} m/s")
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 各エージェントの速度をプロット（機体1=青、機体2=赤、機体3=緑）
    agent_colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
    
    for idx, (agent_name, vel_data) in enumerate(velocity_data.items()):
        color = agent_colors[idx % len(agent_colors)]
        agent = agents_history[idx]
        
        ax.plot(time_array, vel_data, color=color, linewidth=2.5, 
                label=agent_name, alpha=0.8)
        
        # ロック取得時刻に縦線を表示
        lock_time = agent.get('lock_acquired_time', -1)
        if lock_time >= 0:
            ax.axvline(x=lock_time, color=color, 
                      linestyle='--', alpha=0.6, linewidth=2.5,
                      label=f'{agent_name} ロック取得')
        
        # ゴースト化時刻に丸印を表示
        ghost_time = agent.get('ghost_time', -1)
        if ghost_time >= 0:
            # ゴースト化時刻に最も近いステップを見つける
            ghost_step = int(ghost_time / dt)
            if ghost_step < len(vel_data):
                ax.plot(ghost_time, vel_data[ghost_step], 'o', 
                       color=color, markersize=12, 
                       markeredgecolor='black', markeredgewidth=2,
                       label=f'{agent_name} ゴースト化')
    
    # グラフ装飾
    ax.set_xlabel('時間 $t$ [s]', fontsize=18)
    ax.set_ylabel('速度ノルム $||v||$ [m/s]', fontsize=18)
    ax.set_title('各機体の速度推移', 
                 fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='best')
    ax.set_ylim(0, 5)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()


def plot_control_inputs(agents_history, P, save_path=None):
    """
    各エージェントの制御入力時系列グラフを作成
    """
    n_agents = len(agents_history)
    num_steps = min(agent.get('seq_ctrl', agent['trial_state']).shape[-1] 
                   for agent in agents_history)
    dt = P['dt']
    time_array = np.arange(num_steps) * dt
    
    # 制御入力データを取得
    thrust_data = {}
    
    print("\n制御入力を解析中...")
    for idx, agent in enumerate(agents_history):
        agent_name = f"Agent {idx + 1}"
        
        # seq_ctrlから推力（F）を取得
        if 'seq_ctrl' in agent:
            seq_ctrl = agent['seq_ctrl']  # shape: (4, 1, num_steps) or similar
            if seq_ctrl.ndim == 3:
                thrust = seq_ctrl[0, 0, :num_steps]  # 推力F
            else:
                thrust = seq_ctrl[0, :num_steps]
        else:
            # seq_ctrlがない場合はスキップ
            continue
        
        thrust_data[agent_name] = thrust
        
        # 統計情報
        max_thrust = np.max(thrust)
        avg_thrust = np.mean(thrust)
        std_thrust = np.std(thrust)
        print(f"{agent_name}: 最大推力 {max_thrust:.2f} N, 平均 {avg_thrust:.2f} N, 標準偏差 {std_thrust:.2f} N")
    
    if not thrust_data:
        print("制御入力データが見つかりませんでした。")
        return
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # ホバリング推力を計算
    m = P.get('m', 1.3)
    g = P.get('g', 9.8)
    hover_thrust = m * g
    
    # 各エージェントの推力をプロット（機体1=青、機体2=赤、機体3=緑）
    agent_colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
    for idx, (agent_name, thrust) in enumerate(thrust_data.items()):
        color = agent_colors[idx % len(agent_colors)]
        ax.plot(time_array, thrust, color=color, linewidth=2, 
                label=agent_name, alpha=0.7)
    
    # ホバリング推力の参照線
    ax.axhline(y=hover_thrust, color='gray', linestyle='--', linewidth=1.5,
               label=f'ホバリング推力 ({hover_thrust:.2f} N)', alpha=0.6)
    
    # グラフ装飾
    ax.set_xlabel('時間 $t$ [s]', fontsize=18)
    ax.set_ylabel('推力 $F$ [N]', fontsize=18)
    ax.set_title('制御入力（推力）時系列', 
                 fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='best')
    
    # チャタリング分析
    chattering_text = "チャタリング分析:\n"
    for idx, (agent_name, thrust) in enumerate(thrust_data.items()):
        # 推力の微分（変化率）
        thrust_diff = np.diff(thrust)
        avg_change = np.mean(np.abs(thrust_diff))
        chattering_text += f"{agent_name}: 平均変化率 {avg_change:.3f} N/step\n"
    
    ax.text(0.98, 0.02, chattering_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            color='darkblue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()
