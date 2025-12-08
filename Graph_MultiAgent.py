"""
マルチエージェント用グラフ描画（Graph_xベース）
機体数が増えるだけで、ロジックは同じ
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

def Graph_MultiAgent(agents_data):
    """
    マルチエージェント軌跡アニメーション
    Graph_xと同じロジック、機体数が増えるだけ
    """
    # 基本パラメータ
    P = agents_data[0]['P']
    n_agents = len(agents_data)
    
    # 障害物描画
    fig, ax = plt.subplots()
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    plt.box(True)
    plt.axis(P['axis'])
    ax.set_aspect('equal')
    
    # 静的障害物描画
    if 'object' in P:
        obj = P['object']
        if obj.ndim == 3:
            n_obs = obj.shape[2]
            for i in range(n_obs):
                xv = obj[0, :, i]
                yv = obj[1, :, i]
                ax.fill(xv, yv, color='blue', alpha=0.5, edgecolor='k')
        else:
            xv = obj[0, :]
            yv = obj[1, :]
            ax.fill(xv, yv, color='blue', alpha=0.5, edgecolor='k')
    
    # 各機体のスタート・ゴール座標
    for idx, agent in enumerate(agents_data):
        P_agent = agent['P']
        ax.plot(P_agent['Init_State'][0, 0], P_agent['Init_State'][1, 0], 
               'o', color=[0.5, 0, 1], markersize=5, markerfacecolor=[0.5, 0, 1], linewidth=2)
        ax.plot(P_agent['Goal_state'][0, 0], P_agent['Goal_state'][1, 0], 
               marker='D', color=[0, 0, 1], markersize=5, markerfacecolor=[0, 0, 1], linewidth=2)
    
    gif_filename = 'multi_agent_animation.gif'
    delay = 0.1
    images = []
    frame_interval = 2
    show_trajectory = True
    
    # 各機体の経路用ラインオブジェクト
    line_objects = []
    if show_trajectory:
        for idx in range(n_agents):
            g = 0 if n_agents == 1 else idx / (n_agents - 1)
            h, = ax.plot([], [], '-', color=[1-g, 0+g, 0], linewidth=1.5)
            line_objects.append(h)
    
    # 最大ステップ数
    max_steps = max(agent['trial_state'].shape[1] for agent in agents_data)
    
    # 動的障害物用のパッチを先に作成
    if 'dynamic' in P and P['dynamic']:
        dyn_obj = P['dynamic_obj']
        
        if dyn_obj.ndim == 3:
            n_obstacles = dyn_obj.shape[2]
        else:
            n_obstacles = 1
            dyn_obj = np.expand_dims(dyn_obj, axis=2)
        
        ax._dynamic_patches = []
        ax._base_circles = []
        ax._waypoints_list = []
        ax._seg_times_list = []
        
        for obs_idx in range(n_obstacles):
            base_circle = dyn_obj[:, :, obs_idx]
            
            if isinstance(P.get('dynamic_waypoints'), list):
                waypoints_for_draw = np.asarray(P['dynamic_waypoints'][obs_idx])
            else:
                waypoints_for_draw = np.asarray(P.get('dynamic_waypoints'))
            
            if isinstance(P.get('dynamic_segment_times'), list):
                seg_times_for_draw = np.asarray(P['dynamic_segment_times'][obs_idx])
            else:
                seg_times_for_draw = np.asarray(P.get('dynamic_segment_times'))
            
            color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
            patch = ax.add_patch(
                Polygon(base_circle.T, facecolor=[color_intensity, 0, 0], alpha=0.5, edgecolor='k')
            )
            
            ax._dynamic_patches.append(patch)
            ax._base_circles.append(base_circle)
            ax._waypoints_list.append(waypoints_for_draw)
            ax._seg_times_list.append(seg_times_for_draw)
    
    # 各機体の状態追跡
    collision_markers_drawn = [False] * n_agents
    goal_reached = [False] * n_agents
    
    # 時刻表示用のテキスト
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # アニメーションループ（Graph_xと同じ）
    for k in range(max_steps):
        # 各機体の経路を更新
        if show_trajectory:
            for idx, agent in enumerate(agents_data):
                # すでに終了している場合はスキップ
                if goal_reached[idx] or collision_markers_drawn[idx]:
                    continue
                
                trial_state = agent['trial_state']
                x = trial_state[0, :]
                y = trial_state[1, :]
                z = trial_state[2, :]
                
                # 範囲外チェック
                if k >= len(x) or z[k] == 0:
                    continue
                
                # 目標到達チェック
                goal_x = agent['P']['Goal_state'][0, 0]
                goal_y = agent['P']['Goal_state'][1, 0]
                goal_threshold = agent['P'].get('goal_threshold', 0.2)
                distance_to_goal = np.sqrt((x[k] - goal_x)**2 + (y[k] - goal_y)**2)
                
                if distance_to_goal <= goal_threshold:
                    goal_reached[idx] = True
                    line_objects[idx].set_data(x[:k+1], y[:k+1])
                    continue
                
                # 衝突チェック
                is_collision = False
                if k + 1 >= len(x):
                    is_collision = True
                elif z[k+1] == 0:
                    is_collision = True
                
                if is_collision:
                    collision_pos = agent.get('collision_pos')
                    if collision_pos is not None and not collision_markers_drawn[idx]:
                        g = 0 if n_agents == 1 else idx / (n_agents - 1)
                        ax.plot(collision_pos[0], collision_pos[1], marker='x', 
                               color=[1-g, 0+g, 0], markersize=8, markeredgewidth=3)
                        line_objects[idx].set_data(x[:k], y[:k])
                        collision_markers_drawn[idx] = True
                else:
                    line_objects[idx].set_data(x[:k+1], y[:k+1])
        
        # 動的障害物の更新（Graph_xと同じ）
        if 'dynamic' in P and P['dynamic'] and k % frame_interval == 0:
            current_time = k * P['dt']
            
            for obs_idx in range(len(ax._dynamic_patches)):
                waypoints_for_draw = ax._waypoints_list[obs_idx]
                seg_times_for_draw = ax._seg_times_list[obs_idx]
                base_circle = ax._base_circles[obs_idx]
                
                cumsum_times = np.cumsum(seg_times_for_draw)
                current_segment = np.searchsorted(cumsum_times, current_time, side='right')
                
                if current_segment >= len(waypoints_for_draw):
                    current_pos = waypoints_for_draw[-1]
                    current_center = current_pos
                elif current_segment == 0:
                    original_center = base_circle.mean(axis=1)
                    seg_duration = seg_times_for_draw[0] if len(seg_times_for_draw) > 0 else 1.0
                    progress = current_time / seg_duration if seg_duration > 0 else 0
                    next_pos = waypoints_for_draw[0]
                    current_center = original_center * (1-progress) + next_pos * progress
                else:
                    prev_time = cumsum_times[current_segment-1] if current_segment > 0 else 0
                    seg_duration = seg_times_for_draw[current_segment] if current_segment < len(seg_times_for_draw) else seg_times_for_draw[-1]
                    progress = (current_time - prev_time) / seg_duration if seg_duration > 0 else 0
                    current_pos = waypoints_for_draw[current_segment-1] if current_segment > 0 else waypoints_for_draw[0]
                    next_pos = waypoints_for_draw[current_segment] if current_segment < len(waypoints_for_draw) else waypoints_for_draw[-1]
                    current_center = current_pos * (1-progress) + next_pos * progress
                
                current_center = np.asarray(current_center).reshape(2, 1)
                old_center = base_circle.mean(axis=1).reshape(2, 1)
                shifted_circle = base_circle - old_center + current_center
                ax._dynamic_patches[obs_idx].set_xy(shifted_circle.T)
        
        # 時刻表示
        current_time = k * P['dt']
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        if k % frame_interval == 0:
            plt.draw()
            fig.canvas.draw()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())
            image = image[:, :, :3]
            images.append(image.copy())
            plt.pause(0.001)
    
    # 各機体の終点マーカー
    for idx, agent in enumerate(agents_data):
        g = 0 if n_agents == 1 else idx / (n_agents - 1)
        trial_state = agent['trial_state']
        x = trial_state[0, :]
        y = trial_state[1, :]
        
        collision_pos = agent.get('collision_pos')
        print(f"Agent {idx}: collision_pos = {collision_pos}, collision_occurred = {agent.get('collision_occurred', False)}")
        if collision_pos is not None:
            print(f"  Plotting X at ({collision_pos[0]:.2f}, {collision_pos[1]:.2f})")
            ax.plot(collision_pos[0], collision_pos[1], marker='x', 
                    color=[1-g, 0+g, 0], markersize=12, markeredgewidth=3)
        else:
            ax.plot(x[-1], y[-1], marker='*', color=[1-g, 0+g, 0], markersize=6, linewidth=2)
    
    # 動的障害物のパッチを削除
    if 'dynamic' in P and P['dynamic'] and hasattr(ax, '_dynamic_patches'):
        for patch in ax._dynamic_patches:
            patch.remove()
        ax._dynamic_patches = []
        ax._base_circles = []
        ax._waypoints_list = []
        ax._seg_times_list = []
    
    plt.close(fig)
    if images:
        imageio.mimsave(gif_filename, images, duration=delay)
