"""
マルチエージェント用グラフ描画（Graph_xと同じ4視点構造）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg


def draw_wall_3d(ax, base_points, wall_height, color='blue', alpha=0.5):
    """壁を3Dで描画"""
    xv = base_points[0, :]
    yv = base_points[1, :]
    n_pts = len(xv)
    
    bottom = np.array([xv, yv, np.zeros(n_pts)])
    top = np.array([xv, yv, np.full(n_pts, wall_height)])
    
    for i in range(n_pts):
        j = (i + 1) % n_pts
        face = [
            [bottom[0, i], bottom[1, i], bottom[2, i]],
            [bottom[0, j], bottom[1, j], bottom[2, j]],
            [top[0, j], top[1, j], top[2, j]],
            [top[0, i], top[1, i], top[2, i]],
        ]
        ax.add_collection3d(Poly3DCollection([face], facecolors=color, alpha=alpha, edgecolors='k', linewidth=0.5))
    
    top_face = [list(zip(top[0, :], top[1, :], top[2, :]))]
    ax.add_collection3d(Poly3DCollection(top_face, facecolors=color, alpha=alpha*0.5, edgecolors='k', linewidth=0.5))


def draw_sphere_surface(ax, center, radius, color='red', alpha=0.5):
    """3D球体を描画"""
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')


def get_dynamic_obstacle_center(base_circle):
    """動的障害物の点群から中心座標を計算"""
    return np.array([base_circle[0, :].mean(), base_circle[1, :].mean(), base_circle[2, :].mean()])


def get_dynamic_obstacle_radius(base_circle):
    """動的障害物の点群から半径を計算"""
    center = get_dynamic_obstacle_center(base_circle)
    return np.sqrt((base_circle[0, 0]-center[0])**2 + (base_circle[1, 0]-center[1])**2 + (base_circle[2, 0]-center[2])**2)


def get_2d_coords(x, y, z, use_y, use_z):
    """3D座標を2D座標に変換"""
    if use_y and use_z:
        return (y, z)
    elif use_y and not use_z:
        return (x, y)
    else:
        return (x, z)


def draw_static_obstacles_2d(ax, P, use_y=False, use_z=False, view_name='top'):
    """静的障害物を2Dで描画"""
    if 'object' not in P:
        return
        
    obj = P['object']
    
    if obj.ndim == 3:
        n_obs = obj.shape[2]
        for i in range(n_obs):
            xv = obj[0, :, i]
            yv = obj[1, :, i]
            zv = obj[2, :, i]
            
            if len(xv) == 4:
                # Top View（X-Y平面）のみ壁を描画
                if use_y and not use_z:
                    polygon = Polygon(np.column_stack([xv, yv]), facecolor='gray', alpha=0.5, edgecolor='k', linewidth=1)
                    ax.add_patch(polygon)
                else:
                    continue
            else:
                # 球体は円として描画
                center_x = xv.mean()
                center_y = yv.mean()
                center_z = zv.mean()
                
                if use_y and use_z:
                    radius = np.sqrt((yv[0]-center_y)**2 + (zv[0]-center_z)**2)
                    circle = plt.Circle((center_y, center_z), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                elif use_y and not use_z:
                    radius = np.sqrt((xv[0]-center_x)**2 + (yv[0]-center_y)**2)
                    circle = plt.Circle((center_x, center_y), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                else:
                    radius = np.sqrt((xv[0]-center_x)**2 + (zv[0]-center_z)**2)
                    circle = plt.Circle((center_x, center_z), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                ax.add_patch(circle)


def draw_static_obstacles_3d(ax, P):
    """静的障害物を3Dで描画"""
    if 'object' not in P:
        return
        
    obj = P['object']
    
    if obj.ndim == 3:
        n_obs = obj.shape[2]
        for i in range(n_obs):
            xv = obj[0, :, i]
            yv = obj[1, :, i]
            zv = obj[2, :, i]
            
            if len(xv) == 4 and np.all(np.abs(zv - zv[0]) < 0.01):
                # 壁は3Dビューで非表示
                pass
            else:
                center = np.array([xv.mean(), yv.mean(), zv.mean()])
                radius = np.sqrt((xv[0]-center[0])**2 + (yv[0]-center[1])**2 + (zv[0]-center[2])**2)
                draw_sphere_surface(ax, center, radius, color='gray', alpha=0.5)


def setup_3d_axis(ax, P):
    """3D軸の設定"""
    ax.set_xlabel('X[m]', fontsize=16)
    ax.set_ylabel('Y[m]', fontsize=16)
    ax.set_zlabel('Z[m]', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
    if 'axis' in P:
        x_range = P['axis'][1] - P['axis'][0]
        y_range = P['axis'][3] - P['axis'][2]
        ax.set_xlim(P['axis'][0], P['axis'][1])
        ax.set_ylim(P['axis'][2], P['axis'][3])
    else:
        x_range = 10
        y_range = 10
    
    if 'max_height' in P:
        z_range = P['max_height']
        ax.set_zlim(0, P['max_height'])
    else:
        z_range = 5.0
        ax.set_zlim(0, 5.0)
    
    try:
        ax.set_box_aspect([x_range, y_range, z_range])
    except:
        pass


def Graph_MultiAgent(agents_data, save_dir='Result_Multi_Animation'):
    """マルチエージェント軌跡アニメーション（4視点）"""
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"フォルダ作成: {save_dir}")

    P = agents_data[0]['P']
    
    # 4視点の設定
    views = [
        (90, -90, 'top', 'Top View (X-Y)', True, True, False, True, True, False),
        (0, -90, 'side_y', 'Side View (Y-Z)', False, True, True, True, True, True),
        (0, 0, 'side_x', 'Side View (X-Z)', True, False, True, True, False, True),
        (30, -60, '3d', '3D View', True, True, True, False, False, False),
    ]
    
    for elev, azim, name, title, show_x, show_y, show_z, is_2d, use_y, use_z in views:
        print(f"'{name}' ビューのアニメーション生成中...")
        create_single_view_animation(agents_data, P, elev, azim, name, title, 
                                    is_2d=is_2d, use_y=use_y, use_z=use_z,
                                    save_dir=save_dir)
    
    print("全てのアニメーション生成完了")


def create_single_view_animation(agents_data, P, elev, azim, name, title, 
                                is_2d=False, use_y=False, use_z=False,
                                save_dir='Result_Multi_Animation'):
    """単一視点のアニメーション生成"""
    import os
    view_name = name
    n_agents = len(agents_data)
    gif_filename = os.path.join(save_dir, f"multi_agent_animation_{name}.gif")
    delay = 0.1  # フレーム間の遅延時間（秒） 
    
    fig = plt.figure(figsize=(10, 8))
    
    if is_2d:
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=18)
        if 'axis' in P:
            if use_y and use_z:
                ax.set_xlim(P['axis'][2], P['axis'][3])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('Y[m]', fontsize=18)
                ax.set_ylabel('Z[m]', fontsize=18)
            elif use_y and not use_z:
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(P['axis'][2], P['axis'][3])
                ax.set_xlabel('X[m]', fontsize=18)
                ax.set_ylabel('Y[m]', fontsize=18)
            else:
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('X[m]', fontsize=18)
                ax.set_ylabel('Z[m]', fontsize=18)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # 静的障害物を2Dで描画
        draw_static_obstacles_2d(ax, P, use_y, use_z, view_name)
        
        # スタート・ゴール
        agent_colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
        for idx, agent in enumerate(agents_data):
            start_coords = get_2d_coords(agent['P']['Init_State'][0,0], agent['P']['Init_State'][1,0], 
                                        agent['P']['Init_State'][2,0], use_y, use_z)
            agent_color = agent_colors[idx % len(agent_colors)]
            ax.plot(start_coords[0], start_coords[1], 'o', color=agent_color, markersize=10, 
                   markeredgecolor='k', markeredgewidth=2)
        
        # ゴール（共通）
        if 'original_goal' in agents_data[0]:
            goal_pos = agents_data[0]['original_goal']
        else:
            goal_pos = agents_data[0]['P']['Goal_state'][0:3, 0]
        goal_coords = get_2d_coords(goal_pos[0], goal_pos[1], goal_pos[2], use_y, use_z)
        ax.plot(goal_coords[0], goal_coords[1], 'D', color=[0, 0, 1], markersize=10, 
               markeredgecolor='k', markeredgewidth=2)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=18)
        setup_3d_axis(ax, P)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
        
        # 静的障害物を3Dで描画
        draw_static_obstacles_3d(ax, P)
        
        # スタート・ゴール
        agent_colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
        for idx, agent in enumerate(agents_data):
            agent_color = agent_colors[idx % len(agent_colors)]
            ax.scatter(agent['P']['Init_State'][0,0], agent['P']['Init_State'][1,0], 
                      agent['P']['Init_State'][2,0], color=agent_color, s=100, marker='o', 
                      edgecolors='k', linewidths=2)
        
        # ゴール（共通）
        if 'original_goal' in agents_data[0]:
            goal_pos = agents_data[0]['original_goal']
        else:
            goal_pos = agents_data[0]['P']['Goal_state'][0:3, 0]
        ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color=[0, 0, 1], s=100, 
                  marker='D', edgecolors='k', linewidths=2)
    
    # 経路線の初期化
    agent_colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
    line_objects = []
    for idx in range(n_agents):
        agent_color = agent_colors[idx % len(agent_colors)]
        if is_2d:
            line, = ax.plot([], [], color=agent_color, linewidth=2)
        else:
            line, = ax.plot([], [], [], color=agent_color, linewidth=2)
        line_objects.append(line)
    
    # 時刻表示
    if is_2d:
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=16, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=16, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 最大ステップ数
    max_steps = max(agent['trial_state'].shape[1] for agent in agents_data)
    
    # 各機体の状態追跡（ゴール到達・衝突判定用）
    collision_markers_drawn = [False] * n_agents
    goal_reached = [False] * n_agents
    
    # 動的障害物の準備
    dynamic_objects = []
    sphere_data = []
    
    if 'dynamic' in P and P['dynamic']:
        dyn_obj = P['dynamic_obj']
        
        if dyn_obj.ndim == 3:
            n_obstacles = dyn_obj.shape[2]
        else:
            n_obstacles = 1
            dyn_obj = np.expand_dims(dyn_obj, axis=2)
        
        for obs_idx in range(n_obstacles):
            base_circle = dyn_obj[:, :, obs_idx]
            center = get_dynamic_obstacle_center(base_circle)
            radius = get_dynamic_obstacle_radius(base_circle)
            
            # waypoint/segment_timeデータを取得
            if isinstance(P.get('dynamic_waypoints'), list):
                waypoints = np.asarray(P['dynamic_waypoints'][obs_idx])
            else:
                waypoints = np.asarray(P.get('dynamic_waypoints'))

            if isinstance(P.get('dynamic_segment_times'), list):
                seg_times = np.asarray(P['dynamic_segment_times'][obs_idx])
            else:
                seg_times = np.asarray(P.get('dynamic_segment_times'))
            
            # 障害物ごとの色（赤系グラデーション）
            color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
            color = [color_intensity, 0, 0]
            
            if is_2d:
                # 2D：円として描画
                coords_2d = get_2d_coords(center[0], center[1], center[2], use_y, use_z)
                circle = plt.Circle(coords_2d, radius, facecolor=color, alpha=0.5, edgecolor='k', linewidth=1)
                ax.add_patch(circle)
                dynamic_objects.append(circle)
            else:
                # 3D：球体として描画
                sphere = draw_sphere_surface(ax, center, radius, color=color, alpha=0.5)
                dynamic_objects.append(sphere)
            
            sphere_data.append({
                'base_center': center,
                'radius': radius,
                'waypoints': waypoints,
                'seg_times': seg_times,
                'color': color,
            })
    
    # アニメーションフレーム生成
    images = []
    frame_interval = 1  # フレーム間隔
    
    for k in range(0, int(max_steps), int(frame_interval)):
        # 各機体の経路を更新（2D版と同じロジック）
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
            
            # 目標到達チェック（3D版、original_goalを使用）
            if 'original_goal' in agent:
                goal_pos_3d = agent['original_goal']
            else:
                goal_pos_3d = agent['P']['Goal_state'][0:3, 0]
            
            goal_threshold = agent['P'].get('goal_threshold', 0.2)
            distance_to_goal = np.sqrt((x[k] - goal_pos_3d[0])**2 + 
                                      (y[k] - goal_pos_3d[1])**2 + 
                                      (z[k] - goal_pos_3d[2])**2)
            
            if distance_to_goal <= goal_threshold:
                goal_reached[idx] = True
                # 最後の経路を描画
                if is_2d:
                    coords_2d = [get_2d_coords(x[i], y[i], z[i], use_y, use_z) for i in range(k+1)]
                    if coords_2d:
                        xs, ys = zip(*coords_2d)
                        line_objects[idx].set_data(xs, ys)
                else:
                    line_objects[idx].set_data_3d(x[:k+1], y[:k+1], z[:k+1])
                continue
            
            # 衝突チェック（2D版と同じ）
            is_collision = False
            if k + 1 >= len(x):
                is_collision = True
            elif z[k+1] == 0:
                is_collision = True
            
            if is_collision:
                collision_pos = agent.get('collision_pos')
                if collision_pos is not None and not collision_markers_drawn[idx]:
                    agent_color = agent_colors[idx % len(agent_colors)]
                    if is_2d:
                        coll_coords = get_2d_coords(collision_pos[0], collision_pos[1], collision_pos[2], use_y, use_z)
                        ax.plot(coll_coords[0], coll_coords[1], marker='x', 
                               color=agent_color, markersize=15, markeredgewidth=3)
                        coords_2d = [get_2d_coords(x[i], y[i], z[i], use_y, use_z) for i in range(k)]
                        if coords_2d:
                            xs, ys = zip(*coords_2d)
                            line_objects[idx].set_data(xs, ys)
                    else:
                        ax.scatter(collision_pos[0], collision_pos[1], collision_pos[2],
                                  color=agent_color, marker='x', s=150, linewidths=3)
                        line_objects[idx].set_data_3d(x[:k], y[:k], z[:k])
                    collision_markers_drawn[idx] = True
            else:
                # 経路を更新
                if is_2d:
                    coords_2d = [get_2d_coords(x[i], y[i], z[i], use_y, use_z) for i in range(k+1)]
                    if coords_2d:
                        xs, ys = zip(*coords_2d)
                        line_objects[idx].set_data(xs, ys)
                else:
                    line_objects[idx].set_data_3d(x[:k+1], y[:k+1], z[:k+1])
        
        # 動的障害物の更新
        if 'dynamic' in P and P['dynamic']:
            current_time = k * P['dt']
            
            for obs_idx, data in enumerate(sphere_data):
                waypoints = data['waypoints']
                seg_times = data['seg_times']
                base_center = data['base_center']
                radius = data['radius']
                color = data['color']
                
                # 現在位置を計算
                cumsum_times = np.cumsum(seg_times)
                current_segment = np.searchsorted(cumsum_times, current_time, side='right')
                
                if current_segment >= len(waypoints):
                    new_center = waypoints[-1]
                elif current_segment == 0:
                    seg_duration = seg_times[0] if len(seg_times) > 0 else 1.0
                    progress = current_time / seg_duration if seg_duration > 0 else 0
                    new_center = base_center * (1-progress) + waypoints[0] * progress
                else:
                    prev_time = cumsum_times[current_segment-1] if current_segment > 0 else 0
                    seg_duration = seg_times[current_segment] if current_segment < len(seg_times) else seg_times[-1]
                    progress = (current_time - prev_time) / seg_duration if seg_duration > 0 else 0
                    current_pos = waypoints[current_segment-1] if current_segment > 0 else waypoints[0]
                    next_pos = waypoints[current_segment] if current_segment < len(waypoints) else waypoints[-1]
                    new_center = current_pos * (1-progress) + next_pos * progress
                
                # 障害物を更新
                if is_2d:
                    # 2D：円の中心を更新
                    new_coords_2d = get_2d_coords(new_center[0], new_center[1], new_center[2], use_y, use_z)
                    dynamic_objects[obs_idx].center = new_coords_2d
                else:
                    # 3D：古い球体を削除して新しい位置に描画
                    dynamic_objects[obs_idx].remove()
                    dynamic_objects[obs_idx] = draw_sphere_surface(ax, new_center, radius, color=color, alpha=0.5)
        
        # 時刻表示更新
        time_text.set_text(f'Time: {k * P["dt"]:.1f}s')
        
        # フレーム保存
        fig.canvas.draw()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(image.copy())
    
    plt.close(fig)
    
    if images:
        imageio.mimsave(gif_filename, images, duration=delay)
        print(f"  '{gif_filename}' を保存しました")

