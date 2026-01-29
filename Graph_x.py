import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

def draw_sphere_surface(ax, center, radius, color='red', alpha=0.5):
    """真の3D球体を描画（plot_surface使用）"""
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')


def draw_static_obstacles_3d(ax, P):
    """静的障害物（壁と球体）を3Dで描画"""
    if 'object' not in P:
        return
        
    obj = P['object']
    
    if obj.ndim == 3:
        n_obs = obj.shape[2]
        for i in range(n_obs):
            xv = obj[0, :, i]
            yv = obj[1, :, i]
            zv = obj[2, :, i]
            
            # 壁判定：点数が4つで、z座標が全て同じ（底面定義）
            if len(xv) == 4 and np.all(np.abs(zv - zv[0]) < 0.01):
                            # 壁は描画しない（3Dビューでは非表示）
                continue
            else:
                # 静的球体障害物：点群から中心と半径を計算
                center = np.array([xv.mean(), yv.mean(), zv.mean()])
                radius = np.sqrt((xv[0]-center[0])**2 + (yv[0]-center[1])**2 + (zv[0]-center[2])**2)
                draw_sphere_surface(ax, center, radius, color='blue', alpha=0.5)
    elif obj.ndim == 2:
        xv = obj[0, :]
        yv = obj[1, :]
        zv = obj[2, :]
        center = np.array([xv.mean(), yv.mean(), zv.mean()])
        radius = np.sqrt((xv[0]-center[0])**2 + (yv[0]-center[1])**2 + (zv[0]-center[2])**2)
        draw_sphere_surface(ax, center, radius, color='blue', alpha=0.5)


def draw_static_obstacles_2d(ax, P, use_y=False, use_z=False, view_name='top'):
    """静的障害物を2Dで描画（平面ビュー用）"""
    if 'object' not in P:
        return
        
    obj = P['object']
    
    if obj.ndim == 3:
        n_obs = obj.shape[2]
        for i in range(n_obs):
            xv = obj[0, :, i]
            yv = obj[1, :, i]
            zv = obj[2, :, i]
            
            # 壁判定：点数が4つ
            if len(xv) == 4:
                # Top View（X-Y平面）のみ壁を描画
                if use_y and not use_z:
                    # X-Y平面：壁の輪郭を描画
                    polygon = Polygon(np.column_stack([xv, yv]), facecolor='gray', alpha=0.5, edgecolor='k', linewidth=1)
                    ax.add_patch(polygon)
                else:
                    # Side View（Y-Z, X-Z平面）では壁を非表示
                    continue
            else:
                # 球体は円として描画
                center_x = xv.mean()
                center_y = yv.mean()
                center_z = zv.mean()
                
                if use_y and use_z:
                    # Y-Z平面
                    radius = np.sqrt((yv[0]-center_y)**2 + (zv[0]-center_z)**2)
                    circle = plt.Circle((center_y, center_z), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                elif use_y and not use_z:
                    # X-Y平面
                    radius = np.sqrt((xv[0]-center_x)**2 + (yv[0]-center_y)**2)
                    circle = plt.Circle((center_x, center_y), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                else:
                    # X-Z平面
                    radius = np.sqrt((xv[0]-center_x)**2 + (zv[0]-center_z)**2)
                    circle = plt.Circle((center_x, center_z), radius, facecolor='blue', alpha=0.5, edgecolor='k', linewidth=1)
                ax.add_patch(circle)


def setup_3d_axis(ax, P):
    """3D軸の設定"""
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    ax.set_zlabel('Z[m]')
    
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
    
    # 軸範囲に基づいたアスペクト比を設定して球体を正球体として表示
    try:
        ax.set_box_aspect([x_range, y_range, z_range])
    except:
        # 古いmatplotlibバージョンの場合はスキップ
        pass


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
        # Y-Z平面
        return (y, z)
    elif use_y and not use_z:
        # X-Y平面
        return (x, y)
    else:
        # X-Z平面
        return (x, z)


def Graph_x(ds_state_list, P, agbp_list, bpc_list, collision_list=None):
    """メインのグラフ描画関数"""
    views = [
        (90, -90, 'top', 'Top View (X-Y)', True, True, False, True, True, False),        # 真上から（X-Y平面）
        (0, -90, 'side_y', 'Side View (Y-Z)', False, True, True, True, True, True),      # Y軸方向から（Y-Z平面）
        (0, 0, 'side_x', 'Side View (X-Z)', True, False, True, True, False, True),       # X軸方向から（X-Z平面）
        (30, -60, '3d', '3D View', True, True, True, False, False, False),               # 斜め（3D）
    ]
    
    for elev, azim, name, title, show_x, show_y, show_z, is_2d, use_y, use_z in views:
        print(f"'{name}' ビューのアニメーション生成中...")
        create_single_view_animation(ds_state_list, P, agbp_list, bpc_list, collision_list,
                                    elev, azim, name, title, show_x, show_y, show_z, is_2d, use_y, use_z)
    
    print("全てのアニメーション生成完了")


def create_single_view_animation(ds_state_list, P, agbp_list, bpc_list, collision_list,
                                  elev, azim, name, title, show_x=True, show_y=True, show_z=True, is_2d=False, use_y=False, use_z=False):
    """1つの視点のアニメーションを生成"""
    view_name = name  # ビュー名を保存
    fig = plt.figure(figsize=(10, 8))
    
    if is_2d:
        # 2D平面として描画
        ax = fig.add_subplot(111)
        ax.set_title(title)
        if 'axis' in P:
            if use_y and use_z:
                # Y-Z平面
                ax.set_xlim(P['axis'][2], P['axis'][3])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('Y[m]')
                ax.set_ylabel('Z[m]')
            elif use_y and not use_z:
                # X-Y平面（Top View）
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(P['axis'][2], P['axis'][3])
                ax.set_xlabel('X[m]')
                ax.set_ylabel('Y[m]')
            else:
                # X-Z平面
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('X[m]')
                ax.set_ylabel('Z[m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.5)
        
        # 静的障害物を2Dで描画
        draw_static_obstacles_2d(ax, P, use_y, use_z, view_name)
        
        # スタート・ゴール
        start_coords = get_2d_coords(ds_state_list[0][0,0], ds_state_list[0][0,1], ds_state_list[0][0,2], use_y, use_z)
        goal_coords = get_2d_coords(P['Goal_state'][0,0], P['Goal_state'][1,0], P['Goal_state'][2,0], use_y, use_z)
        ax.plot(start_coords[0], start_coords[1],
                'o', color=[0.5, 0, 1], markersize=10, markeredgecolor='k', markeredgewidth=2)
        ax.plot(goal_coords[0], goal_coords[1],
                'D', color=[0, 0, 1], markersize=10, markeredgecolor='k', markeredgewidth=2)
    else:
        # 3Dとして描画
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        setup_3d_axis(ax, P)
        
        # 視点に応じて不要な軸ラベルを非表示
        if not show_x:
            ax.set_xlabel('')
        if not show_y:
            ax.set_ylabel('')
        if not show_z:
            ax.set_zlabel('')
        
        # 静的障害物を3Dで描画
        draw_static_obstacles_3d(ax, P)
        
        # スタート・ゴール
        ax.scatter(ds_state_list[0][0,0], ds_state_list[0][0,1], ds_state_list[0][0,2],
                   color=[0.5, 0, 1], s=100, marker='o', edgecolors='k', linewidth=2)
        ax.scatter(P['Goal_state'][0,0], P['Goal_state'][1,0], P['Goal_state'][2,0],
                   color=[0, 0, 1], s=100, marker='D', edgecolors='k', linewidth=2)

    gif_filename = f'animation_{name}.gif'
    delay = 0.01
    images = []
    frame_interval = 0.5

    show_trajectory = True  # 経路表示フラグ（Trueで表示、Falseで非表示）
    
    # 経路用ラインオブジェクト
    line_objects = []
    if show_trajectory:
        for i in range(P['Trial_num']):
            if is_2d:
                line, = ax.plot([], [], color=[0, 1, 0], linewidth=2)
            else:
                line, = ax.plot([], [], [], color=[0, 1, 0], linewidth=2)
            line_objects.append(line)
    
    # ステップ数
    if show_trajectory:
        max_steps = sum(len(ds_state_list[i][:, 0]) for i in range(P['Trial_num']))
    else:
        max_steps = int(P['Trial_time'] / P['dt'])
    
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
            
            # 障害物ごとの色（赤系グラデーション、2Dと同様）
            color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
            color = [color_intensity, 0, 0]
            
            if is_2d:
                # 2D：円として描画（適切な平面に投影）
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
    
    # アニメーション用の状態変数
    collision_markers_drawn = [False] * P['Trial_num']
    goal_reached = [False] * P['Trial_num']
    trial_current_step = [0] * P['Trial_num']
    current_trial = 0
    
    goal_x = P['Goal_state'][0, 0]
    goal_y = P['Goal_state'][1, 0]
    goal_z = P['Goal_state'][2, 0]
    goal_threshold = P.get('goal_threshold', 0.2)
    
    # 時刻表示
    if is_2d:
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for k in range(max_steps):
        # 経路更新
        if show_trajectory:
            for i in range(P['Trial_num']):
                if goal_reached[i] or collision_markers_drawn[i]:
                    continue
                
                if i > 0:
                    prev_finished = goal_reached[i-1] or collision_markers_drawn[i-1]
                    if not prev_finished:
                        continue
                    if current_trial == i - 1:
                        current_trial = i
                
                ds_state = ds_state_list[i]
                x = ds_state[:, 0]
                y = ds_state[:, 1]
                z = ds_state[:, 2]
                agbp = agbp_list[i]
                bpc = bpc_list[i]
                
                step = trial_current_step[i]
                
                if step >= len(x):
                    goal_reached[i] = True
                    if i + 1 < P['Trial_num']:
                        current_trial = i + 1
                    continue
                
                # 目標到達チェック
                dist = np.sqrt((x[step]-goal_x)**2 + (y[step]-goal_y)**2 + (z[step]-goal_z)**2)
                if dist <= goal_threshold:
                    goal_reached[i] = True
                    if is_2d:
                        coords_x = []
                        coords_y = []
                        for s in range(step+1):
                            c = get_2d_coords(x[s], y[s], z[s], use_y, use_z)
                            coords_x.append(c[0])
                            coords_y.append(c[1])
                        line_objects[i].set_data(coords_x, coords_y)
                    else:
                        line_objects[i].set_data_3d(x[:step+1], y[:step+1], z[:step+1])
                    if i + 1 < P['Trial_num']:
                        current_trial = i + 1
                    continue
                
                # 衝突チェック
                is_collision = False
                if step + 1 >= len(x) or z[step+1] == 0:
                    is_collision = True
                
                if is_collision:
                    if collision_list is not None and i < len(collision_list) and collision_list[i] is not None:
                        collision_pos = collision_list[i]
                        g = 0 if P['Trial_num'] == 1 else (i)/(P['Trial_num']-1)
                        if is_2d:
                            coll_coords = get_2d_coords(collision_pos[0], collision_pos[1], collision_pos[2], use_y, use_z)
                            ax.plot(coll_coords[0], coll_coords[1],
                                   'x', color=[0, 1, 0], markersize=15, markeredgewidth=3)
                            coords_x = []
                            coords_y = []
                            for s in range(step):
                                c = get_2d_coords(x[s], y[s], z[s], use_y, use_z)
                                coords_x.append(c[0])
                                coords_y.append(c[1])
                            line_objects[i].set_data(coords_x, coords_y)
                        else:
                            ax.scatter(collision_pos[0], collision_pos[1], collision_pos[2],
                                       color=[0, 1, 0], marker='x', s=100, linewidths=3)
                            line_objects[i].set_data_3d(x[:step], y[:step], z[:step])
                        collision_markers_drawn[i] = True
                        if i + 1 < P['Trial_num']:
                            current_trial = i + 1
                else:
                    if is_2d:
                        coords_x = []
                        coords_y = []
                        for s in range(step+1):
                            c = get_2d_coords(x[s], y[s], z[s], use_y, use_z)
                            coords_x.append(c[0])
                            coords_y.append(c[1])
                        line_objects[i].set_data(coords_x, coords_y)
                    else:
                        line_objects[i].set_data_3d(x[:step+1], y[:step+1], z[:step+1])
                    
                    # ロック発生座標マーカー
                    if agbp is not None and bpc is not None and bpc > 0:
                        for idx in range(bpc):
                            if abs(x[step] - agbp[0, idx]) < 1e-6 and abs(y[step] - agbp[1, idx]) < 1e-6:
                                if is_2d:
                                    lock_coords = get_2d_coords(agbp[0, idx], agbp[1, idx], agbp[2, idx], use_y, use_z)
                                    ax.plot(lock_coords[0], lock_coords[1],
                                           's', color=[0, 0, 1], markersize=8, markeredgecolor='k')
                                else:
                                    ax.scatter(agbp[0, idx], agbp[1, idx], agbp[2, idx],
                                              marker='s', color=[0, 0, 1], s=50, edgecolors='k')
                    
                    trial_current_step[i] += 1
        
        # 動的障害物の更新
        if 'dynamic' in P and P['dynamic'] and k % frame_interval == 0:
            if show_trajectory:
                current_time = trial_current_step[current_trial] * P['dt']
            else:
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
        if show_trajectory:
            elapsed = trial_current_step[current_trial] * P['dt']
            if is_2d:
                time_text.set_text(f'Time:{elapsed:.1f}s')
            else:
                time_text.set_text(f'Time:{elapsed:.1f}s')
        else:
            time_text.set_text(f'Time: {k * P["dt"]:.1f}s')
        
        # フレーム保存
        if k % frame_interval == 0:
            fig.canvas.draw()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())[:, :, :3]
            images.append(image.copy())
    
    # 終点マーカー
    for i in range(P['Trial_num']):
        g = 0 if P['Trial_num'] == 1 else (i)/(P['Trial_num']-1)
        ds_state = ds_state_list[i]
        x = ds_state[:, 0]
        y = ds_state[:, 1]
        z = ds_state[:, 2]
        
        if collision_list is not None and i < len(collision_list) and collision_list[i] is not None:
            collision_pos = collision_list[i]
            if is_2d:
                end_2d = get_2d_coords(collision_pos[0], collision_pos[1], collision_pos[2], use_y, use_z)
                ax.plot(end_2d[0], end_2d[1],
                       'x', color=[1-g, 0+g, 0], markersize=20, markeredgewidth=3)
            else:
                ax.scatter(collision_pos[0], collision_pos[1], collision_pos[2],
                           color=[1-g, 0+g, 0], marker='x', s=150, linewidths=3)
        else:
            if is_2d:
                end_2d = get_2d_coords(x[-1], y[-1], z[-1], use_y, use_z)
                ax.plot(end_2d[0], end_2d[1], '*', color=[1-g, 0+g, 0], markersize=12, markeredgecolor='k', markeredgewidth=2)
            else:
                ax.scatter(x[-1], y[-1], z[-1], color=[1-g, 0+g, 0], marker='*', s=80, edgecolors='k', linewidths=2)
    
    plt.close(fig)
    
    if images:
        imageio.mimsave(gif_filename, images, duration=delay)
        print(f"  '{gif_filename}' を保存しました")

