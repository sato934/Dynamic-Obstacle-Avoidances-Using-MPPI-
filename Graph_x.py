import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from check import inpolygon_numba, point_line_segment_distance

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

def Graph_x(ds_state_list, P, agbp_list, bpc_list, collision_list=None, obs_list=None,
            save_dir='Result_Single_Animation'):
    """メインのグラフ描画関数"""
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"フォルダ作成: {save_dir}")

    views = [
        (90, -90, 'top', 'Top View (X-Y)', True, True, False, True, True, False),        # 真上から（X-Y平面）
        (0, -90, 'side_y', 'Side View (Y-Z)', False, True, True, True, True, True),      # Y軸方向から（Y-Z平面）
        (0, 0, 'side_x', 'Side View (X-Z)', True, False, True, True, False, True),       # X軸方向から（X-Z平面）
        (30, -60, '3d', '3D View', True, True, True, False, False, False),               # 斜め（3D）
    ]
    
    for elev, azim, name, title, show_x, show_y, show_z, is_2d, use_y, use_z in views:
        print(f"'{name}' ビューのアニメーション生成中...")
        create_single_view_animation(ds_state_list, P, agbp_list, bpc_list, collision_list, obs_list,
                                    elev, azim, name, title, show_x, show_y, show_z, is_2d, use_y, use_z,
                                    save_dir=save_dir)
    
    print("全てのアニメーション生成完了")
    
    # 各試行の最終状態を4視点で1枚の画像に保存
    print("\n最終状態画像を生成中...")
    save_final_state_4views(ds_state_list, P, collision_list, obs_list, save_dir=save_dir)
    print("最終状態画像の生成完了")

def create_single_view_animation(ds_state_list, P, agbp_list, bpc_list, collision_list, obs_list,
                                  elev, azim, name, title, show_x=True, show_y=True, show_z=True, is_2d=False, use_y=False, use_z=False,
                                  save_dir='Result_Single_Animation'):
    """1つの視点のアニメーションを生成"""
    view_name = name  # ビュー名を保存
    fig = plt.figure(figsize=(10, 8))
    
    if is_2d:
        # 2D平面として描画
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=15)
        if 'axis' in P:
            if use_y and use_z:
                # Y-Z平面
                ax.set_xlim(P['axis'][2], P['axis'][3])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('Y[m]',fontsize=15)
                ax.set_ylabel('Z[m]',fontsize=15)
            elif use_y and not use_z:
                # X-Y平面（Top View）
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(P['axis'][2], P['axis'][3])
                ax.set_xlabel('X[m]',fontsize=15)
                ax.set_ylabel('Y[m]',fontsize=15)
            else:
                # X-Z平面
                ax.set_xlim(P['axis'][0], P['axis'][1])
                ax.set_ylim(0, P['max_height'])
                ax.set_xlabel('X[m]',fontsize=15)
                ax.set_ylabel('Z[m]',fontsize=15)
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
        info_text = ax.text(0.85, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # 3Dとして描画
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=15)
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
        info_text = ax.text2D(0.85, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    import os
    gif_filename = os.path.join(save_dir, f'single_animation_{name}.gif')
    delay = 0.01
    images = []
    durations = []
    frame_interval = 0.5
    
    target_obs_list = obs_list if obs_list is not None else [P] * P['Trial_num']

    #試行ごとのループ
    for trial_idx in range(P['Trial_num']):
        print(f"  Drawing Trial {trial_idx + 1}/{P['Trial_num']}...")
        
        # 1. 前の試行のアーティストを削除するためのリスト
        current_artists = []
        
        # 2. 現在の試行のパラメータ取得
        current_P = target_obs_list[trial_idx]
        ds_state = ds_state_list[trial_idx]
        agbp = agbp_list[trial_idx]
        bpc = bpc_list[trial_idx]
        
        # 3. 動的障害物の初期化
        current_obstacles = [] # 更新用リスト
        current_obs_data = []  # データ保持用
        
        if 'dynamic' in current_P and current_P['dynamic']:
            dyn_obj = current_P['dynamic_obj']
            if dyn_obj.ndim == 3: n_obstacles = dyn_obj.shape[2]
            else: n_obstacles = 1; dyn_obj = np.expand_dims(dyn_obj, axis=2)
            
            for obs_idx in range(n_obstacles):
                base_circle = dyn_obj[:, :, obs_idx]
                center = get_dynamic_obstacle_center(base_circle)
                radius = get_dynamic_obstacle_radius(base_circle)
                
                if isinstance(current_P.get('dynamic_waypoints'), list): waypoints = np.asarray(current_P['dynamic_waypoints'][obs_idx])
                else: waypoints = np.asarray(current_P.get('dynamic_waypoints'))
                
                if isinstance(current_P.get('dynamic_segment_times'), list): seg_times = np.asarray(current_P['dynamic_segment_times'][obs_idx])
                else: seg_times = np.asarray(current_P.get('dynamic_segment_times'))
                
                # 障害物ごとの色（赤系グラデーション）
                color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
                color = [color_intensity, 0, 0]
                
                if is_2d:
                    coords_2d = get_2d_coords(center[0], center[1], center[2], use_y, use_z)
                    patch = plt.Circle(coords_2d, radius, facecolor=color, alpha=0.5, edgecolor='k', linewidth=1)
                    ax.add_patch(patch)
                    current_obstacles.append(patch)
                    current_artists.append(patch)
                else:
                    sphere = draw_sphere_surface(ax, center, radius, color=color, alpha=0.5)
                    current_obstacles.append(sphere)
                    current_artists.append(sphere)
                
                current_obs_data.append({
                    'base_center': center, 'radius': radius, 'waypoints': waypoints, 'seg_times': seg_times, 'color': color
                })
        
        # 4. 経路ラインの初期化
        color_line = [0, 1, 0] # 緑色
        if is_2d:
            line, = ax.plot([], [], color=color_line, linewidth=2)
        else:
            line, = ax.plot([], [], [], color=color_line, linewidth=2)
        current_artists.append(line)
        
        # 5. ステップごとの描画ループ
        x = ds_state[:, 0]; y = ds_state[:, 1]; z = ds_state[:, 2]
        max_steps = len(x)
        goal_reached = False
        collision_occurred = False
        
        
        
        for k in range(max_steps):
            if k % frame_interval != 0: continue # 間引き
            
            current_time = k * P['dt']
            info_text.set_text(f'Trial: {trial_idx + 1}/{P["Trial_num"]}\nTime: {current_time:.1f}s')
            
            # 経路更新
            if not (goal_reached or collision_occurred):
                # ゴール判定
                dist_g = np.sqrt((x[k]-P['Goal_state'][0,0])**2 + (y[k]-P['Goal_state'][1,0])**2 + (z[k]-P['Goal_state'][2,0])**2)
                if dist_g <= P.get('goal_threshold', 0.2):
                    goal_reached = True
                
                # 衝突判定（簡易チェック）
                if 0 < k+1 < max_steps and z[k+1] == 0: # 墜落判定
                     collision_occurred = True
                     break
                
                # ライン更新
                if is_2d:
                    cx, cy = [], []
                    for s in range(k+1): c = get_2d_coords(x[s], y[s], z[s], use_y, use_z); cx.append(c[0]); cy.append(c[1])
                    line.set_data(cx, cy)
                else:
                    line.set_data_3d(x[:k+1], y[:k+1], z[:k+1])
                
                # ロック地点（あれば）
                if agbp is not None and bpc > 0:
                    for idx in range(bpc):
                        if abs(x[k] - agbp[0, idx]) < 1e-6 and abs(y[k] - agbp[1, idx]) < 1e-6:
                            if is_2d:
                                lc = get_2d_coords(agbp[0, idx], agbp[1, idx], agbp[2, idx], use_y, use_z)
                                m, = ax.plot(lc[0], lc[1], 's', color=[0, 0, 1], markersize=8, markeredgecolor='k')
                            else:
                                m = ax.scatter(agbp[0, idx], agbp[1, idx], agbp[2, idx], marker='s', color=[0, 0, 1], s=50, edgecolors='k')
                            current_artists.append(m)

            # 動的障害物の更新
            for obs_idx, data in enumerate(current_obs_data):
                waypoints = data['waypoints']; seg_times = data['seg_times']
                base_center = data['base_center']; radius = data['radius']; color = data['color']
                
                cumsum_times = np.cumsum(seg_times)
                current_segment = np.searchsorted(cumsum_times, current_time, side='right')
                
                if current_segment >= len(waypoints): new_center = waypoints[-1]
                elif current_segment == 0:
                    seg_dur = seg_times[0] if len(seg_times) > 0 else 1.0
                    prog = current_time / seg_dur if seg_dur > 0 else 0
                    new_center = base_center * (1-prog) + waypoints[0] * prog
                else:
                    prev_t = cumsum_times[current_segment-1]
                    seg_dur = seg_times[current_segment] if current_segment < len(seg_times) else seg_times[-1]
                    prog = (current_time - prev_t) / seg_dur if seg_dur > 0 else 0
                    curr_p = waypoints[current_segment-1]; next_p = waypoints[current_segment] if current_segment < len(waypoints) else waypoints[-1]
                    new_center = curr_p * (1-prog) + next_p * prog
                
                if is_2d:
                    new_c_2d = get_2d_coords(new_center[0], new_center[1], new_center[2], use_y, use_z)
                    current_obstacles[obs_idx].center = new_c_2d
                else:
                    current_obstacles[obs_idx].remove()
                    # 3Dの場合はremoveしたあとリストから消えるわけではないが、再描画したものをリストに入れ直す
                    # ただしartistsリストには古いものをremove済みとして扱いたいが、collectionなのでremove()で消える
                    new_sphere = draw_sphere_surface(ax, new_center, radius, color=color, alpha=0.5)
                    current_obstacles[obs_idx] = new_sphere
                    # 新しいsphereも次回のループで消すために登録が必要だが、
                    # 3Dのsurfaceは更新毎に作り直すので、ここでは「最後に残ったもの」だけ消せば良いわけではない
                    # update毎に古いものを消して新しいものを描いているので、ループ内での管理が必要
                    # current_obstacles[obs_idx] は常に「現在画面にあるオブジェクト」を指すようにする
            
            # フレーム保存
            fig.canvas.draw()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())[:, :, :3]
            images.append(image.copy())
            durations.append(delay)
            
            
        # 試行終了時のマーカー（ゴール or 衝突）
        marker = None
        if collision_list is not None and trial_idx < len(collision_list) and collision_list[trial_idx] is not None:
            c_pos = collision_list[trial_idx]
            if is_2d:
                ce = get_2d_coords(c_pos[0], c_pos[1], c_pos[2], use_y, use_z)
                marker, = ax.plot(ce[0], ce[1], 'x', color='red', markersize=15, markeredgewidth=3)
            else:
                marker = ax.scatter(c_pos[0], c_pos[1], c_pos[2], color='red', marker='x', s=150, linewidths=3)
        
        fig.canvas.draw()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        final_image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        
        images.append(final_image.copy())
        durations.append(8.0) 
        
        # --- リセット処理 ---
        # 次の試行のために、この試行で描画した動的オブジェクトを削除
        line.remove()
        if marker: marker.remove()
        
        for artist in current_artists:
            try: artist.remove()
            except: pass
        
        # 3Dの動的障害物（current_obstaclesに残っているもの）を削除
        for obs in current_obstacles:
            try: obs.remove()
            except: pass
    
    plt.close(fig)
    
    if images:
        imageio.mimsave(gif_filename, images, duration=durations)
        print(f"  '{gif_filename}' を保存しました")

def save_final_state_4views(ds_state_list, P, collision_list=None, obs_list=None, save_dir='Result_Single_Animation'):
    """
    各試行の最終状態（目標到達時または終了時）を4視点で1枚の画像にまとめて保存
    配置: 左上=3D, 右上=Top(X-Y), 左下=Y-Z, 右下=X-Z
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"フォルダ作成: {save_dir}")
    
    def _find_first_obstacle_collision(x, y, z, P_local):
        """パスの中で障害物に最初に衝突するステップを探す"""
        if 'object' not in P_local or P_local['object'].size == 0:
            return None, None
        agent_radius = P_local.get('agent_radius', 0.3)
        wall_height = P_local.get('object_height', P_local.get('wall_height', 3.0))
        n_obs = P_local['object'].shape[2]
        for step in range(len(x)):
            if step > 0 and x[step] == 0 and y[step] == 0 and z[step] == 0:
                break  # ゼロ埋め領域
            for obj_idx in range(n_obs):
                xv = P_local['object'][0, :, obj_idx]
                yv = P_local['object'][1, :, obj_idx]
                if 0 <= z[step] <= wall_height:
                    if inpolygon_numba(x[step], y[step], xv, yv):
                        return step, np.array([x[step], y[step], z[step]])
                    n_verts = len(xv)
                    for e in range(n_verts):
                        e_next = (e + 1) % n_verts
                        dist = point_line_segment_distance(
                            x[step], y[step], xv[e], yv[e], xv[e_next], yv[e_next])
                        if dist < agent_radius:
                            return step, np.array([x[step], y[step], z[step]])
        return None, None
    
    # 配置: (subplot位置, elev, azim, title, is_2d, use_y, use_z)
    # subplot: 1=左上, 2=右上, 3=左下, 4=右下
    views = [
        (1, 30, -60, '3D View', False, False, False),         # 左上: 3D View
        (2, 90, -90, 'Top View (X-Y)', True, True, False),    # 右上: Top View
        (3, 0, -90, 'Side View (Y-Z)', True, True, True),     # 左下: Y-Z平面
        (4, 0, 0, 'Side View (X-Z)', True, False, True),      # 右下: X-Z平面
    ]
    
    target_obs_list = obs_list if obs_list is not None else [P] * len(ds_state_list)
    
    for trial_idx in range(len(ds_state_list)):
        print(f"試行 {trial_idx + 1}/{len(ds_state_list)} の最終状態画像を作成中...")
        
        current_P = target_obs_list[trial_idx]
        ds_state = ds_state_list[trial_idx]
        
        # 目標到達ステップを検出
        x = ds_state[:, 0]; y = ds_state[:, 1]; z = ds_state[:, 2]
        goal_pos = P['Goal_state'][0:3, 0]
        goal_threshold = P.get('goal_threshold', 0.2)
        
        final_step = len(x) - 1
        goal_reached = False
        has_collision = (collision_list is not None and trial_idx < len(collision_list) and collision_list[trial_idx] is not None)
        
        # 衝突時：最後の有効ステップを検出（ゼロ埋めされた範囲を除外）
        if has_collision:
            for step in range(len(x) - 1, 0, -1):
                if np.any(ds_state[step, :] != 0):
                    final_step = step
                    break
        
        for step in range(len(x)):
            dist_to_goal = np.sqrt((x[step]-goal_pos[0])**2 + (y[step]-goal_pos[1])**2 + (z[step]-goal_pos[2])**2)
            if dist_to_goal <= goal_threshold:
                final_step = step
                goal_reached = True
                break
        
        # 衝突位置の取得
        collision_pos = collision_list[trial_idx] if has_collision else None
        
        # 衝突が地面付近の場合、パスから障害物との実際の衝突地点を探索
        if has_collision and collision_pos is not None:
            min_height = P.get('min_height', 0.0)
            agent_radius = P.get('agent_radius', 0.3)
            # collision_posが地面/天井付近の場合、障害物との最初の衝突を探す
            if collision_pos[2] <= min_height + agent_radius + 0.5:
                obs_step, obs_pos = _find_first_obstacle_collision(x, y, z, current_P)
                if obs_step is not None:
                    collision_pos = obs_pos
                    final_step = obs_step
                    print(f"    障害物衝突地点を検出: step={obs_step}, pos=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}, {obs_pos[2]:.2f})")
        
        final_time = final_step * P['dt']
        
        # 4視点のfigureを作成（画面いっぱいに表示）
        fig = plt.figure(figsize=(18, 14))
        
        for subplot_pos, elev, azim, title, is_2d, use_y, use_z in views:
            if is_2d:
                ax = fig.add_subplot(2, 2, subplot_pos)
                ax.set_title(title, fontsize=14, fontweight='bold')
                
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
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, alpha=0.5)
                
                # 静的障害物
                draw_static_obstacles_2d(ax, P, use_y, use_z, title)
                
                # スタート・ゴール
                start_coords = get_2d_coords(x[0], y[0], z[0], use_y, use_z)
                goal_coords = get_2d_coords(goal_pos[0], goal_pos[1], goal_pos[2], use_y, use_z)
                ax.plot(start_coords[0], start_coords[1], 'o', color=[0.5, 0, 1], markersize=10, markeredgecolor='k', markeredgewidth=2)
                ax.plot(goal_coords[0], goal_coords[1], 'D', color=[0, 0, 1], markersize=10, markeredgecolor='k', markeredgewidth=2)
                
                # 経路（衝突時は1ステップ前まで）
                draw_end = max(final_step - 1, 0) if has_collision else final_step
                cx, cy = [], []
                for s in range(draw_end + 1):
                    c = get_2d_coords(x[s], y[s], z[s], use_y, use_z)
                    cx.append(c[0]); cy.append(c[1])
                ax.plot(cx, cy, color='green', linewidth=2)
                
                # 最終位置マーカー（衝突時はcollision_posの位置に×印）
                if collision_pos is not None:
                    coll_coords = get_2d_coords(collision_pos[0], collision_pos[1], collision_pos[2], use_y, use_z)
                    ax.plot(coll_coords[0], coll_coords[1], 'x', color='red', markersize=15, markeredgewidth=3)
                
                # 動的障害物（最終時刻の位置）
                if 'dynamic' in current_P and current_P['dynamic']:
                    _draw_dynamic_obstacles_2d_at_time(ax, current_P, final_time, use_y, use_z)
                    
            else:
                ax = fig.add_subplot(2, 2, subplot_pos, projection='3d')
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(title, fontsize=15, fontweight='bold')
                setup_3d_axis(ax, P)
                
                # 静的障害物
                draw_static_obstacles_3d(ax, P)
                
                # スタート・ゴール
                ax.scatter(x[0], y[0], z[0], color=[0.5, 0, 1], s=100, marker='o', edgecolors='k', linewidth=2)
                ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color=[0, 0, 1], s=100, marker='D', edgecolors='k', linewidth=2)
                
                # 経路（衝突時は1ステップ前まで）
                draw_end = max(final_step - 1, 0) if has_collision else final_step
                ax.plot(x[:draw_end+1], y[:draw_end+1], z[:draw_end+1], color='green', linewidth=2)
                
                # 最終位置マーカー（衝突時はcollision_posの位置に×印）
                if collision_pos is not None:
                    ax.scatter(collision_pos[0], collision_pos[1], collision_pos[2], color='red', marker='x', s=150, linewidths=3)
                
                # 動的障害物（最終時刻の位置）
                if 'dynamic' in current_P and current_P['dynamic']:
                    _draw_dynamic_obstacles_3d_at_time(ax, current_P, final_time)
        
        # タイトル
        status = "Goal Reached" if goal_reached else "Finished"
        if collision_list is not None and trial_idx < len(collision_list) and collision_list[trial_idx] is not None:
            status = "Collision"
        fig.suptitle(f'Trial {trial_idx + 1} - Final State ({status}, t={final_time:.2f}s)', fontsize=25, fontweight='bold')
        
        # 4つの画像を中央に寄せて適切な余白を確保
        plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12, wspace=0.25, hspace=0.25)
        
        # 保存
        save_path = os.path.join(save_dir, f'single_final_state_trial_{trial_idx + 1}.png')
        plt.savefig(save_path, dpi=300)
        print(f"  保存: {save_path}")
        plt.close(fig)
    
    print("全試行の最終状態画像を保存完了")

def _draw_dynamic_obstacles_2d_at_time(ax, P, t, use_y, use_z):
    """指定時刻の動的障害物を2Dで描画"""
    dyn_obj = P['dynamic_obj']
    if dyn_obj.ndim == 3:
        n_obstacles = dyn_obj.shape[2]
    else:
        n_obstacles = 1
        dyn_obj = np.expand_dims(dyn_obj, axis=2)
    
    for obs_idx in range(n_obstacles):
        base_circle = dyn_obj[:, :, obs_idx]
        base_center = get_dynamic_obstacle_center(base_circle)
        radius = get_dynamic_obstacle_radius(base_circle)
        
        if isinstance(P.get('dynamic_waypoints'), list):
            waypoints = np.asarray(P['dynamic_waypoints'][obs_idx])
        else:
            waypoints = np.asarray(P.get('dynamic_waypoints'))
        
        if isinstance(P.get('dynamic_segment_times'), list):
            seg_times = np.asarray(P['dynamic_segment_times'][obs_idx])
        else:
            seg_times = np.asarray(P.get('dynamic_segment_times'))
        
        # 現在の中心を計算
        new_center = _calc_dynamic_center_at_time(base_center, waypoints, seg_times, t)
        
        color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
        color = [color_intensity, 0, 0]
        
        coords_2d = get_2d_coords(new_center[0], new_center[1], new_center[2], use_y, use_z)
        patch = plt.Circle(coords_2d, radius, facecolor=color, alpha=0.5, edgecolor='k', linewidth=1)
        ax.add_patch(patch)

def _draw_dynamic_obstacles_3d_at_time(ax, P, t):
    """指定時刻の動的障害物を3Dで描画"""
    dyn_obj = P['dynamic_obj']
    if dyn_obj.ndim == 3:
        n_obstacles = dyn_obj.shape[2]
    else:
        n_obstacles = 1
        dyn_obj = np.expand_dims(dyn_obj, axis=2)
    
    for obs_idx in range(n_obstacles):
        base_circle = dyn_obj[:, :, obs_idx]
        base_center = get_dynamic_obstacle_center(base_circle)
        radius = get_dynamic_obstacle_radius(base_circle)
        
        if isinstance(P.get('dynamic_waypoints'), list):
            waypoints = np.asarray(P['dynamic_waypoints'][obs_idx])
        else:
            waypoints = np.asarray(P.get('dynamic_waypoints'))
        
        if isinstance(P.get('dynamic_segment_times'), list):
            seg_times = np.asarray(P['dynamic_segment_times'][obs_idx])
        else:
            seg_times = np.asarray(P.get('dynamic_segment_times'))
        
        new_center = _calc_dynamic_center_at_time(base_center, waypoints, seg_times, t)
        
        color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
        color = [color_intensity, 0, 0]
        
        draw_sphere_surface(ax, new_center, radius, color=color, alpha=0.5)

def _calc_dynamic_center_at_time(base_center, waypoints, seg_times, t):
    """指定時刻における動的障害物の中心位置を計算"""
    if len(waypoints) == 0 or len(seg_times) == 0:
        return base_center
    
    cumsum_times = np.cumsum(seg_times)
    current_segment = np.searchsorted(cumsum_times, t, side='right')
    
    if current_segment >= len(waypoints):
        return waypoints[-1]
    elif current_segment == 0:
        seg_dur = seg_times[0] if len(seg_times) > 0 else 1.0
        prog = t / seg_dur if seg_dur > 0 else 0
        return base_center * (1-prog) + waypoints[0] * prog
    else:
        prev_t = cumsum_times[current_segment-1]
        seg_dur = seg_times[current_segment] if current_segment < len(seg_times) else seg_times[-1]
        prog = (t - prev_t) / seg_dur if seg_dur > 0 else 0
        curr_p = waypoints[current_segment-1]
        next_p = waypoints[current_segment] if current_segment < len(waypoints) else waypoints[-1]
        return curr_p * (1-prog) + next_p * prog


