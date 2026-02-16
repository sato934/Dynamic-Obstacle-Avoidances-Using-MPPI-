import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from check import inpolygon_numba, get_dynamic_obstacle_position, point_to_segment_distance_3d

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

def point_to_segment_distance(point, seg_start, seg_end):
    """3D点と線分の最短距離（check.pyのラッパー）"""
    return point_to_segment_distance_3d(
        np.asarray(point, dtype=np.float64),
        np.asarray(seg_start, dtype=np.float64),
        np.asarray(seg_end, dtype=np.float64)
    )

def compute_distance_to_obstacles(state, P, t):
    """指定時刻における機体と全障害物との最小表面間距離を計算"""
    pos = state[0:3]; agent_radius = P.get('agent_radius', 0.3)
    min_distance = np.inf; obstacle_type = 'none'
    
    # 1. 静的障害物（壁）
    if 'object' in P and P['object'].size > 0:
        num_obstacles = P['object'].shape[2]; wall_height = P.get('object_height', P.get('wall_height', 3.0))
        for obj_idx in range(num_obstacles):
            obj_points = P['object'][:, :, obj_idx]
            poly_x = obj_points[0, :]; poly_y = obj_points[1, :]
            
            min_edge_dist = np.inf
            n_vertices = obj_points.shape[1]
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                p1 = obj_points[:, i].copy(); p1[2] = 0
                p2 = obj_points[:, j].copy(); p2[2] = 0
                curr_pos_2d = pos.copy(); curr_pos_2d[2] = 0
                dist = point_to_segment_distance(curr_pos_2d, p1, p2)
                if dist < min_edge_dist: min_edge_dist = dist
            
            if inpolygon_numba(pos[0], pos[1], poly_x, poly_y):
                dist_horizontal = -min_edge_dist
            else:
                dist_horizontal = min_edge_dist
                
            if 0 <= pos[2] <= wall_height:
                dist_vertical = 0
            else:
                dist_vertical = min(abs(pos[2] - 0), abs(pos[2] - wall_height))
            
            if dist_vertical > 0 and dist_horizontal > 0:
                final_dist_center = np.sqrt(dist_horizontal**2 + dist_vertical**2)
            elif dist_vertical > 0:
                final_dist_center = dist_vertical
            else:
                final_dist_center = dist_horizontal

            surface_dist = final_dist_center - agent_radius
            if surface_dist < min_distance:
                min_distance = surface_dist
                obstacle_type = 'static_wall'
    
    # 2. 動的障害物
    if P.get('dynamic', False) and 'dynamic_obj' in P:
        num_spheres = P['dynamic_obj'].shape[2]; sphere_radius = 0.3
        for sphere_idx in range(num_spheres):
            center = get_dynamic_obstacle_position(P, sphere_idx, t)
            dist_center = np.linalg.norm(pos - center)
            surface_dist = dist_center - agent_radius - sphere_radius
            if surface_dist < min_distance: min_distance = surface_dist; obstacle_type = 'dynamic_sphere'
    
    # 3. 床・天井
    min_height = P.get('min_height', 0.0); max_height = P.get('max_height', 3.0)
    floor_dist = pos[2] - min_height - agent_radius
    if floor_dist < min_distance: min_distance = floor_dist; obstacle_type = 'floor'
    ceiling_dist = max_height - pos[2] - agent_radius
    if ceiling_dist < min_distance: min_distance = ceiling_dist; obstacle_type = 'ceiling'
    
    return min_distance, obstacle_type

def calc_distance_series(state, P):
    num_steps = state.shape[1]; dt = P['dt']
    
    goal_pos = P['Goal_state'][0:3, 0]; goal_threshold = P.get('goal_threshold', 0.2)
    goal_reached_step = None
    
    # 目標到達ステップを特定
    for step in range(num_steps):
        pos = state[0:3, step]
        if np.linalg.norm(pos - goal_pos) <= goal_threshold:
            goal_reached_step = step; break
    
    plot_times = []; plot_dists = []; plot_types = []
    collision_time = None
    prev_t = None; prev_d = None
    
    # 目標到達時はそこで終了
    end_step = goal_reached_step + 1 if goal_reached_step is not None else num_steps

    for step in range(end_step):
        curr_state = state[:, step]; curr_t = step * dt
        
        # 1. 中間点の計算 (step > 0)
        if step > 0:
            prev_state_loop = state[:, step-1]; prev_t_loop = (step - 1) * dt
            mid_t = (prev_t_loop + curr_t) / 2
            mid_state = (prev_state_loop + curr_state) / 2
            d_mid, type_mid = compute_distance_to_obstacles(mid_state, P, mid_t)
            
            if d_mid <= 0:
                if prev_d is not None and prev_d > 0:
                    ratio = prev_d / (prev_d - d_mid)
                    collision_time = prev_t + ratio * (mid_t - prev_t)
                    plot_times.append(collision_time)
                    plot_dists.append(0.0)
                    plot_types.append(type_mid)
                else:
                    collision_time = mid_t
                    plot_times.append(mid_t)
                    plot_dists.append(0.0)
                    plot_types.append(type_mid)
                return np.array(plot_times), np.array(plot_dists), plot_types, goal_reached_step, collision_time
            
            plot_times.append(mid_t); plot_dists.append(d_mid); plot_types.append(type_mid)
            prev_t = mid_t; prev_d = d_mid

        # 2. 現在点の計算
        d_curr, type_curr = compute_distance_to_obstacles(curr_state, P, curr_t)
        
        if d_curr <= 0:
            if prev_d is not None and prev_d > 0:
                ratio = prev_d / (prev_d - d_curr)
                collision_time = prev_t + ratio * (curr_t - prev_t)
                plot_times.append(collision_time)
                plot_dists.append(0.0)
                plot_types.append(type_curr)
            else:
                collision_time = curr_t
                plot_times.append(curr_t)
                plot_dists.append(0.0)
                plot_types.append(type_curr)
            return np.array(plot_times), np.array(plot_dists), plot_types, goal_reached_step, collision_time
            
        plot_times.append(curr_t); plot_dists.append(d_curr); plot_types.append(type_curr)
        prev_t = curr_t; prev_d = d_curr
    
    return np.array(plot_times), np.array(plot_dists), plot_types, goal_reached_step, collision_time

def plot_distance_time_graph(ds_state_list, P, save_path=None):
    """1つの試行（リストの末尾）のグラフを描画"""
    state = ds_state_list[-1]
    time_array, distance_array, obstacle_types, goal_reached_step, collision_time = calc_distance_series(state, P)
    
    min_idx = np.argmin(distance_array)
    min_distance = distance_array[min_idx]
    min_time = time_array[min_idx]
    min_obstacle = obstacle_types[min_idx] if min_idx < len(obstacle_types) else 'unknown'
    
    print(f"最小距離: {min_distance:.3f}m (@{min_time:.2f}s) Type: {min_obstacle}")
    if collision_time is not None: print(f"衝突発生: {collision_time:.3f}s")
    if goal_reached_step is not None: print(f"目標到達: {goal_reached_step * P['dt']:.2f}s")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(time_array, distance_array, 'g-', linewidth=1.5, label='表面間距離')
    ax.axhline(y=0, color='blue', linestyle='--', label='衝突限界')
    
    # 衝突時：×印のみ表示
    if collision_time is not None:
        ax.plot(collision_time, 0, 'rx', markersize=15, markeredgewidth=3, label=f'衝突点 ({collision_time:.2f}s)', zorder=10)
    # 目標到達時：最小接近点に赤丸
    elif goal_reached_step is not None:
        ax.plot(min_time, min_distance, 'ro', markersize=8, label=f'最小接近 ({min_distance:.3f}m)', zorder=10)
    
    ax.set_xlabel('時間 $t$ [s]', fontsize=20, labelpad=10)
    ax.set_ylabel('距離 $d$ [m]', fontsize=20, labelpad=10)
    ax.set_title('機体と障害物の距離', fontsize=20, pad=12)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=14)  # 凡例表示したい場合はコメント解除
    y_min = -0.05
    ax.set_ylim(y_min, max(distance_array)*1.1 if max(distance_array) > 0 else 1.0)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); plt.close()
    else: plt.show()
    return min_distance, min_time, min_obstacle

def plot_all_trials_distance_graph(ds_state_list, obs_list, save_path=None):
    """全ての試行の距離グラフを重ねて描画"""
    print("合計グラフを作成中...")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='衝突限界', zorder=10)
    max_time = 0
    for i, (state, P) in enumerate(zip(ds_state_list, obs_list)):
        time_array, distance_array, _, goal_reached_step, collision_time = calc_distance_series(state, P)
        if len(time_array) > 0 and time_array[-1] > max_time: max_time = time_array[-1]
        
        g = i / (len(ds_state_list) - 1) if len(ds_state_list) > 1 else 0
        color = [1-g, 0+g, 0]
        ax.plot(time_array, distance_array, '-', color=color, linewidth=1.5, alpha=0.7, label=f'試行 {i+1}')
        
        # 最小接近点を表示
        min_idx = np.argmin(distance_array)
        if collision_time is not None:
            # 衝突時：終端に×印
            ax.plot(collision_time, 0, 'x', color=color, markersize=10, markeredgewidth=2.5, zorder=15)
        else:
            # 最小接近点に赤丸
            ax.plot(time_array[min_idx], distance_array[min_idx], 'o', color='red', markersize=6, zorder=15)

    ax.set_xlabel('時間 $t$ [s]', fontsize=20, labelpad=10)
    ax.set_ylabel('距離 $d$ [m]', fontsize=20, labelpad=10)
    #ax.set_title('機体と障害物の距離（全試行）', fontsize=20, pad=12)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    #if len(ds_state_list) <= 10: ax.legend(loc='upper right', fontsize=12)  # 凡例表示したい場合はコメント解除
    ax.set_ylim(bottom=-0.05)
    ax.set_xlim(0, max_time)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300); print(f"\n合計グラフを保存しました: {save_path}")
    plt.close()

def save_all_distance_graphs(ds_state_list, obs_list, P_fallback=None):
    print("\n=== 距離時系列グラフの保存処理を開始 ===")
    save_dir = f"Result_Graphs"
    if not os.path.exists(save_dir): os.makedirs(save_dir); print(f"フォルダ作成: {save_dir}")
    formatted_state_list = []
    for ds in ds_state_list:
        if isinstance(ds, np.ndarray):
            if ds.shape[0] > ds.shape[1] and ds.shape[1] == 12: formatted_state_list.append(ds.T)
            else: formatted_state_list.append(ds)
        else: formatted_state_list.append(ds)
    if not obs_list and P_fallback: target_obs_list = [P_fallback] * len(formatted_state_list)
    else: target_obs_list = obs_list
    for i in range(len(formatted_state_list)):
        current_P = target_obs_list[i]
        current_state_wrapper = [formatted_state_list[i]] 
        file_name = f'single_distance_{i+1}.png'
        save_path = os.path.join(save_dir, file_name)
        print(f"  - 保存中: {file_name}")
        plot_distance_time_graph(current_state_wrapper, current_P, save_path=save_path)
    total_file_name = 'single_distance_total.png'
    total_save_path = os.path.join(save_dir, total_file_name)
    print(f"  - 保存中: {total_file_name}")
    plot_all_trials_distance_graph(formatted_state_list, target_obs_list, save_path=total_save_path)
    print("=== 保存処理完了 ===\n")