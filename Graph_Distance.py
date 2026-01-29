import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False


def get_dynamic_obstacle_position(P, obstacle_idx, t):
    """
    動的障害物の指定時刻における中心位置を計算
    """
    # 初期中心位置
    base_circle = P['dynamic_obj'][:, :, obstacle_idx]  # (3, n_points)
    base_center = np.array([
        base_circle[0, :].mean(),
        base_circle[1, :].mean(),
        base_circle[2, :].mean()
    ])
    
    # waypoint情報を取得
    if isinstance(P.get('dynamic_waypoints'), list):
        waypoints = np.asarray(P['dynamic_waypoints'][obstacle_idx])
    else:
        waypoints = np.asarray(P.get('dynamic_waypoints'))
    
    if isinstance(P.get('dynamic_segment_times'), list):
        seg_times = np.asarray(P['dynamic_segment_times'][obstacle_idx])
    else:
        seg_times = np.asarray(P.get('dynamic_segment_times'))
    
    start_time = P.get('dynamic_start_time', 0.0)
    end_time = P.get('dynamic_end_time', P['Trial_time'])
    
    # 時刻範囲外の場合
    if t < start_time:
        return base_center
    if t >= end_time:
        # 最終位置
        if len(waypoints) > 0:
            return waypoints[-1]
        return base_center
    
    # 累積時間を計算
    elapsed_time = t - start_time
    cumulative_times = np.cumsum(seg_times)
    
    # 現在のセグメントを特定
    if elapsed_time <= 0:
        return base_center
    
    current_segment = np.searchsorted(cumulative_times, elapsed_time)
    
    if current_segment >= len(waypoints):
        # 最終waypoint到達後
        return waypoints[-1]
    
    # セグメント内での補間
    if current_segment == 0:
        start_pos = base_center
        end_pos = waypoints[0]
        segment_start_time = 0.0
        segment_duration = seg_times[0]
    else:
        start_pos = waypoints[current_segment - 1]
        end_pos = waypoints[current_segment]
        segment_start_time = cumulative_times[current_segment - 1]
        segment_duration = seg_times[current_segment]
    
    # セグメント内の進行率
    segment_elapsed = elapsed_time - segment_start_time
    ratio = segment_elapsed / segment_duration if segment_duration > 0 else 0.0
    ratio = np.clip(ratio, 0.0, 1.0)
    
    # 線形補間
    center = start_pos + ratio * (end_pos - start_pos)
    
    return center


def compute_distance_to_obstacles(state, P, t):
    """
    指定時刻における機体と全障害物との最小表面間距離を計算
    """
    pos = state[0:3]  # 機体位置 (x, y, z)
    agent_radius = P.get('agent_radius', 0.3)
    
    min_distance = np.inf
    obstacle_type = 'none'
    
    # 1. 静的障害物（壁）との距離
    if 'object' in P and P['object'].size > 0:
        num_obstacles = P['object'].shape[2]
        wall_height = P.get('object_height', P.get('wall_height', 3.0))
        
        for obj_idx in range(num_obstacles):
            obj_points = P['object'][:, :, obj_idx]  # (3, n_vertices)
            
            # 壁の各辺との距離を計算（3D）
            n_vertices = obj_points.shape[1]
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                
                # 底面の辺
                p1_bottom = obj_points[:, i]
                p2_bottom = obj_points[:, j]
                
                # 上面の辺
                p1_top = p1_bottom.copy()
                p1_top[2] = wall_height
                p2_top = p2_bottom.copy()
                p2_top[2] = wall_height
                
                # 4つの辺との距離を計算
                for p1, p2 in [(p1_bottom, p2_bottom), (p1_top, p2_top),
                               (p1_bottom, p1_top), (p2_bottom, p2_top)]:
                    dist = point_to_segment_distance(pos, p1, p2)
                    if dist < min_distance:
                        min_distance = dist
                        obstacle_type = 'static_wall'
    
    # 2. 動的障害物（球体）との距離
    if P.get('dynamic', False) and 'dynamic_obj' in P:
        num_spheres = P['dynamic_obj'].shape[2]
        sphere_radius = 0.3  # 球体の半径
        
        for sphere_idx in range(num_spheres):
            center = get_dynamic_obstacle_position(P, sphere_idx, t)
            
            # 中心間距離
            dist_center = np.linalg.norm(pos - center)
            
            # 表面間距離 = 中心間距離 - 機体半径 - 球体半径
            surface_dist = dist_center - agent_radius - sphere_radius
            
            if surface_dist < min_distance:
                min_distance = surface_dist
                obstacle_type = 'dynamic_sphere'
    
    # 3. 床・天井との距離
    min_height = P.get('min_height', 0.0)
    max_height = P.get('max_height', 3.0)
    
    # 床との表面間距離
    floor_dist = pos[2] - min_height - agent_radius
    if floor_dist < min_distance:
        min_distance = floor_dist
        obstacle_type = 'floor'
    
    # 天井との表面間距離
    ceiling_dist = max_height - pos[2] - agent_radius
    if ceiling_dist < min_distance:
        min_distance = ceiling_dist
        obstacle_type = 'ceiling'
    
    return min_distance, obstacle_type


def point_to_segment_distance(point, seg_start, seg_end):
    """
    点から線分への最短距離を計算（3D）
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq < 1e-10:
        # 線分が点の場合
        return np.linalg.norm(point - seg_start)
    
    # 線分上の最近点のパラメータ t を計算
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)  # 線分上に制限
    
    # 最近点
    closest_point = seg_start + t * seg_vec
    
    return np.linalg.norm(point - closest_point)


def plot_distance_time_graph(ds_state_list, P, save_path=None):
    """
    機体と障害物の表面間距離の時系列グラフを作成
    """
    # 最良試行（最後の試行）のデータを使用
    best_state = ds_state_list[-1]  # shape: (12, num_steps)
    num_steps = best_state.shape[1]
    dt = P['dt']
    
    # 目標到達時刻を検出
    goal_pos = P['Goal_state'][0:3, 0]
    goal_threshold = P.get('goal_threshold', 0.2)
    goal_reached_step = num_steps  # デフォルトは全ステップ
    
    for step in range(num_steps):
        pos = best_state[0:3, step]
        dist_to_goal = np.linalg.norm(pos - goal_pos)
        if dist_to_goal <= goal_threshold:
            goal_reached_step = step
            print(f"目標到達: {step * dt:.2f}秒後（ステップ {step}）")
            break
    
    # 目標到達までのステップ数に制限
    analysis_steps = goal_reached_step + 1  # 到達時刻を含む
    
    # 時刻配列
    time_array = np.arange(analysis_steps) * dt
    
    # 距離配列を初期化
    distance_array = np.zeros(analysis_steps)
    obstacle_types = []
    
    print("距離計算中...")
    for step in range(analysis_steps):
        state = best_state[:, step]
        t = time_array[step]
        
        min_dist, obs_type = compute_distance_to_obstacles(state, P, t)
        distance_array[step] = min_dist
        obstacle_types.append(obs_type)
    
    # 最小距離を記録
    min_distance = np.min(distance_array)
    min_time = time_array[np.argmin(distance_array)]
    min_obstacle = obstacle_types[np.argmin(distance_array)]
    goal_arrival_time = time_array[-1]
    
    print(f"\n=== 安全性分析結果 ===")
    print(f"最小表面間距離: {min_distance:.3f} m")
    print(f"最接近時刻: {min_time:.2f} s")
    print(f"最接近障害物: {min_obstacle}")
    print(f"機体半径: {P.get('agent_radius', 0.3):.2f} m")
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 距離の推移曲線
    ax.plot(time_array, distance_array, 'green', linewidth=2, label='表面間距離')
    
    # 安全限界線（衝突ライン）
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='衝突限界 (d = 0 m)')
    
    # 目標到達時刻の縦線
    ax.axvline(x=goal_arrival_time, color='blue', linestyle='-.', linewidth=2, 
               label=f'目標到達 (t = {goal_arrival_time:.2f}s)', alpha=0.7)
    
    # 最小距離の点を強調
    ax.plot(min_time, min_distance, 'ro', markersize=8, 
            label=f'最接近点 ({min_time:.2f}s, {min_distance:.3f}m)')
    
    # グラフ装飾
    ax.set_xlabel('時間 $t$ [s]', fontsize=14)
    ax.set_ylabel('表面間距離 $d_{surface}$ [m]', fontsize=14)
    ax.set_title('機体と障害物の距離推移', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # y軸の範囲を調整（負の値も表示できるように）
    y_min = min(-0.1, min_distance - 0.2)
    y_max = max(distance_array) * 1.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()
    
    return min_distance, min_time, min_obstacle