import numpy as np
from numba import njit, prange

# --- 1. 共通計算ロジック（Numba高速化） ---

@njit(fastmath=True)
def inpolygon_numba(x, y, poly_x, poly_y):
    """
    点 (x, y) が多角形 (poly_x, poly_y) の中にあるか判定
    Return: 1 (True) or 0 (False)
    """
    num_vertices = len(poly_x)
    inside = False
    
    j = num_vertices - 1
    for i in range(num_vertices):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        
        if intersect:
            inside = not inside
        j = i
        
    return 1 if inside else 0

@njit(parallel=True, fastmath=True)
def check_inpolygon_batch(xp, yp, xv, yv):
    n = len(xp)
    result = np.zeros(n, dtype=np.int64)
    for k in prange(n):
        result[k] = inpolygon_numba(xp[k], yp[k], xv, yv)
    return result

@njit(fastmath=True)
def point_line_segment_distance(px, py, x1, y1, x2, y2):
    """2D点 (px, py) と線分 (x1,y1)-(x2,y2) の最短距離"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@njit(fastmath=True)
def point_to_segment_distance_3d(point, seg_start, seg_end):
    """3D点と線分の最短距離（Graph_Distance用）"""
    seg_vec = seg_end - seg_start
    seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2 + seg_vec[2]**2
    if seg_len_sq < 1e-10:
        return np.sqrt((point[0]-seg_start[0])**2 + (point[1]-seg_start[1])**2 + (point[2]-seg_start[2])**2)
    t = ((point[0]-seg_start[0])*seg_vec[0] + (point[1]-seg_start[1])*seg_vec[1] + (point[2]-seg_start[2])*seg_vec[2]) / seg_len_sq
    if t < 0: t = 0.0
    if t > 1: t = 1.0
    closest_x = seg_start[0] + t * seg_vec[0]
    closest_y = seg_start[1] + t * seg_vec[1]
    closest_z = seg_start[2] + t * seg_vec[2]
    return np.sqrt((point[0]-closest_x)**2 + (point[1]-closest_y)**2 + (point[2]-closest_z)**2)

@njit(parallel=True, fastmath=True)
def check_wall_collision_batch(xp, yp, xv, yv, radius):
    """壁との衝突判定（内部判定 + 縁との距離判定）"""
    n = len(xp)
    result = np.zeros(n, dtype=np.int64)
    num_edges = len(xv)
    for k in prange(n):
        # 1. 重心が内部にあるか
        if inpolygon_numba(xp[k], yp[k], xv, yv):
            result[k] = 1
            continue
        # 2. 縁との距離が半径以内か
        for i in range(num_edges):
            j = (i + 1) % num_edges
            dist = point_line_segment_distance(xp[k], yp[k], xv[i], yv[i], xv[j], yv[j])
            if dist < radius:
                result[k] = 1
                break
    return result

# --- 2. 共通位置計算ロジック（Numba高速化版） ---

@njit(fastmath=True)
def _compute_dynamic_position_numba(base_center, waypoints, cumulative_times, seg_times,
                                    t, start_time, end_time):
    """
    Numba高速化：動的障害物の指定時刻における中心位置を計算
    引数は全てNumPy配列またはスカラー（辞書不可）
    """
    # 時刻範囲外の場合
    if t < start_time:
        return base_center.copy()
    
    n_waypoints = waypoints.shape[0] if waypoints.ndim > 0 else 0
    if t >= end_time:
        if n_waypoints > 0:
            return waypoints[-1].copy()
        return base_center.copy()
    
    elapsed_time = t - start_time
    
    if elapsed_time <= 0:
        return base_center.copy()
    if len(cumulative_times) == 0:
        return base_center.copy()

    # np.searchsorted の代替実装
    current_segment = 0
    for i in range(len(cumulative_times)):
        if cumulative_times[i] > elapsed_time:
            current_segment = i
            break
        current_segment = i + 1
    
    if current_segment >= n_waypoints:
        return waypoints[-1].copy()
    
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
    
    segment_elapsed = elapsed_time - segment_start_time
    if segment_duration > 0:
        ratio = segment_elapsed / segment_duration
    else:
        ratio = 0.0
    
    # np.clip の代替実装
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    
    # 線形補間
    center = np.empty(3)
    for i in range(3):
        center[i] = start_pos[i] + ratio * (end_pos[i] - start_pos[i])
    return center


def get_dynamic_obstacle_position(P, obstacle_idx, t):
    """
    動的障害物の指定時刻における中心位置を計算（ラッパー関数）
    Pythonの辞書からデータを抽出し、Numba関数を呼び出す
    """
    if 'dynamic_obj' not in P:
        return np.zeros(3)

    base_circle = P['dynamic_obj'][:, :, obstacle_idx]
    # 2D/3D対応：base_circleの次元数をチェック
    if base_circle.shape[0] == 2:
        # 2Dモード：Z座標は0として扱う
        base_center = np.array([
            base_circle[0, :].mean(),
            base_circle[1, :].mean(),
            0.0
        ], dtype=np.float64)
    else:
        # 3Dモード
        base_center = np.array([
            base_circle[0, :].mean(),
            base_circle[1, :].mean(),
            base_circle[2, :].mean()
        ], dtype=np.float64)
    
    # ウェイポイントの取得
    if isinstance(P.get('dynamic_waypoints'), list):
        if len(P['dynamic_waypoints']) > obstacle_idx:
            waypoints = np.asarray(P['dynamic_waypoints'][obstacle_idx], dtype=np.float64)
        else:
            waypoints = np.zeros((0, 3), dtype=np.float64)
    else:
        wp = P.get('dynamic_waypoints', [])
        waypoints = np.asarray(wp, dtype=np.float64) if len(wp) > 0 else np.zeros((0, 3), dtype=np.float64)
    
    # セグメント時間の取得
    if isinstance(P.get('dynamic_segment_times'), list):
        if len(P['dynamic_segment_times']) > obstacle_idx:
            seg_times = np.asarray(P['dynamic_segment_times'][obstacle_idx], dtype=np.float64)
        else:
            seg_times = np.zeros(0, dtype=np.float64)
    else:
        st = P.get('dynamic_segment_times', [])
        seg_times = np.asarray(st, dtype=np.float64) if len(st) > 0 else np.zeros(0, dtype=np.float64)
    
    start_time = float(P.get('dynamic_start_time', 0.0))
    end_time = float(P.get('dynamic_end_time', P.get('Trial_time', 30.0)))
    
    # 累積時間の計算
    if len(seg_times) > 0:
        cumulative_times = np.cumsum(seg_times)
    else:
        cumulative_times = np.zeros(0, dtype=np.float64)
    
    # ウェイポイントが空の場合
    if waypoints.size == 0:
        return base_center
    
    # 2D配列として整形（n_waypoints x 3）
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(1, -1)
    if waypoints.shape[1] != 3:
        waypoints = np.column_stack([waypoints, np.zeros(waypoints.shape[0])])
    
    # Numba関数を呼び出し
    return _compute_dynamic_position_numba(
        base_center, waypoints, cumulative_times, seg_times,
        float(t), start_time, end_time
    )

# --- 3. 衝突判定ロジック (内部用) ---

def get_dynamic_obstacles_at_t(P, t):
    """(内部計算用) 全障害物の位置を一括計算して返す"""
    if 'dynamic' not in P or not P['dynamic']:
        return None

    dynamic_obj = P['dynamic_obj']
    n_obstacles = dynamic_obj.shape[2] if dynamic_obj.ndim == 3 else 1
    current_dynamic_list = []
    
    for obs_idx in range(n_obstacles):
        # 共通化した関数を利用して中心位置を取得
        current_center = get_dynamic_obstacle_position(P, obs_idx, t)
        
        # 点群を移動させる
        if dynamic_obj.ndim == 3:
            base_circle = dynamic_obj[:, :, obs_idx]
        else:
            base_circle = dynamic_obj
            
        # 2D/3D対応：次元数に応じて中心位置を計算
        dim = base_circle.shape[0]
        if dim == 2:
            original_center = np.array([base_circle[0, :].mean(), base_circle[1, :].mean()])
        else:
            original_center = np.array([base_circle[0, :].mean(), base_circle[1, :].mean(), base_circle[2, :].mean()])
        
        current_center = current_center[:dim].reshape(dim, 1)
        original_center = original_center.reshape(dim, 1)
        
        shifted_circle = base_circle - original_center + current_center
        current_dynamic_list.append(shifted_circle)
        
    if len(current_dynamic_list) > 0:
        return np.stack(current_dynamic_list, axis=2)
    return None

def check(states1, states2, P, t=None): 
    states1 = np.atleast_2d(states1)
    states2 = np.atleast_2d(states2)
    if states1.shape[0] != 12 and states1.shape[1] == 12: states1 = states1.T
    if states2.shape[0] != 12 and states2.shape[1] == 12: states2 = states2.T
    n = states1.shape[1]
    fale = np.zeros(n, dtype=int)

    pos_end = states1[0:3, :]  
    pos_start = states2[0:3, :] 
    pos_mid = (pos_end + pos_start) / 2 

    dt = P.get('dt', 0.2)
    agent_radius = P.get('agent_radius', 0.3)

    # 動的障害物チェック
    if t is not None and 'dynamic' in P and P['dynamic']:
        check_timings = [
            (t, pos_end),
            (t - dt/2, pos_mid),
            (t - dt, pos_start)
        ]
        
        for check_t, check_pos in check_timings:
            obs_at_t = get_dynamic_obstacles_at_t(P, check_t)
            
            if obs_at_t is not None:
                if check_t == t: P['current_dynamic_obj'] = obs_at_t
                
                for i in range(obs_at_t.shape[2]):
                    sphere_center = obs_at_t[:, :, i].mean(axis=1)
                    calc_radius = np.max(np.linalg.norm(obs_at_t[:, :, i] - sphere_center.reshape(-1, 1), axis=0))
                    sphere_radius = max(calc_radius, 0.3)
                    
                    if len(sphere_center) == 3:
                        cx, cy, cz = sphere_center
                        dist = np.sqrt((check_pos[0] - cx)**2 + (check_pos[1] - cy)**2 + (check_pos[2] - cz)**2)
                        collision_limit = sphere_radius + agent_radius
                        collision = (dist < collision_limit).astype(int)
                        fale += collision
                    else:
                        xv = obs_at_t[0, :, i]; yv = obs_at_t[1, :, i]
                        in_poly = check_wall_collision_batch(check_pos[0].flatten(), check_pos[1].flatten(), xv, yv, agent_radius)
                        fale += in_poly.flatten()

    # 静的障害物（壁）チェック
    wall_height = P.get('object_height', P.get('wall_height', 5.0))
    if 'object' in P:
        for i in range(P['object'].shape[2]):
            xv = P['object'][0, :, i]
            yv = P['object'][1, :, i]
            
            for pos in [pos_end, pos_mid, pos_start]:
                xp, yp, zp = pos[0, :], pos[1, :], pos[2, :]
                in_wall = check_wall_collision_batch(xp, yp, xv, yv, agent_radius)
                height_check = (zp >= 0) & (zp <= wall_height)
                fale += (in_wall.astype(int) * height_check.astype(int))

    # 地面・天井判定
    min_height = P.get('min_height', 0.0)
    max_height = P.get('max_height', 5.0)
    fale = fale + (states1[2, :] <= min_height + agent_radius)
    fale = fale + (states1[2, :] >= max_height - agent_radius)
    
    fale = fale >= 1
    return fale