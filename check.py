import numpy as np
from numba import njit, prange

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
        # 多角形の辺 (i, j) と、点から伸びる水平線が交差するか判定
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        
        if intersect:
            inside = not inside
        j = i
        
    return 1 if inside else 0

# まとめて判定するラッパー関数
@njit(parallel=True, fastmath=True)
def check_inpolygon_batch(xp, yp, xv, yv):
    n = len(xp)
    result = np.zeros(n, dtype=np.int64) # int64で返す
    
    # 複数の点を並列チェック
    for k in prange(n):
        result[k] = inpolygon_numba(xp[k], yp[k], xv, yv)
        
    return result

def check(states1, states2, P, t=None): # t: 現在時刻（動的障害物用、Noneなら無視） 
    states1 = np.atleast_2d(states1)
    states2 = np.atleast_2d(states2)
    if states1.shape[0] != 12 and states1.shape[1] == 12:
        states1 = states1.T
    if states2.shape[0] != 12 and states2.shape[1] == 12:
        states2 = states2.T
    n = states1.shape[1]
    fale = np.zeros(n, dtype=int)

    xp = states1[0, :]
    yp = states1[1, :]
    zp = states1[2, :]  
    xm = (states1[0, :] + states2[0, :]) / 2
    ym = (states1[1, :] + states2[1, :]) / 2
    zm = (states1[2, :] + states2[2, :]) / 2  

    # 動的障害物の現在位置を計算（時刻tが指定されている場合） 
    if t is not None and 'dynamic' in P and P['dynamic']:
        # 複数の動的障害物に対応
        dynamic_obj = P['dynamic_obj']
        n_obstacles = dynamic_obj.shape[2] if dynamic_obj.ndim == 3 else 1
        
        current_dynamic_list = []
        
        for obs_idx in range(n_obstacles):
            # 各障害物のwaypoint/segment_timeデータを取得
            if isinstance(P.get('dynamic_waypoints'), list) and len(P['dynamic_waypoints']) > obs_idx:
                waypoints = np.asarray(P['dynamic_waypoints'][obs_idx])
            else:
                waypoints = np.asarray(P.get('dynamic_waypoints', []))
            
            if isinstance(P.get('dynamic_segment_times'), list) and len(P['dynamic_segment_times']) > obs_idx:
                seg_times = np.asarray(P['dynamic_segment_times'][obs_idx])
            else:
                seg_times = np.asarray(P.get('dynamic_segment_times', []))
            
            if len(waypoints) == 0 or len(seg_times) == 0:
                # データがない場合は元の位置を使用
                if dynamic_obj.ndim == 3:
                    current_dynamic_list.append(dynamic_obj[:, :, obs_idx])
                else:
                    current_dynamic_list.append(dynamic_obj)
                continue
            
            # 現在のセグメントとその中での進行度を計算
            cumsum_times = np.cumsum(seg_times)
            current_segment = np.searchsorted(cumsum_times, t, side='right')
            
            if current_segment >= len(waypoints):
                # 最後のウェイポイントで停止
                current_center = waypoints[-1]
            elif current_segment == 0:
                # t=0または最初のセグメント内の場合、元の中心から最初のwaypointへ移動
                # 元の中心位置を取得
                if dynamic_obj.ndim == 3:
                    base_circle = dynamic_obj[:, :, obs_idx]
                else:
                    base_circle = dynamic_obj
                original_center = base_circle.mean(axis=1)
                
                # 最初のセグメントの進行度を計算
                seg_duration = seg_times[0] if len(seg_times) > 0 else 1.0
                if seg_duration > 0:
                    progress = t / seg_duration
                else:
                    progress = 0
                
                # 元の位置から最初のwaypointへ補間
                next_pos = waypoints[0]
                # next_posとoriginal_centerの次元を合わせる
                if len(next_pos) < len(original_center):
                    # 2D waypoint → 3D centerの場合、z座標を追加
                    next_pos = np.append(next_pos, original_center[len(next_pos):])
                elif len(next_pos) > len(original_center):
                    # 3D waypoint → 2D centerの場合、次元を削減
                    next_pos = next_pos[:len(original_center)]
                current_center = original_center * (1-progress) + next_pos * progress
            else:
                # 補間処理
                prev_time = cumsum_times[current_segment-1] if current_segment > 0 else 0
                seg_duration = seg_times[current_segment] if current_segment < len(seg_times) else seg_times[-1]
                
                if seg_duration > 0:
                    progress = (t - prev_time) / seg_duration
                else:
                    progress = 0
                
                # 現在と次のウェイポイント
                current_pos = waypoints[current_segment-1] if current_segment > 0 else waypoints[0]
                next_pos = waypoints[current_segment] if current_segment < len(waypoints) else waypoints[-1]
                
                # 次元を合わせる
                if dynamic_obj.ndim == 3:
                    base_circle = dynamic_obj[:, :, obs_idx]
                else:
                    base_circle = dynamic_obj
                ref_center = base_circle.mean(axis=1)
                
                # 3D対応：次元数を参照から取得
                dim = len(ref_center)
                
                if len(current_pos) < dim:
                    current_pos = np.append(current_pos, ref_center[len(current_pos):])
                elif len(current_pos) > dim:
                    current_pos = current_pos[:dim]
                    
                if len(next_pos) < dim:
                    next_pos = np.append(next_pos, ref_center[len(next_pos):])
                elif len(next_pos) > dim:
                    next_pos = next_pos[:dim]
                
                # 現在位置を補間
                current_center = current_pos * (1-progress) + next_pos * progress
            
            # 球体障害物の頂点を更新（3D対応）
            if dynamic_obj.ndim == 3:
                base_circle = dynamic_obj[:, :, obs_idx]
            else:
                base_circle = dynamic_obj
            
            dim = base_circle.shape[0]  # 2D or 3D
            current_center = np.asarray(current_center).reshape(dim, 1)
            old_center = base_circle.mean(axis=1).reshape(dim, 1)
            shifted_circle = base_circle - old_center + current_center
            current_dynamic_list.append(shifted_circle)
        
        # 動的障害物の頂点座標として設定
        if len(current_dynamic_list) > 0:
            P['current_dynamic_obj'] = np.stack(current_dynamic_list, axis=2)

    # 静的障害物のチェック（3D対応：壁の高さを考慮）
    # Load_Settings.pyで設定可能なパラメータ:
    #   P['wall_height']: 壁の高さ[m]（デフォルト: 5.0）
    #   P['object_height']: 個別障害物の高さ[m]（優先）
    wall_height = P.get('object_height', P.get('wall_height', 5.0))
    
    for i in range(P['object'].shape[2]):
        xv = P['object'][0, :, i]
        yv = P['object'][1, :, i]
        # 壁の底面での衝突判定（xy平面）
        in1 = check_inpolygon_batch(xp, yp, xv, yv)
        in2 = check_inpolygon_batch(xm, ym, xv, yv)
        
        # 高さ判定：壁の高さ範囲内（0 <= z <= wall_height）にいるかチェック
        height_check1 = (zp >= 0) & (zp <= wall_height)
        height_check2 = (zm >= 0) & (zm <= wall_height)
        
        # xy平面で衝突 AND 高さ範囲内なら衝突
        fale += (in1.astype(int) * height_check1.astype(int)) + (in2.astype(int) * height_check2.astype(int))

    # 動的障害物のチェック（3D対応：球体との距離判定）
    # Load_Settings.pyで設定可能なパラメータ:
    #   P['agent_radius']: 機体の半径[m]（デフォルト: 0.3）
    if 'dynamic' in P and P['dynamic']:
        dynamic_obj = P.get('current_dynamic_obj', P['dynamic_obj'])
        agent_radius = P.get('agent_radius', 0.3)
        
        for i in range(dynamic_obj.shape[2]):
            # 球体の中心座標を計算
            sphere_center = dynamic_obj[:, :, i].mean(axis=1)  # (x, y, z) or (x, y)
            
            if len(sphere_center) == 3:
                # 3D球体との距離判定
                cx, cy, cz = sphere_center
                
                # エージェント位置との距離
                dist1 = np.sqrt((xp - cx)**2 + (yp - cy)**2 + (zp - cz)**2)
                dist2 = np.sqrt((xm - cx)**2 + (ym - cy)**2 + (zm - cz)**2)
                
                # 球体の半径を推定（点群から）
                sphere_radius = np.max(np.linalg.norm(dynamic_obj[:, :, i] - sphere_center.reshape(-1, 1), axis=0))
                
                # 衝突判定：距離 < 球体半径 + 機体半径
                collision1 = (dist1 < sphere_radius + agent_radius).astype(int)
                collision2 = (dist2 < sphere_radius + agent_radius).astype(int)
                
                fale += collision1 + collision2
            else:
                # 2D円形（後方互換）
                xv = dynamic_obj[0, :, i]
                yv = dynamic_obj[1, :, i]
                in1 = check_inpolygon_batch(xp, yp, xv, yv)
                in2 = check_inpolygon_batch(xm, ym, xv, yv)
                fale += in1.astype(int) + in2.astype(int)

    # 地面・天井判定（3D）
    min_height = P.get('min_height', 0.0)
    max_height = P.get('max_height', 5.0)
    
    fale = fale + (states1[2, :] <= min_height)  # 地面衝突
    fale = fale + (states1[2, :] >= max_height)  # 天井衝突
    fale = fale >= 1
    return fale
