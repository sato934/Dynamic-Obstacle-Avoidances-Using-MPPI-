import numpy as np
from matplotlib.path import Path

def inpolygon(x, y, xv, yv):
    # MATLAB の inpolygon 相当 (点が多角形内か判定)
    points = np.vstack((x, y)).T
    poly = np.vstack((xv, yv)).T
    return Path(poly).contains_points(points)

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
    xm = (states1[0, :] + states2[0, :]) / 2
    ym = (states1[1, :] + states2[1, :]) / 2

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
                
                if len(current_pos) < len(ref_center):
                    current_pos = np.append(current_pos, ref_center[len(current_pos):])
                elif len(current_pos) > len(ref_center):
                    current_pos = current_pos[:len(ref_center)]
                    
                if len(next_pos) < len(ref_center):
                    next_pos = np.append(next_pos, ref_center[len(next_pos):])
                elif len(next_pos) > len(ref_center):
                    next_pos = next_pos[:len(ref_center)]
                
                # 現在位置を補間
                current_center = current_pos * (1-progress) + next_pos * progress
            
            # 円形障害物の頂点を更新
            if dynamic_obj.ndim == 3:
                base_circle = dynamic_obj[:, :, obs_idx]
            else:
                base_circle = dynamic_obj
            
            current_center = np.asarray(current_center).reshape(2, 1)
            old_center = base_circle.mean(axis=1).reshape(2, 1)
            shifted_circle = base_circle - old_center + current_center
            current_dynamic_list.append(shifted_circle)
        
        # 動的障害物の頂点座標として設定
        if len(current_dynamic_list) > 0:
            P['current_dynamic_obj'] = np.stack(current_dynamic_list, axis=2)

    # 静的障害物のチェック
    for i in range(P['object'].shape[2]):
        xv = P['object'][0, :, i]
        yv = P['object'][1, :, i]
        in1 = inpolygon(xp, yp, xv, yv)
        in2 = inpolygon(xm, ym, xv, yv)
        fale += in1.astype(int) + in2.astype(int)

    # 動的障害物のチェック 
    if 'dynamic' in P and P['dynamic']:
        dynamic_obj = P.get('current_dynamic_obj', P['dynamic_obj'])
        for i in range(dynamic_obj.shape[2]):
            xv = dynamic_obj[0, :, i]
            yv = dynamic_obj[1, :, i]
            in1 = inpolygon(xp, yp, xv, yv)
            in2 = inpolygon(xm, ym, xv, yv)
            fale += in1.astype(int) + in2.astype(int)

    fale = fale + (states1[2, :] <= 0)
    fale = fale >= 1
    return fale
