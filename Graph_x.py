import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

def Graph_x(ds_state_list, P, agbp_list, bpc_list, collision_list=None):
    # 障害物描画
    fig, ax = plt.subplots()
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    plt.box(True)
    plt.axis(P['axis'])
    ax.set_aspect('equal')
    if 'object' in P:
        obj = P['object']
        if obj.ndim == 3:
            n_obs = obj.shape[2]  # 全ての静止障害物を描画
            for i in range(n_obs):
                xv = obj[0, :, i]
                yv = obj[1, :, i]
                ax.fill(xv, yv, color='blue', alpha=0.5, edgecolor='k')
        else:
            xv = obj[0, :]
            yv = obj[1, :]
            ax.fill(xv, yv, color='blue', alpha=0.5, edgecolor='k')

    # スタート・ゴール座標
    ax.plot(ds_state_list[0][0,0], ds_state_list[0][0,1], 'o', color=[0.5, 0, 1], markersize=5, markerfacecolor=[0.5, 0, 1], linewidth=2)
    ax.plot(P['Goal_state'][0,0], P['Goal_state'][1,0], marker='D', color=[0,0,1], markersize=5, markerfacecolor=[0,0,1], linewidth=2)

    gif_filename = 'animation.gif'
    delay = 0.03  # フレーム間の遅延
    images = []
    frame_interval = 1  # フレームの間引き(表示を軽くする)

    # 経路表示のフラグ（Falseにすると経路を非表示，障害物の動きを見るため）True or False
    show_trajectory = True
    
    # 全試行の経路を同時に描画するためのラインオブジェクトを作成
    line_objects = []
    if show_trajectory:
        for i in range(P['Trial_num']):
            g = 0 if P['Trial_num'] == 1 else (i)/(P['Trial_num']-1)
            h, = ax.plot([], [], '-', color=[1-g, 0+g, 0], linewidth=1.5)
            line_objects.append(h)
    
    # 最大のステップ数を取得
    # 順番表示の場合は全試行の合計ステップ数が必要
    if show_trajectory:
        # 各試行のステップ数を合計
        max_steps = sum(len(ds_state_list[i][:, 0]) for i in range(P['Trial_num']))
    else:
        # 障害物の動きだけを見る場合は、Trial_time全体を表示
        max_steps = int(P['Trial_time'] / P['dt'])
    
    # 動的障害物用のパッチを先に作成
    if 'dynamic' in P and P['dynamic']:
        dyn_obj = P['dynamic_obj']
        
        # 障害物の数を判定
        if dyn_obj.ndim == 3:
            n_obstacles = dyn_obj.shape[2]
        else:
            n_obstacles = 1
            dyn_obj = np.expand_dims(dyn_obj, axis=2)
        
        # 複数障害物用のデータを準備
        ax._dynamic_patches = []
        ax._base_circles = []
        ax._waypoints_list = []
        ax._seg_times_list = []
        
        for obs_idx in range(n_obstacles):
            base_circle = dyn_obj[:, :, obs_idx]
            
            # 各障害物のwaypoint/segment_timeデータを取得
            if isinstance(P.get('dynamic_waypoints'), list):
                waypoints_for_draw = np.asarray(P['dynamic_waypoints'][obs_idx])
            else:
                waypoints_for_draw = np.asarray(P.get('dynamic_waypoints'))

            if isinstance(P.get('dynamic_segment_times'), list):
                seg_times_for_draw = np.asarray(P['dynamic_segment_times'][obs_idx])
            else:
                seg_times_for_draw = np.asarray(P.get('dynamic_segment_times'))
            
            # 各障害物の色を変える（赤系のグラデーション）
            color_intensity = 0.3 + 0.7 * (obs_idx / max(n_obstacles - 1, 1))
            patch = ax.add_patch(
                Polygon(base_circle.T, facecolor=[color_intensity, 0, 0], alpha=0.5, edgecolor='k')
            )
            
            ax._dynamic_patches.append(patch)
            ax._base_circles.append(base_circle)
            ax._waypoints_list.append(waypoints_for_draw)
            ax._seg_times_list.append(seg_times_for_draw)
    
    # 各ステップごとにアニメーションを更新
    collision_markers_drawn = [False] * P['Trial_num']  # 衝突マーカーを描画したか追跡
    collision_step = [None] * P['Trial_num']  # 各試行の衝突ステップを記録
    goal_reached = [False] * P['Trial_num']  # 目標到達したか追跡
    trial_start_step = [0] * P['Trial_num']  # 各試行の開始ステップを記録（全体のkでの開始位置）
    trial_current_step = [0] * P['Trial_num']  # 各試行の現在のステップ（0から始まる独立カウンター）
    
    # 目標座標
    goal_x = P['Goal_state'][0, 0]
    goal_y = P['Goal_state'][1, 0]
    goal_threshold = P.get('goal_threshold', 0.2)  # 目標到達とみなす距離の閾値[m]
    
    # 時刻表示用のテキストオブジェクトを作成
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 現在表示中の試行を追跡
    current_trial = 0
    
    for k in range(max_steps):
        # 全試行の経路を1つずつ順番に更新（経路表示がオンの場合のみ）
        if show_trajectory:
            for i in range(P['Trial_num']):
                # すでに目標到達または衝突した試行はスキップ
                if goal_reached[i] or collision_markers_drawn[i]:
                    continue
                
                # 前の試行が完了するまで待機（順番表示制御）
                if i > 0:
                    prev_trial_finished = goal_reached[i-1] or collision_markers_drawn[i-1]
                    if not prev_trial_finished:
                        # 前の試行がまだ終わっていないので、この試行は表示しない
                        continue
                    # 前の試行が今完了した場合、この試行の開始ステップを記録
                    if trial_start_step[i] == 0 and current_trial == i - 1:
                        trial_start_step[i] = k
                        current_trial = i
                
                ds_state = ds_state_list[i]
                x = ds_state[:, 0]
                y = ds_state[:, 1]
                z = ds_state[:, 2]
                agbp = agbp_list[i]
                bpc = bpc_list[i]
                
                # この試行の独立したステップカウンターを更新
                step_in_trial = trial_current_step[i]
                
                # 現在のステップが有効範囲外なら試行完了
                if step_in_trial >= len(x) or z[step_in_trial] == 0:
                    goal_reached[i] = True  # 完了フラグを立てる
                    # 次の試行に移行
                    if i + 1 < P['Trial_num']:
                        current_trial = i + 1
                    continue
                
                # 目標到達チェック
                distance_to_goal = np.sqrt((x[step_in_trial] - goal_x)**2 + (y[step_in_trial] - goal_y)**2)
                if distance_to_goal <= goal_threshold:
                    goal_reached[i] = True
                    # 目標到達時点まで経路を描画して終了
                    line_objects[i].set_data(x[:step_in_trial+1], y[:step_in_trial+1])
                    # 次の試行に移行
                    if i + 1 < P['Trial_num']:
                        current_trial = i + 1
                    continue
                
                # 衝突チェック: 次のステップがゼロまたは範囲外なら衝突
                is_collision = False
                if step_in_trial + 1 >= len(x):
                    is_collision = True
                elif z[step_in_trial+1] == 0:
                    is_collision = True
                
                # 衝突が検出された場合、×印を描画して以降の描画を停止
                if is_collision:
                    if collision_list is not None and i < len(collision_list) and collision_list[i] is not None:
                        collision_pos = collision_list[i]
                        g = 0 if P['Trial_num'] == 1 else (i)/(P['Trial_num']-1)
                        # ×印を描画
                        ax.plot(collision_pos[0], collision_pos[1], marker='x', 
                            color=[1-g, 0+g, 0], markersize=8, markeredgewidth=3)
                        # 経路は×印の1ステップ前（衝突位置）までで停止
                        line_objects[i].set_data(x[:step_in_trial], y[:step_in_trial])
                        collision_markers_drawn[i] = True
                        collision_step[i] = step_in_trial
                        # 次の試行に移行
                        if i + 1 < P['Trial_num']:
                            current_trial = i + 1
                # 衝突も目標到達もしていない場合は経路を更新
                else:
                    line_objects[i].set_data(x[:step_in_trial+1], y[:step_in_trial+1])
                    
                    # ロック発生座標のマーカー描画
                    if agbp is not None and bpc is not None and bpc > 0:
                        for idx in range(bpc):
                            # 軌跡がロック発生座標に到達した時だけマーカーを描画
                            if abs(x[step_in_trial] - agbp[0, idx]) < 1e-6 and abs(y[step_in_trial] - agbp[1, idx]) < 1e-6:
                                ax.plot(
                                    agbp[0, idx], agbp[1, idx],
                                    marker='s', color=[0, 0, 1],
                                    markerfacecolor='none',
                                    markersize=6, markeredgewidth=1
                                )
                    
                    # 次のステップに進める
                    trial_current_step[i] += 1
        
        # 動的障害物の更新と描画（frame_interval毎に更新）
        if 'dynamic' in P and P['dynamic'] and k % frame_interval == 0:
            # 現在表示中の試行の経過時間を使用
            if show_trajectory:
                current_time = trial_current_step[current_trial] * P['dt']
            else:
                current_time = k * P['dt']
            
            # 各障害物を更新
            for obs_idx in range(len(ax._dynamic_patches)):
                waypoints_for_draw = ax._waypoints_list[obs_idx]
                seg_times_for_draw = ax._seg_times_list[obs_idx]
                base_circle = ax._base_circles[obs_idx]
                
                # 現在のセグメントとその中での進行度を計算
                cumsum_times = np.cumsum(seg_times_for_draw)
                current_segment = np.searchsorted(cumsum_times, current_time, side='right')
                
                # セグメントが有効範囲を超えたら最後のウェイポイントで停止
                if current_segment >= len(waypoints_for_draw):
                    current_pos = waypoints_for_draw[-1]
                    current_center = current_pos
                elif current_segment == 0:
                    # t=0または最初のセグメント内の場合、元の中心から最初のwaypointへ移動
                    # 元の中心位置を取得
                    original_center = base_circle.mean(axis=1)
                    
                    # 最初のセグメントの進行度を計算
                    seg_duration = seg_times_for_draw[0] if len(seg_times_for_draw) > 0 else 1.0
                    if seg_duration > 0:
                        progress = current_time / seg_duration
                    else:
                        progress = 0
                    
                    # 元の位置から最初のwaypointへ補間
                    next_pos = waypoints_for_draw[0]
                    current_center = original_center * (1-progress) + next_pos * progress
                else:
                    # waypoint間の補間処理
                    prev_time = cumsum_times[current_segment-1] if current_segment > 0 else 0
                    seg_duration = seg_times_for_draw[current_segment] if current_segment < len(seg_times_for_draw) else seg_times_for_draw[-1]
                    
                    if seg_duration > 0:
                        progress = (current_time - prev_time) / seg_duration
                    else:
                        progress = 0
                    
                    # 現在と次のウェイポイント
                    current_pos = waypoints_for_draw[current_segment-1] if current_segment > 0 else waypoints_for_draw[0]
                    next_pos = waypoints_for_draw[current_segment] if current_segment < len(waypoints_for_draw) else waypoints_for_draw[-1]
                    
                    # 現在位置を補間
                    current_center = current_pos * (1-progress) + next_pos * progress
                
                # 円形障害物の頂点を更新
                current_center = np.asarray(current_center).reshape(2, 1)
                old_center = base_circle.mean(axis=1).reshape(2, 1)
                shifted_circle = base_circle - old_center + current_center
                
                # パッチの頂点を更新
                ax._dynamic_patches[obs_idx].set_xy(shifted_circle.T)
        
        # 時刻表示を更新（現在表示中の試行の経過時間）
        if show_trajectory:
            elapsed_time = trial_current_step[current_trial] * P['dt']
            time_text.set_text(f'Trial {current_trial + 1} - Time: {elapsed_time:.1f}s')
        else:
            current_time = k * P['dt']
            time_text.set_text(f'Time: {current_time:.1f}s')
        
        if k % frame_interval == 0:  # frame_interval毎にフレームを保存
            plt.draw()
            # GIF用画像保存
            fig.canvas.draw()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())
            image = image[:, :, :3]
            images.append(image.copy())
            plt.pause(0.001)  # 表示更新用の最小遅延
    
    # 全試行の終点にマーカーを描画
    for i in range(P['Trial_num']):
        g = 0 if P['Trial_num'] == 1 else (i)/(P['Trial_num']-1)
        ds_state = ds_state_list[i]
        x = ds_state[:, 0]
        y = ds_state[:, 1]
        
        # 衝突があった場合は×印を描画、なければ終点に星印
        if collision_list is not None and i < len(collision_list) and collision_list[i] is not None:
            collision_pos = collision_list[i]
            ax.plot(collision_pos[0], collision_pos[1], marker='x', 
                    color=[1-g, 0+g, 0], markersize=12, markeredgewidth=3,
                    label=f'Trial {i+1} 衝突' if i == 0 else '')
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

