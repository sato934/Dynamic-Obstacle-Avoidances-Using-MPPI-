import numpy as np
from Cost_Fcn import Cost_Fcn
from Sim_Model import Sim_Model
from Term_Cost import Term_Cost 
from check import check
from shapely.geometry import Polygon, LineString
import time

def MPPI_GT(P, agbp, bpc, parameter):

    banned_point = np.full((3, 100), np.nan)
    ban_count = 0
    bpc_interval = 0
    collision_position = None  # 衝突位置を記録

    trial_state = np.zeros((P['State_dim'], P['Trial_size']))
    seq_ctrl = np.zeros((P['Ctrl_dim'], 1, P['Trial_size'] + P['Horizon_size']))
    seq_ctrl[0, 0, :] = P['initial_controll'][0, 0]
    seq_ctrl[1, 0, :] = P['initial_controll'][1, 0]
    seq_ctrl[2, 0, :] = P['initial_controll'][2, 0]
    seq_ctrl[3, 0, :] = P['initial_controll'][3, 0]
    trial_state[:, 0] = P['Init_State'].flatten()
    cost_diff = np.zeros(P['Trial_size'])
    if parameter is None:
        parameter = np.zeros((2, P['Trial_size'] ))
    block_check_state = np.zeros((P['State_dim'], int(P['check'] / P['dt'])))

    for i in range(P['Trial_size'] - 1):
        loop_start = time.time()
        # 現在の時刻を計算
        current_time = i * P['dt']
        # コスト正規化
        cost_diff[i] = Cost_Fcn(trial_state[:, i], trial_state[:, i], P, banned_point, current_time) / Cost_Fcn(P['Init_State'], P['Init_State'], P, banned_point, 0)
        if cost_diff[i] > P['vlu']:
            cost_diff[i] = P['vlu']
        elif cost_diff[i] < P['vll']:
            cost_diff[i] = P['vll']
        noise_var = P['var'] * cost_diff[i]
        if noise_var.ndim==1:
            noise_var= noise_var.reshape(-1,1)
        parameter[1, i] = cost_diff[i]

        trj_cost = np.zeros(P['K'])
        sim_state = np.tile(trial_state[:, i].reshape(-1, 1), (1, P['K']))
        noise_seq = np.random.randn(P['Ctrl_dim'], P['K'], P['Horizon_size'])
        for i_noise in range(P['Horizon_size']):
            noise_seq[:, :, i_noise] = noise_var * noise_seq[:, :, i_noise]
        horizon_input = np.tile(seq_ctrl[:, :, i:i+P['Horizon_size']], (1, P['K'], 1))
        horizon_input[:, :int(P['random_sample_rate']*P['K']), :] = 0

        for i_sim in range(P['Horizon_size']):
            next_sim_state = Sim_Model(sim_state, horizon_input[:, :, i_sim] + noise_seq[:, :, i_sim], P)
            # 動的障害物のための時刻を計算
            current_time = (i + i_sim) * P['dt']
            trj_cost += Cost_Fcn(next_sim_state, sim_state, P, banned_point, current_time)
            sim_state = next_sim_state

        trj_cost += Term_Cost(sim_state)
        min_cost = np.min(trj_cost)
        norm_cost = np.sum(np.exp(-1 / P['Temp'] * (trj_cost - min_cost)))
        weight = np.exp(-1 / P['Temp'] * (trj_cost - min_cost)) / norm_cost
        for i_update in range(P['Horizon_size']):
            update = np.dot(noise_seq[:, :, i_update], weight)  # (4,)
            update = update.reshape(-1, 1)  # (4, 1)                      
            seq_ctrl[:, :, i + i_update] += update 
                    
        trial_state[:, i + 1] = Sim_Model(trial_state[:, i], seq_ctrl[:, :, i], P).flatten()
        
        # 目標到達判定
        distance_to_goal = np.linalg.norm(trial_state[0:3, i+1] - P['Goal_state'][0:3, 0])
        goal_threshold = P.get('goal_threshold', 0.2)  # 目標到達とみなす距離の閾値[m]
        if distance_to_goal <= goal_threshold:
            print('ゴール！！')
            print(f'ゴール到達位置: x={trial_state[0, i+1]:.2f}, y={trial_state[1, i+1]:.2f}, z={trial_state[2, i+1]:.2f}')
            print(f'目標座標との距離: {distance_to_goal:.2f}m')
            break
        
        # 動的障害物のための時刻を計算
        current_time = (i + 1) * P['dt']
        fale = check(trial_state[:, i + 1], trial_state[:, i], P, current_time)
        if np.any(fale >= 1):
            print('衝突')
            # 衝突した1ステップ前の位置を記録
            collision_position = trial_state[0:3, i].copy()
            print(f'衝突位置（1ステップ前）: x={collision_position[0]:.2f}, y={collision_position[1]:.2f}, z={collision_position[2]:.2f}')
            break
        # bp_switch の分岐
        if P.get('bp_switch', 0) == 1:
            if (
                i >= int(P['check'] / P['dt'])
                and np.linalg.norm(P['Goal_state'][0:2, 0] - trial_state[0:2, i]) >= 1
                and bpc_interval <= 0                                                     #禁止点を探す
                ): 
                NGvec = P['Goal_state'][0:2, 0] - trial_state[0:2, i]
                NGang = np.arccos(np.dot(NGvec, np.array([0, 1])) / np.linalg.norm(NGvec))
                Z = -NGang
                ROT = np.array([
                    [np.cos(Z), -np.sin(Z)],
                    [np.sin(Z),  np.cos(Z)]
                ])
                idx_start = int(1 + i - P['check'] / P['dt'])
                idx_end = int(i + 1)
                block_check_state = trial_state[:, idx_start:idx_end].copy()
                for j in range(block_check_state.shape[1]):
                    block_check_state[3:5, j] = ROT @ block_check_state[3:5, j]
                    block_check_state[0:2, j] = block_check_state[0:2, j] - trial_state[0:2, i]
                    block_check_state[0:2, j] = ROT @ block_check_state[0:2, j]

                # if np.mean(block_check_state[1, :] <= 1e-8):
                if np.max(block_check_state[1, :]) - np.min(block_check_state[1, :]) <= 0.5: #ロック判定 ゴールへ向かう向きに対する座標のブレ（Max-Min）が0.5以下
                    ban_count += 1
                    print('ロック発生')
                    banned_point[:, ban_count - 1] = trial_state[0:3, i]
                    bpc += 1
                    agbp[:, bpc - 1] = trial_state[0:3, i]
                    bpc_interval = int(P['check'] / P['dt'])

            bpc_interval -= 1

            ng_st = 0
            if 'object' in P:
                obj = P['object']
                if obj.ndim == 3:
                    n_obs = obj.shape[2]
                else:
                    n_obs = 1
                for a in range(n_obs):
                    if obj.ndim == 3:
                        xv = obj[0, :, a]
                        yv = obj[1, :, a]
                    else:
                        xv = obj[0, :]
                        yv = obj[1, :]
                    poly = Polygon(np.stack([xv, yv], axis=1))
                    ng_line = LineString([
                        trial_state[0:2, i],
                        P['Goal_state'][0:2, 0]
                        ])
                    # intersect: polyと線分が交差していればTrue
                    if poly.intersects(ng_line):
                        continue
                    else:
                        ng_st += 1
                if ng_st == n_obs:
                    banned_point = np.full((3, 100), np.nan)
        print(f"Trial loop {i+1}/{P['Trial_size']-1} 経過時間: {time.time() - loop_start:.3f} 秒")
    return trial_state, seq_ctrl, agbp, bpc, parameter, collision_position
