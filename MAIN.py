import numpy as np
from Load_Settings import Load_Settings
from MPPI_GT import MPPI_GT
from Graph_x import Graph_x
from Graph_Distance import plot_distance_time_graph
from datetime import datetime

# --- 初期化 ---
print(datetime.now())

P = Load_Settings(8)  # パラメータ設定の読み込み 【ここの引数を変えることで障害物の形状を変更可能】
np.random.seed(P['seed'])

# データ格納用
ds_state = np.full((P['Dataset_size'], P['State_dim']), np.nan)
ds_ctrl = np.full((P['Dataset_size'], P['Ctrl_dim']), np.nan)

# グローバル変数相当
agbp = np.full((3, 1), np.nan)
bpc = 0
parameter = np.full((2, P['Trial_size'] * P['Trial_num']), np.nan)


# MPPI + agbp_list, bpc_list, ds_state_list 構築
agbp_list = []
bpc_list = []
ds_state_list = []
collision_list = []  # 衝突位置リスト
for i in range(P['Trial_num']):
    t0 = datetime.now()
    print(f"Iteration: {i+1}")
    agbp = np.full((3, 100), np.nan)
    bpc = 0
    parameter = np.full((2, P['Trial_size']), np.nan)
    trial_state, seq_ctrl, agbp, bpc, parameter, collision_pos = MPPI_GT(P, agbp, bpc, parameter)
    ds_state[i*P['Trial_size']:(i+1)*P['Trial_size'], :] = trial_state.T
    ds_ctrl[i*P['Trial_size']:(i+1)*P['Trial_size'], :] = np.squeeze(seq_ctrl[:, :, :P['Trial_size']]).T
    agbp_list.append(agbp.copy())
    bpc_list.append(bpc)
    ds_state_list.append(trial_state.T)
    collision_list.append(collision_pos)  # 衝突位置を記録
    t1 = datetime.now()
    print(f"Elapsed: {t1-t0}")

Graph_x(ds_state_list, P, agbp_list, bpc_list, collision_list)

# 距離時系列グラフの作成
print("\n=== 距離時系列グラフを作成中 ===")
# ds_state_listの各要素をndarrayに変換
ds_state_array_list = []
for ds in ds_state_list:
    if isinstance(ds, np.ndarray):
        # shape: (num_steps, 12) -> (12, num_steps) に転置
        ds_state_array_list.append(ds.T)
    else:
        ds_state_array_list.append(ds)

min_dist, min_time, obs_type = plot_distance_time_graph(
    ds_state_array_list, P, 
    save_path='ani_distance_time.png'
)

print('Finish!!')
print(datetime.now())
