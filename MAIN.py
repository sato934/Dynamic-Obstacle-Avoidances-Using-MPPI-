#11/10追加 円形障害物に衝突した時何も起こっていない，複数ではない
#11/12追加 経路を同時に描写，障害物に接触した際バツ印を表示 なぜかできない→円形障害物を障害物と認識していない背景のようになっている静止障害物のように認識させる必要がある
#課題１ 衝突判定がうまくいっていない，←クリア
#課題２ 新たなコスト関数を導入する← Qprobは導入できていない マハラノビス距離使っていない　できたかな
#課題３ パラメータの調整 ←障害物がすくないとうまくいくが多いとうまくいかない
#課題４ 動的障害物の数を増やすと衝突する　よけてはいる　もう少し
#課題５ 効率悪い 障害物からの距離でコストを付与するか決める例半径より半分小さい
#課題６ 時間遅すぎ GPUベースでやってみる
#やってみること  障害物正面にある時離れるのではなく止まるようにしてみる　詳しくは考える
# i=7　評価区間３　サンプル数５０００　が１番良い 許容リスクを変えてみるのもありかも
# iHorizonが長い場合:

#15ステップ分の目標コスト累積 = 25,000 × 15 = 375,000
#障害物コストは接近時の数ステップのみ = 65,766 × 3 ≈ 197,298
#目標コストが勝ってしまう
#正しい対策
#あなたの直感は正しい
#Horizon=10秒に戻して、コストバランスを調整すべき


##const double distanceThreshold = 1.0;
##if (distance >= distanceThreshold) {
##   // trivial case: agent is far away
##   Ped::Tvector desiredDirection = diff.normalized();
##    Ped::Tvector force =
##        (desiredDirection * agentIn.getVmax() - agentIn.getVelocity()) /
##        agentIn.getRelaxationTime();
##    if (desiredDirectionOut != nullptr) {
##      *desiredDirectionOut = desiredDirection;
##    }
##    return force;
##  } else {
##    // agent is already very close to the waypoint
##    Ped::Tvector velocity = agentIn.getVelocity();
##    // → decelerate agent
##    Ped::Tvector decelerationForce = -velocity / agentIn.getRelaxationTime();
##    // → move agent to the correct place
##    Ped::Tvector projection = velocity * agentIn.getRelaxationTime();
##    Ped::Tvector projectedDiff = diff - projection;
##    Ped::Tvector projectionForce = projectedDiff / agentIn.getRelaxationTime();
##
##    Ped::Tvector force = decelerationForce + projectionForce;
##    if (desiredDirectionOut != nullptr) {
##      *desiredDirectionOut = velocity;
##    }
##    return force;
##  }                     エージェントが遠い場合ウェイポイントへ向かう力を計算エージェントが近い場合(distance < 1.0)減速して正確な位置に移動する力を計算
##                        速度を考慮する　今は考慮していない位置の重みで何とかしている状況　　もしかしたらつかうかも
import numpy as np
from Load_Settings import Load_Settings
from MPPI_GT import MPPI_GT
from Graph_x import Graph_x
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

print('Finish!!')
print(datetime.now())
