"""
動的障害物の衝突判定テストスクリプト
もう使わない
"""
import numpy as np
from Load_Settings import Load_Settings
from check import check

# 設定を読み込み
P = Load_Settings(6)
np.random.seed(P['seed'])

print("=" * 60)
print("動的障害物の衝突判定テスト")
print("=" * 60)

# 動的障害物の情報を表示
print(f"\n[設定情報]")
print(f"動的障害物: {P.get('dynamic', False)}")
print(f"動的障害物の形状: {P['dynamic_obj'].shape}")
print(f"障害物数: {P['dynamic_obj'].shape[2]}")
print(f"各障害物の点数: {P['dynamic_obj'].shape[1]}")

# 最初の障害物の初期位置を表示
print(f"\n[障害物0の初期位置]")
obs0 = P['dynamic_obj'][:, :, 0]
center0 = obs0.mean(axis=1)
print(f"中心座標: ({center0[0]:.2f}, {center0[1]:.2f})")

# waypointsとsegment_timesを表示
if isinstance(P['dynamic_waypoints'], list):
    print(f"\n[障害物0のwaypoints]")
    print(P['dynamic_waypoints'][0])
    print(f"\n[障害物0のsegment_times]")
    print(P['dynamic_segment_times'][0])

# テスト1: 初期位置での衝突判定
print("\n" + "=" * 60)
print("テスト1: 初期位置(t=0)で障害物の中心に経路を配置")
print("=" * 60)

# 障害物の中心に点を配置
test_state = np.zeros((12, 1))
test_state[0, 0] = center0[0]  # x座標
test_state[1, 0] = center0[1]  # y座標
test_state[2, 0] = 5.0         # z座標

print(f"テスト位置: ({test_state[0, 0]:.2f}, {test_state[1, 0]:.2f}, {test_state[2, 0]:.2f})")

# 衝突判定を実行（t=0）
result = check(test_state, test_state, P, t=0.0)
print(f"衝突判定結果: {result[0]}")
if result[0]:
    print("✓ 衝突が検出されました！")
else:
    print("✗ 衝突が検出されませんでした（問題あり）")

# テスト2: 時刻t=5秒での衝突判定
print("\n" + "=" * 60)
print("テスト2: 時刻t=5秒での障害物位置での衝突判定")
print("=" * 60)

t_test = 5.0
result2 = check(test_state, test_state, P, t=t_test)
print(f"時刻: {t_test}秒")
if 'current_dynamic_obj' in P:
    updated_center = P['current_dynamic_obj'][:, :, 0].mean(axis=1)
    print(f"更新後の障害物0の中心: ({updated_center[0]:.2f}, {updated_center[1]:.2f})")
print(f"衝突判定結果: {result2[0]}")

# テスト3: 障害物から離れた位置
print("\n" + "=" * 60)
print("テスト3: 障害物から離れた位置での衝突判定")
print("=" * 60)

test_state_far = np.zeros((12, 1))
test_state_far[0, 0] = 10.0  # x座標（遠い位置）
test_state_far[1, 0] = 10.0  # y座標（遠い位置）
test_state_far[2, 0] = 5.0   # z座標

print(f"テスト位置: ({test_state_far[0, 0]:.2f}, {test_state_far[1, 0]:.2f}, {test_state_far[2, 0]:.2f})")
result3 = check(test_state_far, test_state_far, P, t=0.0)
print(f"衝突判定結果: {result3[0]}")
if not result3[0]:
    print("✓ 衝突なし（正常）")
else:
    print("✗ 衝突検出（問題あり）")

print("\n" + "=" * 60)
print("テスト完了")
print("=" * 60)
