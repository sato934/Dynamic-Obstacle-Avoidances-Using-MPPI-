# Dynamic Obstacle Avoidance Using MPPI

MPPIアルゴリズムを用いた動的障害物回避のシミュレーション実装です。単一エージェントおよびマルチエージェント環境における経路計画と障害物回避を実現します。

---

## 概要

本リポジトリは **Model Predictive Path Integral (MPPI)** 制御を用いて、動的に移動する障害物を回避しながら目標地点へ到達するクワッドロータ経路計画シミュレータを実装します。

MPPIはサンプルベースのモデル予測制御手法であり、多数のランダムな制御入力シーケンスをサンプリングし、その結果のコストを評価することで最適な制御入力を計算します。勾配計算が不要なため、非線形・非凸なコスト関数にも対応できます。

---

## ファイル構成

```
.
├── MAIN.py                      # 単一エージェントシミュレーションのメインスクリプト
├── MAIN_MultiAgent.py           # マルチエージェントシミュレーションのメインスクリプト
├── MPPI_GT.py                   # MPPIコントローラの実装
├── MPPI_MultiAgent.py           # マルチエージェント向けMPPI実装
├── Cost_Fcn.py                  # コスト関数（単一エージェント）
├── Cost_Fcn_MultiAgent.py       # コスト関数（マルチエージェント）
├── Ang_Model.py                 # 角度モデル
├── Pos_Model.py                 # 位置モデル
├── Sim_Model.py                 # シミュレーションモデル
├── Term_Cost.py                 # 終端コスト
├── Load_Settings.py             # パラメータ設定
├── FolderCheck.py               # 出力フォルダの確認・作成
├── Graph_x.py                   # 単一エージェントのアニメーション描画
├── Graph_Distance.py            # 距離のグラフ描画
├── Graph_MultiAgent.py          # マルチエージェントのアニメーション描画
├── Graph_MultiAgent_Analysis.py # マルチエージェント結果の分析・描画
└── check.py                     # デバッグ・確認用スクリプト
```

---

## 必要環境

- Python 3.8 以上
- NumPy
- Matplotlib
- Numba（JITコンパイルによる数値計算の高速化）

インストール:

```bash
pip install numpy matplotlib
```

---

## 使い方

### 単一エージェントシミュレーション

```bash
python MAIN.py
```

### マルチエージェントシミュレーション

```bash
python MAIN_MultiAgent.py
```

### シミュレーション設定

`Load_Settings.py` 内でシミュレーションのパラメータを変更できます。主な設定項目：

| パラメータ | 説明 |
|---|---|
| サンプル数 | MPPIがサンプリングする制御入力シーケンスの数 |
| ホライゾン長 | 予測ホライゾンのステップ数 |
| 温度パラメータ λ | コスト重み付けの感度 |
| 障害物の数・速度 | 動的障害物の設定 |
| 目標座標 | クワッドロータの目標位置 |

---

## アルゴリズム概要

MPPIの計算フローは以下の通りです：

1. **サンプリング**：現在の制御入力にガウスノイズを加え、K個の制御シーケンスを生成
2. **ロールアウト**：各サンプルについてシミュレーションモデルで将来の状態を予測
3. **コスト計算**：障害物距離・目標距離・制御量などからコストを計算
4. **重み付け平均**：指数関数的な重みを用いて最適制御入力を計算
5. **適用**：計算した制御入力をシステムへ適用し、次のステップへ

---

## 出力

シミュレーション終了後、以下のグラフが生成されます：

- エージェントの軌跡と障害物の位置
- 状態量（位置・速度・角度），制御の時系列
- 障害物との距離推移

---

## 参考文献

- Williams, G., et al. "Model Predictive Path Integral Control: From Theory to Parallel Computation." *Journal of Guidance, Control, and Dynamics*, 2017.
- Williams, G., et al. "Information Theoretic MPC for Model-Based Reinforcement Learning." *ICRA*, 2017.
