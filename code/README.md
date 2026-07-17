# Pedestrian Flow Prediction Using QA Sampling

QAサンプリングを用いた歩行者フロー予測・サイン配置の実験コード。

このディレクトリは `public_release/` リポジトリの `code/` です。リポジトリ全体の構成や、
ここに含まれる結果と論文記載値との対応は[トップレベルREADME](../README.md)を参照してください。

---

## セットアップ

```bash
# このディレクトリ（code/）で実行する
cd code/

# 依存パッケージのインストール（初回のみ）
uv sync

# 動作確認
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
```

---

## ディレクトリ構成

```
code/
├── src/            コアライブラリ（graph, transition, qubo, trajectory, flow, metrics, solvers/）
├── experiments/    実験スクリプト
├── figures/        論文用図の生成スクリプト
├── analysis/       データ確認・診断スクリプト
├── gridsearch/, distance/, predict/, solve/  補助モジュール
├── data/           入力データ（observations.json。../data/observations.json と同一内容）
├── results/        スクリプトを実行した際の出力先（git管理外。再実行すると生成される）
└── STRUCTURE.md    コード構造の詳細リファレンス
```

各実験について「論文記載値」付きで整理済みの結果（curated results）は、ここではなく
`../experiments/<NN_実験名>/results/` 以下に格納されている。詳細は
[STRUCTURE.md](STRUCTURE.md) と[トップレベルREADME](../README.md)を参照。

---

## 実験の実行

### SQA（シミュレーテッド量子アニーリング）— 論文の主要結果

```bash
# 論文再現（30000 reads、約数分〜数十分）
uv run python experiments/run_sqa.py

# 動作確認用（100 reads、数秒）
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
```

結果は `results/sqa_30k/` に保存される（論文記載値の整理済みコピー: [`../experiments/01_sqa_baseline/results/sqa_30k/`](../experiments/01_sqa_baseline/results/sqa_30k/)）。論文記載値: **RMSE = 0.077708**

### SA（シミュレーテッドアニーリング）

```bash
uv run python experiments/run_sa.py
uv run python experiments/run_sa.py --reads 100 --out results/test_sa
```

結果は `results/sa_30k/` に保存（整理済みコピー: [`../experiments/02_sa_baseline/results/sa_30k/`](../experiments/02_sa_baseline/results/sa_30k/)）。論文参考値: RMSE = 0.0889

### D-Wave 実機（QA）

事前準備: `.env.local` に認証情報を設定する（下記「D-Wave の設定」参照）。

```bash
# 利用可能なソルバーを確認
uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env.local')
from dwave.cloud import Client
print([s.id for s in Client.from_config().get_solvers()])
"

# 実行（30000 reads）
uv run python experiments/run_qa.py

# annealing time を変更する場合
uv run python experiments/run_qa.py --annealing-time 100

# 出力先を変更する場合
uv run python experiments/run_qa.py --out results/qa_srt_test
```

### α/β グリッドサーチ

```bash
# SA バックエンドで探索（推奨: 速い）
uv run python experiments/gridsearch.py --solver sa --reads 100

# SQA バックエンドで探索
uv run python experiments/gridsearch.py --solver sqa --reads 100 --final-reads 30000
```

グリッドサーチは途中再開可能（`history_stage1.csv` / `history_stage2.csv` にキャッシュ）。

---

## 論文用図の生成

### 全図を一括生成

```bash
# SQA と QA の結果から全図を生成（figures/out/ に保存）
uv run python figures/generate_all.py

# 出力先を指定
uv run python figures/generate_all.py --out /path/to/paper/figures/

# 使用する results.json を指定（curated結果を直接使う例）
uv run python figures/generate_all.py \
    --sqa ../experiments/01_sqa_baseline/results/sqa_30k/results.json \
    --qa  ../experiments/03_qa_hardware_baseline/results/qa_advantage2_30k/results.json \
    --out figures/out/
```

### 個別に生成

```bash
# 会場グラフ（Fig 1）
uv run python figures/fig_graph.py --out figures/out/

# 観測データ（ゾーン合計 + 時系列ヒートマップ）
uv run python figures/fig_data.py --out figures/out/

# p_true vs p_prime ヒートマップ（Fig 2）
uv run python figures/fig_proportions.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/

# フロー行列 F（Fig 3）
uv run python figures/fig_flow_matrix.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/

# エッジフロー比率グラフ（Fig 4）
uv run python figures/fig_edge_flow.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --out figures/out/
uv run python figures/fig_edge_flow.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json --top 5  # 上位5を強調
```

---

## 実験結果の確認

各実験の出力ディレクトリに以下が保存される。

| ファイル | 内容 |
|---|---|
| `results.json` | RMSE・violation_rate・P・F・p_true・p_prime 等 |
| `sampleset.json` | 全サンプルセット（SQA: ~300MB、QA: ~60MB） |
| `traj.csv` | 全軌跡（行=サンプル、列=時刻） |
| `energy_histogram_hw.png` | ハードウェア報告エネルギー分布 |
| `energy_histogram_overlay.png` | Raw vs Fixed（QUBO再計算）のエネルギー比較 |
| `energy_histogram_fixed.png` | Fixed のみのエネルギー分布 |

```bash
# RMSE と violation_rate だけ確認したい場合（自分で実行した結果の例）
python -c "
import json; r = json.load(open('results/sqa_30k/results.json'))
print(f'RMSE={r[\"rmse\"]:.6f}  violation={r[\"violation_rate\"]:.4f}')
"

# curated（論文記載値）の結果を直接確認する場合
python -c "
import json; r = json.load(open('../experiments/01_sqa_baseline/results/sqa_30k/results.json'))
print(f'RMSE={r[\"rmse\"]:.6f}  violation={r[\"violation_rate\"]:.4f}')
"
```

---

## D-Wave の設定

`.env.local` に以下を記載する（git 管理外）:

```
DWAVE_SOLVER_NAME=Advantage2_system2.3
DWAVE_API_TOKEN=your-token-here
```

**利用可能なソルバー**は時期によって変わる。事前に上記の確認コマンドで確認すること。

過去に使用したソルバー:
- `Advantage2_system1.6` — 廃止済み（論文記載値はこれで取得）
- `Advantage2_system1.11` — 廃止済みの可能性あり

---

## QUBO パラメータ

論文に使用した値。変更する場合は各 `experiments/run_*.py` 冒頭の定数を編集する。

| パラメータ | 値 | 意味 |
|---|---|---|
| `alpha` | 0.3 | 距離減衰係数 |
| `beta` | 0.55 | 目的地人気度の指数 |
| `lambda_onehot` | 13.0 | One-Hot 制約（ハード制約） |
| `lambda_P` | 5.0 | 遷移確率への追従（ソフト） |
| `lambda_div` | 1.0 | 訪問分散 |
| `lambda_entry` | 2.0 | 外部ノード出入り制御 |
| `lambda_move` | 0.5 | 内部移動の平滑化 |

---

## 既存の実験結果（再実行不要なもの）

論文記載値の整理済み結果（curated results）は `code/` 以下ではなく、リポジトリ直下の
`experiments/<NN_実験名>/results/` 以下に格納されている。

| ディレクトリ（`../experiments/` 以下） | ソルバー | RMSE | violation_rate | 備考 |
|---|---|---|---|---|
| `01_sqa_baseline/results/sqa_30k/` | SQA | 0.077708 | 0.0% | **論文記載値** |
| `03_qa_hardware_baseline/results/qa_advantage2_30k/` | D-Wave Advantage2_system1.6 | 0.079481 | 95.1% | **論文記載値**（ソルバー廃止済み） |
| `02_sa_baseline/results/sa_30k/` | SA | 0.088900 | 0.0% | 参考値 |
| `02_sa_baseline/results/sa_gridsearch/` | SA + 2段階探索 | 0.069900 | 0.0% | α/β 自動決定 |
| `04_classical_pt_baseline/results/pt_30k/` | Parallel Tempering（温度をCVで選択） | 0.069184 | 0.6% | **論文記載値**（10-fold CVでレプリカ温度を選択; `cv_summary.json`参照） |
| `05_dynamic_range_analysis/results/qa_autoscale_30k/` | D-Wave Advantage2_system1（手動スケーリング） | 0.106361 | 56.5% | 追加検証（M1） |
| `06_gauge_averaging_srt/results/qa_srt_30k/` | D-Wave Advantage2_system1（SRT 100ゲージ） | 0.167999 | 51.5% | 追加検証（M2） |
| `10_new_chip_rerun/results/qa_advantage2_system1_30k/` | D-Wave Advantage2_system1 | 0.092575 | 95.5% | 新チップでの再実行（M10補足） |

各実験フォルダのREADMEに、対応する論文中の主張・実行スクリプト・解釈の詳細が書かれている。

