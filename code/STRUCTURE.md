# code/ フォルダ構造メモ

最終更新: 2026-06-10

このファイルは `public_release/code/` の構造を説明する。リポジトリ全体の構成
（`../data/`, `../experiments/` との関係）については[トップレベルREADME](../README.md)を参照。

---

## フォルダツリー

```
code/
├── src/                        # コアライブラリ（importable）
│   ├── __init__.py
│   ├── graph.py                # get_distances() — グラフ・距離行列
│   ├── transition.py           # build_transition_P() — 遷移確率行列
│   ├── qubo.py                 # build_qubo() — QUBO辞書の構築
│   ├── trajectory.py           # decode_traj(), compute_violation_rate(), save_results()
│   ├── flow.py                 # reconstruct_proportions(), build_flow_matrix()
│   ├── metrics.py              # rmse(), squared_loss()
│   └── solvers/
│       ├── __init__.py         # get_solver(type) — ファクトリ関数
│       ├── base.py             # SolverBase, Qubo, SampleConfig 型定義
│       ├── sa.py                # SASolver (OpenJij SA)
│       ├── sqa.py              # SQASolver (OpenJij SQA)
│       ├── qa.py               # QASolver (D-Wave 実機)
│       └── parallel_tempering.py # PTSolver（古典ベースライン, M4）
│
├── experiments/                # 実行スクリプト（uv run python experiments/xxx.py）
│   ├── _pipeline.py            # run_experiment(), run_gridsearch() — 共通パイプライン
│   ├── _embedding_cache.py     # minor-embeddingのキャッシュ
│   ├── run_sqa.py              # SQA 実験（論文再現: α=0.3, β=0.55, reads=30k）
│   ├── run_sa.py               # SA 実験
│   ├── run_qa.py               # D-Wave 実機実験
│   ├── run_qa_autoscale.py     # auto_scale=False + 手動係数スケーリング（M1）／新チップ再実行（10）
│   ├── run_qa_srt.py           # spin-reversal transform / gauge averaging（M2）
│   ├── run_pt.py               # Parallel Tempering 実験（M4）
│   ├── m5_validation.py        # LOO-CV・感度分析・Bootstrap CI（M5）
│   └── gridsearch.py           # α/β グリッドサーチ → 本番実験
│
├── analysis/                   # 図生成・診断スクリプト（results.json を受け取る）
│   ├── plot_proportions.py     # p_true vs p_prime ヒートマップ
│   ├── plot_flow_matrix.py     # フロー行列 F のヒートマップ
│   ├── plot_edge_flow.py       # グラフ上のエッジフロー可視化
│   ├── boltzmann_subproblem.py # β_eff フィット・TV距離（縮小サブ問題, M11）
│   └── kl_divergence.py        # SQA/QA フロー行列間の KL・JS ダイバージェンス（M10）
│
├── figures/                    # 論文用図の生成スクリプト
│   └── out/                    # 生成された図の出力先
│
├── gridsearch/, distance/, predict/, solve/   # 補助モジュール
│
├── results/                    # スクリプトを実行した際の出力先（git管理外。再実行で生成）
│
├── data/
│   └── observations.json       # 入力データ（12時刻×10ゾーンの観測人数）
│                                # ../data/observations.json と同一内容
│
├── pyproject.toml              # uv 依存関係設定
└── STRUCTURE.md                # このファイル
```

「論文記載値」付きで整理済みの実験結果（curated results）は `code/results/` ではなく、
リポジトリ直下の `experiments/<NN_実験名>/results/` 以下にある（下表参照）。
D-Wave実機を使う実験を再実行する場合は `code/.env.local`（git管理外）に
`DWAVE_API_TOKEN` 等を設定する。

---

## 処理フロー

```
data/observations.json
        ↓
src/graph.py          →  距離行列 D (11×11, 中央値正規化)
        ↓
src/transition.py     →  遷移確率行列 P (11×11)  ← α, β で制御
        ↓
src/qubo.py           →  QUBO辞書 Q              ← λ 5種で制御
        ↓
src/solvers/          →  SampleSet (SA / SQA / QA / PT)
        ↓
src/trajectory.py     →  軌跡リスト (num_reads 本)
        ↓
src/flow.py           →  p' (T×N), F (11×11)
        ↓
src/metrics.py        →  RMSE, violation_rate
        ↓
results/<name>/       →  results.json, traj.csv, sampleset.json, *.png
```

---

## 実験の実行方法

```bash
# 動作確認（100 reads）
uv run python experiments/run_sqa.py --reads 100 --out results/test_sqa
uv run python experiments/run_sa.py  --reads 100 --out results/test_sa

# 論文再現（30000 reads）
uv run python experiments/run_sqa.py
uv run python experiments/run_sa.py

# D-Wave実機（code/.env.local にトークン設定必須）
uv run python experiments/run_qa.py

# α/β グリッドサーチ
uv run python experiments/gridsearch.py --solver sqa

# 図生成（curated結果を使う例）
uv run python analysis/plot_proportions.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json
uv run python analysis/plot_flow_matrix.py ../experiments/01_sqa_baseline/results/sqa_30k/results.json
uv run python analysis/plot_edge_flow.py   ../experiments/01_sqa_baseline/results/sqa_30k/results.json
```

---

## QUBO パラメータ（論文値）

| パラメータ | 値 | 意味 |
|---|---|---|
| `lambda_onehot` | 13.0 | One-Hot 制約（ハード） |
| `lambda_P` | 5.0 | 遷移確率への追従 |
| `lambda_div` | 1.0 | 訪問分散 |
| `lambda_entry` | 2.0 | 外部ノード出入り制御 |
| `lambda_move` | 0.5 | 内部移動の平滑化 |
| `alpha` | 0.3 | 距離減衰係数 |
| `beta` | 0.55 | 目的地人気度の指数 |

---

## 既存結果サマリー

論文記載値・追加検証の整理済み結果は `code/` 以下ではなく、リポジトリ直下の
`experiments/<NN_実験名>/results/` に格納されている。

| ディレクトリ（`../experiments/` 以下） | ソルバー | RMSE | violation_rate | 備考 |
|---|---|---|---|---|
| `01_sqa_baseline/results/sqa_30k/` | SQA | 0.0777 | 0% | **論文記載値** |
| `02_sa_baseline/results/sa_30k/` | SA | 0.0889 | 0% | 参考値 |
| `02_sa_baseline/results/sa_gridsearch/` | SA + 2段階探索 | 0.0699 | 0% | α/β自動決定 |
| `03_qa_hardware_baseline/results/qa_advantage2_30k/` | D-Wave Advantage2_system1.6 | 0.0795 | 95.1% | **論文記載値**（1.6 は廃止済み） |
| `04_classical_pt_baseline/results/pt_30k/` | Parallel Tempering | 0.1915 | 0% | 古典ベースライン（多様性が崩壊, M4） |
| `05_dynamic_range_analysis/results/qa_autoscale_30k/` | D-Wave Advantage2_system1（手動スケーリング） | 0.1064 | 56.5% | 追加検証（M1） |
| `06_gauge_averaging_srt/results/qa_srt_30k/` | D-Wave Advantage2_system1（SRT 100ゲージ） | 0.1680 | 51.5% | 追加検証（M2） |
| `07_hyperparameter_validation/results/m5_validation/` | — | LOO-CV 0.308±0.026 | — | 感度分析・Bootstrap CI（M5） |
| `08_kl_divergence_analysis/results/kl_divergence.json` | — | — | — | SQA/QA間のKL・JS divergence（M10） |
| `09_effective_temperature_analysis/results/` | SQA / QA | — | — | β_eff・TV距離（M11） |
| `10_new_chip_rerun/results/qa_advantage2_system1_30k/` | D-Wave Advantage2_system1 | 0.0926 | 95.5% | 新チップでの再実行（M10補足） |
