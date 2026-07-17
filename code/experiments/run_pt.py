"""
Parallel Tempering（交換モンテカルロ法）実験スクリプト。
M4 査読対応: 古典的サンプリングベースラインとの比較。

査読2 FB1対応: 大関先生から「レプリカ選択・温度ラダー設計はフェアか」という
指摘を受け、以下のパイプラインに変更した（旧版は最低温レプリカ index 0 の
サンプルのみを report していたが、これは16レプリカ中ほぼ最悪の選択だった）。

  1. 全16レプリカのサンプルを収集する（`run_all_replicas`; 交換法により
     全レプリカが周期的に交換されるため、中間温度レプリカも追加サンプリング
     コストなしで利用可能）。
  2. どのレプリカ（温度）を採用するかを、10-fold cross-validation で選択する
     （RMSEに基づく後知恵選択にならないよう、train側9-foldで最良のレプリカを
     選び、選択に使っていない残り1-foldだけで評価する; M5 の (alpha,beta)
     選択と同じ発想）。
  3. 全foldで選ばれたレプリカ（コンセンサス）の全 num_reads サンプルを、
     他ソルバーと同一形式の results.json / sampleset.json / traj.csv として
     保存する。10-fold CV の fold 別詳細（circularity-free な held-out 指標）
     は cv_summary.json に保存する。

使い方:
    uv run python experiments/run_pt.py
    uv run python experiments/run_pt.py --reads 200 --cv-folds 5  # 動作確認用
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dimod

from src.flow import build_flow_matrix, reconstruct_proportions
from src.graph import get_distances
from src.metrics import rmse, squared_loss
from src.qubo import build_qubo
from src.solvers.parallel_tempering import run_all_replicas
from src.trajectory import compute_violation_rate, decode_traj, save_results
from src.transition import build_transition_P, _safe_row_normalize

# ===== パラメータ（SQA/SA と同一の QUBO 設定） =====
ALPHA = 0.3
BETA = 0.55
LAMBDA_ONEHOT = 13.0
LAMBDA_P = 5.0
LAMBDA_DIV = 1.0
LAMBDA_ENTRY = 2.0
LAMBDA_MOVE = 0.5
SEED = 3
NUM_READS = 30_000
OUTPUT_DIR = Path("results/pt_30k")

# Parallel Tempering 固有設定（旧版と同一の対数等間隔ラダー）
PT_CONFIG = {
    "n_replicas": 16,
    "beta_min": 0.05,
    "beta_max": 60.0,
    "n_sweeps_burn_in": 3000,
    "sweeps_per_sample": 30,
    "swap_interval": 5,
}

K_FOLDS = 10
CV_SEED = 12345  # fold分割専用のseed（PT本体のseed=SEEDとは独立）


def _rmse_for_indices(traj_all: list, idx: np.ndarray, T: int, N: int, p_true: np.ndarray) -> float:
    subset = [traj_all[i] for i in idx]
    p_prime = reconstruct_proportions(subset, T, N)
    return rmse(p_true, p_prime)


def _select_replica_by_cv(
    traj_by_replica: list[list[list[int]]],
    betas: np.ndarray,
    samples: np.ndarray,
    labels: list,
    T: int,
    N: int,
    Np: int,
    p_true: np.ndarray,
    num_reads: int,
    k_folds: int,
) -> dict:
    """10-fold CV でレプリカ（温度）を選択し、fold別・集計済みの held-out 指標を返す。"""
    fold_rng = np.random.default_rng(CV_SEED)
    perm = fold_rng.permutation(num_reads)
    folds = np.array_split(perm, k_folds)

    n_replicas = len(betas)
    test_rmses, test_violations, test_unique_fracs, selected_replicas = [], [], [], []

    for fold_i in range(k_folds):
        heldout_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])

        train_rmse_by_replica = [
            _rmse_for_indices(traj_by_replica[r], train_idx, T, N, p_true)
            for r in range(n_replicas)
        ]
        r_star = int(np.argmin(train_rmse_by_replica))
        selected_replicas.append(r_star)

        test_rmse = _rmse_for_indices(traj_by_replica[r_star], heldout_idx, T, N, p_true)

        heldout_samples = samples[r_star][heldout_idx]
        sampleset = dimod.SampleSet.from_samples(
            (heldout_samples, labels), vartype="BINARY",
            energy=np.zeros(len(heldout_idx)), num_occurrences=np.ones(len(heldout_idx), dtype=int),
        )
        test_violation = compute_violation_rate(sampleset, T, Np)
        heldout_traj = [traj_by_replica[r_star][i] for i in heldout_idx]
        unique_frac = len({tuple(t) for t in heldout_traj}) / len(heldout_traj)

        test_rmses.append(test_rmse)
        test_violations.append(test_violation)
        test_unique_fracs.append(unique_frac)

        print(f"  fold {fold_i:2d}: selected replica={r_star:2d} (beta={betas[r_star]:.4f})  "
              f"held-out: rmse={test_rmse:.4f}  violation_rate={test_violation:.4f}  "
              f"unique_frac={unique_frac:.4f}")

    test_rmses_arr = np.array(test_rmses)
    test_violations_arr = np.array(test_violations)
    test_unique_fracs_arr = np.array(test_unique_fracs)

    # 全foldで一貫して同じレプリカが選ばれるはず（論文本文で報告している通り）。
    # 念のため、最頻値をコンセンサスレプリカとして採用する。
    consensus_replica = int(np.bincount(selected_replicas).argmax())

    return {
        "k_folds": k_folds,
        "selected_replicas": selected_replicas,
        "selected_betas": [round(float(betas[r]), 6) for r in selected_replicas],
        "consensus_replica": consensus_replica,
        "consensus_beta": round(float(betas[consensus_replica]), 6),
        "unanimous": len(set(selected_replicas)) == 1,
        "test_rmse_per_fold": test_rmses_arr.tolist(),
        "test_violation_per_fold": test_violations_arr.tolist(),
        "test_unique_frac_per_fold": test_unique_fracs_arr.tolist(),
        "test_rmse_mean": float(test_rmses_arr.mean()),
        "test_rmse_std": float(test_rmses_arr.std(ddof=1)) if k_folds > 1 else 0.0,
        "test_violation_mean": float(test_violations_arr.mean()),
        "test_unique_frac_mean": float(test_unique_fracs_arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", type=int, default=NUM_READS)
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--cv-folds", type=int, default=K_FOLDS)
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(__file__).resolve().parents[1] / "data" / "observations.json"
    with open(data_path, encoding="utf-8") as f:
        C_history = np.array(json.load(f), dtype=float)
    T, N = C_history.shape
    Np = N + 1

    p_true = _safe_row_normalize(C_history)
    distances, _ = get_distances()
    P = build_transition_P(C_history, distances, ALPHA, BETA)
    Q = build_qubo(P, T, LAMBDA_ONEHOT, LAMBDA_P, LAMBDA_DIV, LAMBDA_ENTRY, LAMBDA_MOVE)

    print(f"[PT] Running Parallel Tempering, collecting all {PT_CONFIG['n_replicas']} replicas "
          f"({args.reads} reads each) ...")
    run = run_all_replicas(Q, num_reads=args.reads, seed=SEED, **PT_CONFIG)
    labels = run["labels"]
    betas = run["betas"]
    samples = run["samples"]      # (n_replicas, num_reads, n_vars)
    energies = run["energies"]    # (n_replicas, num_reads)
    n_replicas = samples.shape[0]

    print("[PT] Decoding trajectories for all replicas ...")
    decode_rng = np.random.default_rng(SEED)
    traj_by_replica = [
        [decode_traj(dict(zip(labels, samples[r, s])), T, Np, decode_rng) for s in range(args.reads)]
        for r in range(n_replicas)
    ]

    print(f"\n[PT] Selecting replica temperature via {args.cv_folds}-fold CV "
          f"(circularity-free out-of-sample selection) ...")
    cv = _select_replica_by_cv(
        traj_by_replica, betas, samples, labels, T, N, Np, p_true, args.reads, args.cv_folds,
    )
    r_star = cv["consensus_replica"]
    print(f"\n[PT] Consensus replica={r_star} (beta={betas[r_star]:.4f}), "
          f"unanimous across folds={cv['unanimous']}")
    print(f"[PT] Held-out (circularity-free): RMSE={cv['test_rmse_mean']:.6f} ± {cv['test_rmse_std']:.6f}  "
          f"violation_rate={cv['test_violation_mean']:.6f}  unique_frac={cv['test_unique_frac_mean']:.6f}")

    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    print(f"[PT] CV summary saved to {output_dir}/cv_summary.json")

    # --- コンセンサスレプリカの全サンプルを、他ソルバーと同一形式で保存する ---
    num_occurrences = np.ones(args.reads, dtype=int)
    sampleset = dimod.SampleSet.from_samples(
        (samples[r_star], labels), vartype="BINARY",
        energy=energies[r_star], num_occurrences=num_occurrences,
        info={
            "method": "parallel_tempering",
            "n_replicas": n_replicas,
            "betas": betas.tolist(),
            "selected_replica": r_star,
            "selected_beta": float(betas[r_star]),
            "selection_method": f"{args.cv_folds}-fold cross-validation",
            **PT_CONFIG,
        },
    )
    traj_list = traj_by_replica[r_star]
    violation = compute_violation_rate(sampleset, T, Np)
    p_prime = reconstruct_proportions(traj_list, T, N)
    F = build_flow_matrix(traj_list, T, Np)
    loss = squared_loss(p_true, p_prime)
    r = rmse(p_true, p_prime)

    print(f"\n[PT] Selected-replica full-sample metrics (in-sample, all {args.reads} reads): "
          f"violation_rate={violation:.4f}  RMSE={r:.6f}  loss={loss:.6f}")

    save_results(sampleset, traj_list, Q, T, Np, output_dir)

    results = {
        "solver": "PT",
        "alpha": ALPHA,
        "beta": BETA,
        "num_reads": args.reads,
        "num_sweeps": None,
        "seed": SEED,
        "lambda_onehot": LAMBDA_ONEHOT,
        "lambda_P": LAMBDA_P,
        "lambda_div": LAMBDA_DIV,
        "lambda_entry": LAMBDA_ENTRY,
        "lambda_move": LAMBDA_MOVE,
        "violation_rate": round(violation, 6),
        "rmse": round(r, 6),
        "loss": round(loss, 6),
        "C": C_history.tolist(),
        "p_true": p_true.tolist(),
        "p_prime": p_prime.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "pt_config": PT_CONFIG,
        "selected_replica_index": r_star,
        "selected_replica_beta": round(float(betas[r_star]), 6),
        "selection_method": f"{args.cv_folds}-fold cross-validation (see cv_summary.json)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[PT] Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
