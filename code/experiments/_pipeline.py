"""
実験共通パイプライン。
各 run_*.py スクリプトからインポートして使う。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.flow import build_flow_matrix, reconstruct_proportions
from src.metrics import rmse, squared_loss
from src.qubo import build_qubo
from src.solvers.base import SolverBase
from src.trajectory import compute_violation_rate, decode_traj, save_results
from src.transition import build_transition_P, _safe_row_normalize


def run_experiment(
    *,
    C_history: np.ndarray,
    distances: np.ndarray,
    alpha: float,
    beta: float,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    solver: SolverBase,
    num_reads: int,
    num_sweeps: int | None = None,
    seed: int | None,
    output_dir: Path,
    solver_name: str,
) -> dict:
    """
    1回の実験を実行し、results.json 等を output_dir に保存して結果 dict を返す。

    num_sweeps: SA/SQA の sweep 数。None の場合はソルバーのデフォルト値を使用。
                論文では num_sweeps=100 を使用。QA (D-Wave) には不要。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    T, N = C_history.shape
    Np = N + 1
    rng = np.random.default_rng(seed)

    p_true = _safe_row_normalize(C_history)

    # 1. 遷移確率行列
    P = build_transition_P(C_history, distances, alpha, beta)

    # 2. QUBO 構築
    Q = build_qubo(P, T, lambda_onehot, lambda_P, lambda_div, lambda_entry, lambda_move)

    # 3. サンプリング
    print(f"[{solver_name}] Sampling {num_reads} reads …")
    cfg: dict = {"num_reads": num_reads}
    if num_sweeps is not None:
        cfg["num_sweeps"] = num_sweeps
    if seed is not None:
        cfg["seed"] = seed
    sampleset = solver.solve(Q, sample_config=cfg)

    # 4. デコード
    traj_list = [decode_traj(dict(s), T, Np, rng) for s in sampleset.samples()]

    # 5. 評価
    violation = compute_violation_rate(sampleset, T, Np)
    p_prime = reconstruct_proportions(traj_list, T, N)
    F = build_flow_matrix(traj_list, T, Np)
    loss = squared_loss(p_true, p_prime)
    r = rmse(p_true, p_prime)

    print(f"[{solver_name}] violation_rate={violation:.4f}  RMSE={r:.6f}  loss={loss:.6f}")

    # 6. ファイル保存
    save_results(sampleset, traj_list, Q, T, Np, output_dir)

    results = {
        "solver": solver_name,
        "alpha": alpha,
        "beta": beta,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "seed": seed,
        "lambda_onehot": lambda_onehot,
        "lambda_P": lambda_P,
        "lambda_div": lambda_div,
        "lambda_entry": lambda_entry,
        "lambda_move": lambda_move,
        "violation_rate": round(violation, 6),
        "rmse": round(r, 6),
        "loss": round(loss, 6),
        "C": C_history.tolist(),
        "p_true": p_true.tolist(),
        "p_prime": p_prime.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[{solver_name}] Results saved to {output_dir}/")
    return results


def run_gridsearch(
    *,
    C_history: np.ndarray,
    distances: np.ndarray,
    alphas: list[float],
    betas: list[float],
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    solver: SolverBase,
    num_reads: int,
    num_sweeps: int | None = None,
    seed: int | None,
    output_dir: Path,
    solver_name: str,
    history_csv: str = "history.csv",
) -> tuple[float, float, float, float]:
    """
    (α, β) グリッドサーチを実行し、最良の (alpha, beta, loss, rmse) を返す。
    history.csv にキャッシュして途中再開可能。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / history_csv
    T, N = C_history.shape
    Np = N + 1
    rng = np.random.default_rng(seed)
    p_true = _safe_row_normalize(C_history)

    # キャッシュ読み込み
    cache: dict[tuple[float, float], dict] = {}
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    a, b = float(row["alpha"]), float(row["beta"])
                    cache[(a, b)] = {"loss": float(row["loss"]), "rmse": float(row["rmse"]),
                                     "violation_rate": float(row.get("violation_rate", 0))}
                except (ValueError, KeyError):
                    pass
        print(f"[gridsearch] Loaded {len(cache)} cached results.")

    # CSV ヘッダー
    fields = ["alpha", "beta", "loss", "rmse", "violation_rate", "timestamp"]
    write_header = not history_path.exists()
    history_file = open(history_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(history_file, fieldnames=fields)
    if write_header:
        writer.writeheader()

    best_alpha, best_beta, best_loss, best_rmse = None, None, float("inf"), float("inf")

    # キャッシュからベスト復元
    for (a, b), res in cache.items():
        if res["loss"] < best_loss:
            best_alpha, best_beta, best_loss, best_rmse = a, b, res["loss"], res["rmse"]

    total = len(alphas) * len(betas)
    step = 0
    for a in alphas:
        for b in betas:
            step += 1
            if (a, b) in cache:
                print(f"  [{step}/{total}] α={a:.2f} β={b:.2f} — skipped (cached)")
                continue

            P = build_transition_P(C_history, distances, a, b)
            Q = build_qubo(P, T, lambda_onehot, lambda_P, lambda_div, lambda_entry, lambda_move)

            cfg: dict = {"num_reads": num_reads}
            if num_sweeps is not None:
                cfg["num_sweeps"] = num_sweeps
            if seed is not None:
                cfg["seed"] = seed
            sampleset = solver.solve(Q, sample_config=cfg)

            traj_list = [decode_traj(dict(s), T, Np, rng) for s in sampleset.samples()]
            violation = compute_violation_rate(sampleset, T, Np)
            p_prime = reconstruct_proportions(traj_list, T, N)
            loss = squared_loss(p_true, p_prime)
            r = rmse(p_true, p_prime)

            if loss < best_loss:
                best_alpha, best_beta, best_loss, best_rmse = a, b, loss, r

            row = {"alpha": a, "beta": b, "loss": round(loss, 6), "rmse": round(r, 6),
                   "violation_rate": round(violation, 6),
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            writer.writerow(row)
            history_file.flush()

            print(f"  [{step}/{total}] α={a:.2f} β={b:.2f} → RMSE={r:.6f}  best=({best_alpha},{best_beta}) RMSE={best_rmse:.6f}")

    history_file.close()
    print(f"[gridsearch] Best: α={best_alpha}, β={best_beta}, RMSE={best_rmse:.6f}")
    return best_alpha, best_beta, best_loss, best_rmse
