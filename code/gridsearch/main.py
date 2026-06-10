# filename: grid_search_alpha_beta.py
from typing import Optional, Tuple, Dict, Any, Callable, List
import numpy as np
import openjij as oj
import dimod
import csv
import json
import os
from datetime import datetime

from solve.main import (
    QASolver,
    SASolver,
    SQASolver,
)  # サンプラー（コメントで切替例を残します）


# =========================
# Utilities
# =========================
def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """行方向に正規化（行和0ならその行は0のまま）。"""
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(m, np.where(rowsum > eps, rowsum, 1.0))
        out = np.where(rowsum > eps, out, 0.0)
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_progress_cb(info: Dict[str, Any]) -> None:
    msg = (
        f"[{info.get('when')}] {info.get('stage','?')} "
        f"{info.get('step','')}/{info.get('total','')} "
        f"α={info.get('alpha','-')}, β={info.get('beta','-')}, "
        f"loss={info.get('loss','-')}, rmse={info.get('rmse','-')}, "
        f"best=({info.get('best_alpha','-')},{info.get('best_beta','-')} "
        f"L={info.get('best_loss','-')},R={info.get('best_rmse','-')}) "
        f"viol={info.get('violation_rate','-')}"
    )
    print(msg)


def _log_progress_files(
    info: Dict[str, Any],
    history_csv: str = "history.csv",
) -> None:
    # CSVのみ出力
    fields = [
        "when",
        "stage",
        "step",
        "total",
        "alpha",
        "beta",
        "loss",
        "rmse",
        "best_alpha",
        "best_beta",
        "best_loss",
        "best_rmse",
        "violation_rate",
    ]
    os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
    new_file = not os.path.exists(history_csv)
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        row = {k: info.get(k, "") for k in fields}
        w.writerow(row)


def _load_history_cache(
    history_csv: str, stage_name: str
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """
    history.csvから既存データを読み込み、(alpha, beta) -> {loss, rmse, violation_rate} の辞書を返す。
    """
    cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
    if not os.path.exists(history_csv):
        return cache

    try:
        with open(history_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") == stage_name:
                    try:
                        alpha = float(row["alpha"])
                        beta = float(row["beta"])
                        loss = float(row["loss"])
                        rmse = float(row["rmse"])
                        violation_rate = float(row.get("violation_rate", 0.0))
                        cache[(alpha, beta)] = {
                            "loss": loss,
                            "rmse": rmse,
                            "violation_rate": violation_rate,
                        }
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print(f"Warning: Failed to load history cache: {e}")

    return cache


# =========================
# Transition matrix builder（距離は正規化済み前提）
# =========================
def _build_transition_P(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) 正規化済み
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    直近期の割合 p_prev と距離・人気で重み W を作り、行正規化して P を返す（外部=0）。
    """
    T, N = C_history.shape
    Np = N + 1
    if distances.shape != (Np, Np):
        raise ValueError(f"distances shape must be {(Np, Np)}, got {distances.shape}")

    p_hist = _safe_row_normalize(C_history)

    prev_sum = C_history[-2].sum() if T >= 2 else C_history[-1].sum()
    cur_sum = C_history[-1].sum()
    denom = prev_sum if prev_sum > 0 else 1.0
    delta_plus = max(cur_sum - prev_sum, 0.0) / denom
    prev_idx = -2 if T >= 2 else -1
    p_prev = np.concatenate(([delta_plus], p_hist[prev_idx] * (1.0 - delta_plus)))

    # α・β 適用
    W = np.exp(-alpha * distances) * (p_prev[None, :] ** beta)
    P = _safe_row_normalize(W)
    return P


# =========================
# QUBO 構築
# =========================
def _build_qubo_from_P(
    P: np.ndarray,
    T_steps: int,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
) -> Tuple[Dict[Tuple[int, int], float], int]:
    """
    P から QUBO（正規化済み）を作る。戻り値: (Q, Np)
    """
    Np = P.shape[0]
    Q: Dict[Tuple[int, int], float] = {}

    def idx(i: int, t: int) -> int:
        return i * T_steps + t

    def apply_normalized(base_dict: dict, lam: float):
        if not base_dict:
            return
        max_abs = max(abs(v) for v in base_dict.values()) or 1.0
        for key, raw_val in base_dict.items():
            Q[key] = Q.get(key, 0.0) + lam * (raw_val / max_abs)

    # One-Hot（同時刻で複数選択を罰。0個or1個は許容）
    base_onehot: Dict[Tuple[int, int], float] = {}
    for t in range(T_steps):
        for i in range(Np):
            for j in range(i + 1, Np):
                base_onehot[(idx(i, t), idx(j, t))] = (
                    base_onehot.get((idx(i, t), idx(j, t)), 0.0) + 2.0
                )
    apply_normalized(base_onehot, lambda_onehot)

    # P 嗜好（i at t-1 → j at t に -P[i,j]）
    base_P: Dict[Tuple[int, int], float] = {}
    for t in range(1, T_steps):
        for i in range(Np):
            for j in range(Np):
                base_P[(idx(i, t - 1), idx(j, t))] = (
                    base_P.get((idx(i, t - 1), idx(j, t)), 0.0) - P[i, j]
                )
    apply_normalized(base_P, lambda_P)

    # 分散（同一エリアに偏らない）
    avg_visits = T_steps / Np
    base_div: Dict[Tuple[int, int], float] = {}
    for i in range(Np):
        for t in range(T_steps):
            ii = idx(i, t)
            base_div[(ii, ii)] = base_div.get((ii, ii), 0.0) + (1 - 2 * avg_visits)
            for t2 in range(t + 1, T_steps):
                jj = idx(i, t2)
                base_div[(ii, jj)] = base_div.get((ii, jj), 0.0) + 2.0
    apply_normalized(base_div, lambda_div)

    # 外部(0)の出入り抑制
    base_entry: Dict[Tuple[int, int], float] = {}
    for t in range(T_steps - 1):
        ii = idx(0, t)
        jj = idx(0, t + 1)
        base_entry[(ii, ii)] = base_entry.get((ii, ii), 0.0) + 1.0
        base_entry[(jj, jj)] = base_entry.get((jj, jj), 0.0) + 1.0
        base_entry[(ii, jj)] = base_entry.get((ii, jj), 0.0) - 2.0
    apply_normalized(base_entry, lambda_entry)

    # 内部移動の平滑化
    base_move: Dict[Tuple[int, int], float] = {}
    for i in range(1, Np):
        for t in range(T_steps - 1):
            ii = idx(i, t)
            jj = idx(i, t + 1)
            base_move[(ii, ii)] = base_move.get((ii, ii), 0.0) + 1.0
            base_move[(jj, jj)] = base_move.get((jj, jj), 0.0) + 1.0
            base_move[(ii, jj)] = base_move.get((ii, jj), 0.0) - 2.0
    apply_normalized(base_move, lambda_move)

    return Q, Np


# =========================
# サンプル補正・復元
# =========================
def _fix_sample_onehot(
    sample: Dict[int, int],
    Np: int,
    T_steps: int,
    rng: np.random.Generator,
) -> Dict[int, int]:
    """
    One-Hot違反(=同時刻で2個以上1)の箇所だけ乱択で1つ残し、他は0に直す。
    0個 or 1個は変更しない（=違反ではない）。
    """
    fixed = dict(sample)
    for t in range(T_steps):
        actives = [i for i in range(Np) if sample.get(i * T_steps + t, 0) == 1]
        if len(actives) >= 2:
            keep = int(rng.choice(actives))
            for i in actives:
                fixed[i * T_steps + t] = 1 if i == keep else 0
    return fixed


def _decode_traj_from_sample(
    sample: Dict[int, int], Np: int, T_steps: int
) -> List[int]:
    """
    サンプル（One-Hot満たしている前提）を軌跡リストに変換。
    0個なら外部(0)。
    """
    traj = []
    for t in range(T_steps):
        actives = [i for i in range(Np) if sample.get(i * T_steps + t, 0) == 1]
        if len(actives) == 1:
            u = actives[0]
        elif len(actives) == 0:
            u = 0
        else:
            # ここには基本来ない（事前に_fix_sample_onehotで解消）
            u = actives[0]
        traj.append(u)
    return traj


def _reconstruct_proportions_from_trajs(
    traj_list: List[List[int]],
    T_steps: int,
    N: int,  # 内部エリア数（外部を除く）
) -> np.ndarray:
    """
    サンプリング軌跡から各時刻の内部エリア割合 p' (T×N) を復元。
    """
    num_samples = len(traj_list)
    if num_samples == 0:
        return np.zeros((T_steps, N), dtype=float)

    counts = np.zeros((T_steps, N), dtype=float)
    for traj in traj_list:
        for t in range(T_steps):
            u = traj[t]
            if 1 <= u <= N:
                counts[t, u - 1] += 1.0  # 内部は1..N → 0..N-1

    p_prime = counts / float(num_samples)
    return p_prime


# =========================
# グリッドサーチ本体（history.csvのみ出力）
# =========================
def grid_search_alpha_beta(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) 正規化済み
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    alphas: List[float],
    betas: List[float],
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    history_csv: str = "history.csv",
    stage_name: str = "grid",
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    (α,β) のグリッドで p と p' のズレ（LOSS/RMSE）を最小化。
    出力は history.csv のみ（エネルギー図等は出力しない）。
    RMSE計算時は、制約違反サンプルを乱択でOne-Hotに修正してから p' を算出。

    戻り値: (alpha*, beta*, best_loss, best_rmse, meta={'history':[...]})
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")

    # 事前計算
    T, N = C_history.shape
    T_steps = T
    p_true = _safe_row_normalize(C_history)  # 行正規化（割合）

    # キャッシュ読込（同じstageの既存結果はスキップ）
    cached = _load_history_cache(history_csv, stage_name)

    total = len(alphas) * len(betas)
    step = 0
    best = {"alpha": None, "beta": None, "loss": float("inf"), "rmse": float("inf")}
    history: List[Dict[str, Any]] = []

    # 事前にベスト（キャッシュ内）を復元
    for (a, b), result in cached.items():
        loss = result["loss"]
        rmse = result["rmse"]
        viol = result["violation_rate"]

        history.append(
            {
                "alpha": float(a),
                "beta": float(b),
                "loss": loss,
                "rmse": rmse,
                "onehot_violation_rate": float(viol),
            }
        )
        if loss < best["loss"]:
            best.update(
                {"alpha": float(a), "beta": float(b), "loss": loss, "rmse": rmse}
            )

    # 乱択用RNG（seed指定で再現可能）
    rng = np.random.default_rng(seed)

    for a in alphas:
        for b in betas:
            step += 1

            # 既に同じstage_nameで評価済みならスキップ
            if (a, b) in cached:
                progress_cb(
                    {
                        "when": _now(),
                        "stage": stage_name,
                        "step": step,
                        "total": total,
                        "alpha": float(a),
                        "beta": float(b),
                        "loss": cached[(a, b)]["loss"],
                        "rmse": cached[(a, b)]["rmse"],
                        "best_alpha": best["alpha"],
                        "best_beta": best["beta"],
                        "best_loss": best["loss"],
                        "best_rmse": best["rmse"],
                        "violation_rate": cached[(a, b)]["violation_rate"],
                    }
                )
                continue

            # 1) P 構築
            P = _build_transition_P(C_history, distances, a, b)

            # 2) QUBO 構築
            Q, Np = _build_qubo_from_P(
                P,
                T_steps,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
            )

            # 3) サンプリング
            # bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            # sampler = oj.SASampler()
            # sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed, num_sweeps=100)

            solver = SASolver()
            sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
            if seed is not None:
                sample_config["seed"] = seed
            sampleset = solver.solve(Q, sample_config=sample_config)

            # solver = SQASolver()
            # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
            # if seed is not None:
            #     sample_config["seed"] = seed
            # sampleset = solver.solve(Q, sample_config=sample_config)

            # solver = QASolver()
            # sample_config: Dict[str, Any] = {
            #     "num_reads": num_reads,
            #     "annealing_time": 20,
            # }
            # if seed is not None:
            #     sample_config["seed"] = seed
            # sampleset = solver.solve(Q, sample_config=sample_config)

            # 4) 違反率（生サンプルで判定：0個/1個はOK, 2個以上は違反）
            violations = 0
            total_slots = 0
            for sample in sampleset.samples():
                for t in range(T_steps):
                    total_slots += 1
                    cnt = sum(sample.get(i * T_steps + t, 0) for i in range(Np))
                    if cnt not in (0, 1):
                        violations += 1
            violation_rate = violations / total_slots if total_slots else 0.0

            # 5) 制約を満たすよう修正 → 軌跡化 → p' 復元
            traj_list: List[List[int]] = []
            for sample in sampleset.samples():
                fixed = _fix_sample_onehot(dict(sample), Np, T_steps, rng)
                traj = _decode_traj_from_sample(fixed, Np, T_steps)
                traj_list.append(traj)

            p_prime = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)

            # 6) 損失（p vs p' の二乗和）＆ RMSE
            diff = p_true - p_prime
            loss = float(np.sum(diff**2))
            rmse = float(np.sqrt(loss / diff.size))

            # 7) 進捗通知 & history.csv 追記
            info = {
                "when": _now(),
                "stage": stage_name,
                "step": step,
                "total": total,
                "alpha": float(a),
                "beta": float(b),
                "loss": round(loss, 6),
                "rmse": round(rmse, 6),
                "violation_rate": round(violation_rate, 6),
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_loss": best["loss"],
                "best_rmse": best["rmse"],
            }
            progress_cb(info)
            _log_progress_files(info, history_csv=history_csv)

            history.append(
                {
                    "alpha": float(a),
                    "beta": float(b),
                    "loss": loss,
                    "rmse": rmse,
                    "onehot_violation_rate": float(violation_rate),
                }
            )
            if loss < best["loss"]:
                best.update(
                    {"alpha": float(a), "beta": float(b), "loss": loss, "rmse": rmse}
                )

    meta = {"history": history}
    return best["alpha"], best["beta"], best["loss"], best["rmse"], meta
