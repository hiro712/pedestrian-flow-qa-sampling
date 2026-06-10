from typing import Optional, Tuple, Dict, Any, Callable, List
import numpy as np
import openjij as oj
import dimod
import csv
import matplotlib
import json
import hashlib
import os
from datetime import datetime

from solve.main import QASolver, SASolver, SQASolver

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================
def _arr_hash(a: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(a.tobytes())
    return h.hexdigest()


def _safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """行方向に正規化（行和0ならその行は0のまま）。"""
    m = mat.astype(float)
    rowsum = m.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(m, np.where(rowsum > eps, rowsum, 1.0))
        out = np.where(rowsum > eps, out, 0.0)
    return out


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


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
    log_txt: str = "progress.log",
    history_csv: str = "history.csv",
) -> None:
    _ensure_parent_dir(log_txt)
    _ensure_parent_dir(history_csv)
    # JSON Lines
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    # CSV
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
    new_file = not os.path.exists(history_csv)
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        row = {k: info.get(k, "") for k in fields}
        w.writerow(row)


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
# QUBO & Sampling（軌跡長は T = C_history.shape[0]）
# =========================
def _sample_trajectories(
    P: np.ndarray,
    T_steps: int,
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
) -> Tuple[List[List[int]], float]:
    """
    QUBO を構築し OpenJij でサンプリング。長さ T_steps の軌跡と One-Hot 違反率を返す。
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

    # サンプリング
    # bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    # sampler = oj.SASampler()
    # sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed, num_sweeps=100)

    # solver = SASolver()
    # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
    # if seed is not None:
    #     sample_config["seed"] = seed
    # sampleset = solver.solve(Q, sample_config=sample_config)

    # solver = SQASolver()
    # sample_config: Dict[str, Any] = {"num_reads": num_reads, "num_sweeps": 100}
    # if seed is not None:
    #     sample_config["seed"] = seed
    # sampleset = solver.solve(Q, sample_config=sample_config)

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")  # .env.local から環境変数を読み込む
    print("DWAVE_SOLVER_NAME:", os.getenv("DWAVE_SOLVER_NAME"))
    solver = QASolver()
    sample_config: Dict[str, Any] = {
        "num_reads": num_reads,
        "annealing_time": 20,
    }
    if seed is not None:
        sample_config["seed"] = seed
    sampleset = solver.solve(Q, sample_config=sample_config)

    # ---------- ここから追加：乱択デコーダ／One-Hot修正／エネルギー再計算用ヘルパ ----------
    # 乱択用RNG（seed指定で再現可能）
    rng = np.random.default_rng(seed)

    def _decode_traj(sample: Dict[int, int]) -> List[int]:
        """同時刻に1が複数ならランダム、0個なら外部(0)"""
        traj = []
        for t in range(T_steps):
            actives = [i for i in range(Np) if sample.get(idx(i, t), 0) == 1]
            if len(actives) == 1:
                u = actives[0]
            elif len(actives) == 0:
                u = 0
            else:
                u = int(rng.choice(actives))
            traj.append(u)
        return traj

    def _fix_sample_onehot(sample: Dict[int, int]) -> Dict[int, int]:
        """One-Hot違反(=同時刻で2個以上1)の箇所だけ乱択で1つ残し、他は0に直す。
        0個 or 1個は変更しない（=違反ではない）。"""
        fixed = dict(sample)
        for t in range(T_steps):
            actives = [i for i in range(Np) if sample.get(idx(i, t), 0) == 1]
            if len(actives) >= 2:
                keep = int(rng.choice(actives))
                for i in actives:
                    fixed[idx(i, t)] = 1 if i == keep else 0
        return fixed

    def _qubo_energy(sample: Dict[int, int]) -> float:
        """Q（正規化後QUBO）に対するエネルギーを再計算。"""
        e = 0.0
        for (u, v), w in Q.items():
            e += w * sample.get(u, 0) * sample.get(v, 0)
        return float(e)

    # ---------- 追加ここまで -----------------------------------------------------

    # samplesetをJSON保存（to_serializable()で全データを保存）
    sampleset_data = sampleset.to_serializable()
    with open("sampleset.json", "w", encoding="utf-8") as f:
        json.dump(sampleset_data, f, ensure_ascii=False, indent=2)

    # 軌跡CSV保存（乱択タイブレーク/空欄は外部0）
    with open("traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for sample in sampleset.samples():
            traj = _decode_traj(dict(sample))
            writer.writerow(traj)

    # エネルギー分布
    # (A) QAハードウェアが返すエネルギー（参考） + 互換用に従来ファイル名も保存
    hw_energies = sampleset.record.energy
    plt.figure()
    plt.hist(hw_energies, bins=50)
    plt.xlabel("Energy (hardware reported)")
    plt.ylabel("Frequency")
    plt.title("Sampling Energy Histogram (hardware)")
    plt.savefig("energy_histogram_hw.png")
    plt.savefig("energy_histogram.png")  # 互換維持：従来ファイル名
    plt.close()

    # (B) 正規化後QUBO Q に対して「元サンプル」と「One-Hot修正サンプル」を再計算し重ね描き
    raw_energies_recalc = np.array(
        [_qubo_energy(dict(sample)) for sample in sampleset.samples()], dtype=float
    )
    fixed_energies_recalc = np.array(
        [
            _qubo_energy(_fix_sample_onehot(dict(sample)))
            for sample in sampleset.samples()
        ],
        dtype=float,
    )

    all_vals = np.concatenate([raw_energies_recalc, fixed_energies_recalc])
    # 同一値しかないケースにも対応
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    bins_arg: Any
    if np.isclose(vmin, vmax):
        bins_arg = 50
    else:
        bins_arg = np.linspace(vmin, vmax, 50).tolist()

    plt.figure()
    plt.hist(raw_energies_recalc, bins=bins_arg, alpha=0.6, label="Raw (recomputed)")
    plt.hist(
        fixed_energies_recalc,
        bins=bins_arg,
        alpha=0.6,
        label="Fixed (random tie-break; recomputed)",
    )
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Raw vs Fixed (recomputed)")
    plt.legend()
    plt.savefig("energy_histogram_overlay.png")
    plt.close()

    # (C) Fixedだけのヒストグラム
    plt.figure()
    plt.hist(fixed_energies_recalc, bins=50, alpha=0.8)
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Fixed (random tie-break; recomputed)")
    plt.savefig("energy_histogram_fixed.png")
    plt.close()

    # One-Hot違反率（0個 or 1個はOK、2個以上は違反）
    violations = 0
    total_slots = 0
    for sample in sampleset.samples():
        for t in range(T_steps):
            total_slots += 1
            cnt = sum(sample.get(idx(i, t), 0) for i in range(Np))
            if cnt not in (0, 1):
                violations += 1
    violation_rate = violations / total_slots if total_slots else 0.0

    # 軌跡リスト（CSVと同じ乱択デコード）
    traj_list: List[List[int]] = []
    for sample in sampleset.samples():
        traj_list.append(_decode_traj(dict(sample)))

    return traj_list, violation_rate


# =========================
# Proportions p' reconstruction from trajectories
# =========================
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
# Grid runner（p vs p' 最小化）
# =========================
def _load_history_cache(
    history_csv: str, stage_name: str
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """
    history.csvから既存データを読み込み、(alpha, beta) -> {loss, rmse, ...} の辞書を返す。
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


def _run_grid(
    stage_name: str,
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
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    指定グリッドで p と p'（行正規化した割合）の二乗和 LOSS と RMSE を最小化。
    history.csvに既存データがあれば、それを使ってスキップする。
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    T, N = C_history.shape
    T_steps = T
    p_true = _safe_row_normalize(C_history)  # 行正規化（割合）

    # 既存データを読み込む
    cached_results = _load_history_cache(history_csv, stage_name)
    print(
        f"[{stage_name}] Loaded {len(cached_results)} cached results from {history_csv}"
    )

    total = len(alphas) * len(betas)
    step = 0
    best = {"alpha": None, "beta": None, "loss": float("inf"), "rmse": float("inf")}
    history = []

    # キャッシュからベスト値を復元
    for (a, b), result in cached_results.items():
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

    for a in alphas:
        for b in betas:
            step += 1

            # キャッシュにあればスキップ
            if (a, b) in cached_results:
                result = cached_results[(a, b)]
                print(f"[{stage_name}] Skipping {step}/{total} α={a}, β={b} (cached)")
                continue
            # 1) P
            P = _build_transition_P(C_history, distances, a, b)
            # 2) サンプリング → 軌跡
            traj_list, viol = _sample_trajectories(
                P,
                T_steps,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
                seed,
                num_reads,
            )
            # 3) p' を復元
            p_prime = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
            # 4) 損失（p vs p' の二乗和）＆ RMSE
            diff = p_true - p_prime
            loss = float(np.sum(diff**2))
            rmse = float(np.sqrt(loss / diff.size))

            # 進捗通知 & ログ
            info = {
                "when": _now(),
                "stage": stage_name,
                "step": step,
                "total": total,
                "alpha": float(a),
                "beta": float(b),
                "loss": round(loss, 6),
                "rmse": round(rmse, 6),
                "violation_rate": round(viol, 6),
                "best_alpha": best["alpha"],
                "best_beta": best["beta"],
                "best_loss": best["loss"],
                "best_rmse": best["rmse"],
            }
            progress_cb(info)
            _log_progress_files(info)

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

    return best["alpha"], best["beta"], best["loss"], best["rmse"], {"history": history}


# =========================
# 2段階探索：Stage1(広く粗く) → Stage2(局所で細かく)
# =========================
def optimize_alpha_beta_two_stage(
    C_history: np.ndarray,
    distances: np.ndarray,  # 正規化済み前提
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    # Stage1（広く粗く）
    alphas_stage1: Optional[List[float]] = None,
    betas_stage1: Optional[List[float]] = None,
    # Stage2（局所で細かく）±幅と刻み
    stage2_alpha_window: float = 0.6,
    stage2_alpha_step: float = 0.1,
    stage2_beta_window: float = 0.6,
    stage2_beta_step: float = 0.05,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    cache_path: str = "alpha_beta_cache.json",
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """
    自動2段階探索で (α,β) を最適化（p と p′ の LOSS/RMSE 最小化）。
    戻り値: (alpha*, beta*, best_loss, best_rmse, meta)
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    # Stage1 デフォルト（広く粗く）— numpy.float64 → float にして型整合
    if alphas_stage1 is None:
        alphas_stage1 = list(map(float, np.arange(0.0, 6.0 + 1e-9, 0.5)))
    if betas_stage1 is None:
        betas_stage1 = list(map(float, np.arange(0.3, 2.5 + 1e-9, 0.2)))

    # --- Stage 1 ---
    a1, b1, loss1, rmse1, meta1 = _run_grid(
        stage_name="stage1",
        C_history=C_history,
        distances=distances,
        lambda_onehot=lambda_onehot,
        lambda_P=lambda_P,
        lambda_div=lambda_div,
        lambda_entry=lambda_entry,
        lambda_move=lambda_move,
        seed=seed,
        num_reads=num_reads,
        alphas=alphas_stage1,
        betas=betas_stage1,
        progress_cb=progress_cb,
        history_csv="history.csv",
    )

    # Stage2 グリッド（ベスト周辺を細かく）
    a_lo = max(0.0, a1 - stage2_alpha_window)
    a_hi = a1 + stage2_alpha_window
    b_lo = max(0.1, b1 - stage2_beta_window)
    b_hi = b1 + stage2_beta_window
    alphas_stage2 = list(map(float, np.arange(a_lo, a_hi + 1e-9, stage2_alpha_step)))
    betas_stage2 = list(map(float, np.arange(b_lo, b_hi + 1e-9, stage2_beta_step)))

    # --- Stage 2 ---
    a2, b2, loss2, rmse2, meta2 = _run_grid(
        stage_name="stage2",
        C_history=C_history,
        distances=distances,
        lambda_onehot=lambda_onehot,
        lambda_P=lambda_P,
        lambda_div=lambda_div,
        lambda_entry=lambda_entry,
        lambda_move=lambda_move,
        seed=seed,
        num_reads=num_reads,
        alphas=alphas_stage2,
        betas=betas_stage2,
        progress_cb=progress_cb,
        history_csv="history.csv",
    )

    # キャッシュ保存（C,D,探索条件付き）
    key = {
        "C_hash": _arr_hash(C_history),
        "D_hash": _arr_hash(distances),
        "num_reads": num_reads,
        "seed": seed,
        "stage2_window": {
            "alpha": stage2_alpha_window,
            "beta": stage2_beta_window,
            "alpha_step": stage2_alpha_step,
            "beta_step": stage2_beta_step,
        },
    }
    key_str = json.dumps(key, sort_keys=True)

    cache: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    best = {"alpha": a2, "beta": b2, "loss": loss2, "rmse": rmse2}
    cache[key_str] = {
        "key": key,
        "best": best,
        "stage1": {
            "alphas": alphas_stage1,
            "betas": betas_stage1,
            "best_alpha": a1,
            "best_beta": b1,
            "best_loss": loss1,
            "best_rmse": rmse1,
            "history": meta1.get("history", []),
        },
        "stage2": {
            "alphas": alphas_stage2,
            "betas": betas_stage2,
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
            "history": meta2.get("history", []),
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 完了通知
    progress_cb(
        {
            "when": _now(),
            "stage": "done-two-stage",
            "step": "-",
            "total": "-",
            "alpha": a2,
            "beta": b2,
            "loss": loss2,
            "rmse": rmse2,
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
            "violation_rate": "-",
        }
    )

    meta = {
        "stage1": {
            "best_alpha": a1,
            "best_beta": b1,
            "best_loss": loss1,
            "best_rmse": rmse1,
        },
        "stage2": {
            "best_alpha": a2,
            "best_beta": b2,
            "best_loss": loss2,
            "best_rmse": rmse2,
        },
    }
    return a2, b2, loss2, rmse2, meta


# =========================
# Main prediction（αβ未指定→自動2段階探索）
# =========================
def predict_flow_percent(
    C_history: np.ndarray,  # (T,N)
    distances: np.ndarray,  # (N+1,N+1) 正規化済み
    lambda_onehot: float,
    lambda_P: float,
    lambda_div: float,
    lambda_entry: float,
    lambda_move: float,
    seed: Optional[int],
    num_reads: int,
    # α,β を指定可能（未指定なら自動2段階探索）
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    # Stage1/Stage2 のパラメータ（必要なら上書き）
    alphas_stage1: Optional[List[float]] = None,
    betas_stage1: Optional[List[float]] = None,
    stage2_alpha_window: float = 0.6,
    stage2_alpha_step: float = 0.1,
    stage2_beta_window: float = 0.6,
    stage2_beta_step: float = 0.05,
    # 進捗コールバック
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    # キャッシュファイル
    cache_path: str = "alpha_beta_cache.json",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], Dict[str, Any]]:
    """
    量子アニーリングで個人経路をサンプリングし、P・F・p′ を返す。
    α,β 未指定時は「広く粗く→局所で細かく」の自動2段階探索で p と p′ のズレ最小化。
    戻り値: (P, F, p_prime, (alpha,beta), meta)
    """
    if progress_cb is None:
        progress_cb = _default_progress_cb

    T, N = C_history.shape
    T_steps = T

    # 行正規化した真の割合 p_true を常に計算しておく
    p_true = _safe_row_normalize(C_history)

    best_loss = None
    best_rmse = None

    if alpha is None or beta is None:
        alpha, beta, best_loss, best_rmse, two_stage_meta = (
            optimize_alpha_beta_two_stage(
                C_history,
                distances,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
                seed,
                num_reads,
                alphas_stage1=alphas_stage1,
                betas_stage1=betas_stage1,
                stage2_alpha_window=stage2_alpha_window,
                stage2_alpha_step=stage2_alpha_step,
                stage2_beta_window=stage2_beta_window,
                stage2_beta_step=stage2_beta_step,
                progress_cb=progress_cb,
                cache_path=cache_path,
            )
        )
        progress_cb(
            {
                "when": _now(),
                "stage": "select",
                "step": "-",
                "total": "-",
                "alpha": alpha,
                "beta": beta,
                "loss": best_loss,
                "rmse": best_rmse,
                "best_alpha": alpha,
                "best_beta": beta,
                "best_loss": best_loss,
                "best_rmse": best_rmse,
                "violation_rate": "-",
            }
        )
    else:
        two_stage_meta = {}

    # 最終 P
    P = _build_transition_P(C_history, distances, alpha, beta)

    # サンプリング → F と p′
    traj_list, violation_rate = _sample_trajectories(
        P,
        T_steps,
        lambda_onehot,
        lambda_P,
        lambda_div,
        lambda_entry,
        lambda_move,
        seed,
        num_reads,
    )
    # p' を再構成して損失を計算
    p_prime_tmp = _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
    diff_tmp = p_true - p_prime_tmp
    loss_final = float(np.sum(diff_tmp**2))
    rmse_final = float(np.sqrt(loss_final / diff_tmp.size))

    progress_cb(
        {
            "when": _now(),
            "stage": "sample_final",
            "step": "-",
            "total": "-",
            "alpha": alpha,
            "beta": beta,
            "loss": round(loss_final, 6),
            "rmse": round(rmse_final, 6),
            "best_alpha": alpha,
            "best_beta": beta,
            "best_loss": best_loss if best_loss is not None else round(loss_final, 6),
            "best_rmse": best_rmse if best_rmse is not None else round(rmse_final, 6),
            "violation_rate": violation_rate,
        }
    )

    # フロー F（u->v の頻度を行正規化）
    Np = P.shape[0]
    F_counts = np.zeros((Np, Np))
    for traj in traj_list:
        for t in range(T_steps - 1):
            u = traj[t]
            v = traj[t + 1]
            F_counts[u, v] += 1
    F = _safe_row_normalize(F_counts)

    # p′ 復元（割合）
    # 先に計算した p_prime_tmp があるはずなので再利用
    p_prime = (
        _reconstruct_proportions_from_trajs(traj_list, T_steps, N)
        if "p_prime_tmp" not in locals()
        else p_prime_tmp
    )

    # 保存
    results = {
        "C": C_history.tolist(),
        "p_true": _safe_row_normalize(C_history).tolist(),
        "p_prime": p_prime.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "alpha": alpha,
        "beta": beta,
        # best_* は探索で得た最良値。未探索（ユーザ指定）の場合は今回の値を入れる
        "best_loss": (
            best_loss if best_loss is not None else float(round(loss_final, 6))
        ),
        "best_rmse": (
            best_rmse if best_rmse is not None else float(round(rmse_final, 6))
        ),
        "loss": float(round(loss_final, 6)),
        "rmse": float(round(rmse_final, 6)),
        "meta": two_stage_meta,
    }
    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=4)

    meta = {
        "violation_rate": violation_rate,
        "best_loss": best_loss,
        "best_rmse": best_rmse,
        **two_stage_meta,
    }
    return P, F, p_prime, (alpha, beta), meta
