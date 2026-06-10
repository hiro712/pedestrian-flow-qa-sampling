from pathlib import Path
from typing import Any

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dimod import SampleSet

from src.solvers.base import Qubo


def decode_traj(
    sample: dict[int, int],
    T_steps: int,
    Np: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    1サンプルを長さ T_steps の軌跡に変換する。
    - 同時刻で2以上 → ランダムに1つ選択
    - 同時刻で0個  → 外部ノード(0)を選択
    """
    def idx(i: int, t: int) -> int:
        return i * T_steps + t

    traj = []
    for t in range(T_steps):
        actives = [i for i in range(Np) if sample.get(idx(i, t), 0) == 1]
        if len(actives) == 1:
            traj.append(actives[0])
        elif len(actives) == 0:
            traj.append(0)
        else:
            traj.append(int(rng.choice(actives)))
    return traj


def compute_violation_rate(sampleset: SampleSet, T_steps: int, Np: int) -> float:
    """One-Hot 違反率（同時刻で2以上選択されたスロットの割合）。"""
    def idx(i: int, t: int) -> int:
        return i * T_steps + t

    violations = 0
    total = 0
    for sample in sampleset.samples():
        for t in range(T_steps):
            total += 1
            cnt = sum(sample.get(idx(i, t), 0) for i in range(Np))
            if cnt not in (0, 1):
                violations += 1
    return violations / total if total else 0.0


def qubo_energy(sample: dict[int, int], Q: Qubo) -> float:
    """正規化済み QUBO に対してエネルギーを再計算する。"""
    e = 0.0
    for (u, v), w in Q.items():
        e += w * sample.get(u, 0) * sample.get(v, 0)
    return float(e)


def save_results(
    sampleset: SampleSet,
    traj_list: list[list[int]],
    Q: Qubo,
    T_steps: int,
    Np: int,
    output_dir: Path,
) -> None:
    """
    sampleset.json / traj.csv / エネルギーヒストグラム を output_dir に保存する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # sampleset.json
    with open(output_dir / "sampleset.json", "w", encoding="utf-8") as f:
        json.dump(sampleset.to_serializable(), f, ensure_ascii=False, indent=2)

    # traj.csv
    with open(output_dir / "traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for traj in traj_list:
            writer.writerow(traj)

    # エネルギーヒストグラム (A) ハードウェア報告値
    hw_energies = sampleset.record.energy
    plt.figure()
    plt.hist(hw_energies, bins=50)
    plt.xlabel("Energy (hardware reported)")
    plt.ylabel("Frequency")
    plt.title("Sampling Energy Histogram (hardware)")
    plt.savefig(output_dir / "energy_histogram_hw.png")
    plt.close()

    # エネルギーヒストグラム (B) QUBO再計算 raw vs fixed
    rng = np.random.default_rng(0)

    def fix_onehot(sample: dict[int, int]) -> dict[int, int]:
        fixed = dict(sample)
        def _idx(i: int, t: int) -> int:
            return i * T_steps + t
        for t in range(T_steps):
            actives = [i for i in range(Np) if sample.get(_idx(i, t), 0) == 1]
            if len(actives) >= 2:
                keep = int(rng.choice(actives))
                for i in actives:
                    fixed[_idx(i, t)] = 1 if i == keep else 0
        return fixed

    raw_e = np.array([qubo_energy(dict(s), Q) for s in sampleset.samples()])
    fix_e = np.array([qubo_energy(fix_onehot(dict(s)), Q) for s in sampleset.samples()])

    vmin, vmax = float(np.concatenate([raw_e, fix_e]).min()), float(np.concatenate([raw_e, fix_e]).max())
    bins: Any = np.linspace(vmin, vmax, 50).tolist() if not np.isclose(vmin, vmax) else 50

    plt.figure()
    plt.hist(raw_e, bins=bins, alpha=0.6, label="Raw (recomputed)")
    plt.hist(fix_e, bins=bins, alpha=0.6, label="Fixed (random tie-break)")
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Raw vs Fixed")
    plt.legend()
    plt.savefig(output_dir / "energy_histogram_overlay.png")
    plt.close()

    plt.figure()
    plt.hist(fix_e, bins=50, alpha=0.8)
    plt.xlabel("Energy (recomputed on normalized QUBO)")
    plt.ylabel("Frequency")
    plt.title("Energy Histogram: Fixed (random tie-break)")
    plt.savefig(output_dir / "energy_histogram_fixed.png")
    plt.close()
