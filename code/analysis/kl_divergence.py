"""M10: structure-aware comparison of the SQA and QA flow matrices.

There is no ground-truth joint edge-usage histogram (only marginal headcounts
C_{t,i} were observed), so we cannot compute KL(F_obs || F_pred). Instead we
quantify how different the two *predicted* flow structures are from each
other, directly supporting the claim that "RMSE on marginals is close but the
underlying joint structure differs substantially": we treat the (flattened,
renormalized) flow matrices F_sqa and F_qa as discrete probability
distributions over directed edges and report the KL divergence in both
directions plus the (symmetric, bounded) Jensen-Shannon divergence.

Usage:
    uv run python3 analysis/kl_divergence.py
"""
import json
import numpy as np
from scipy.stats import entropy

EPS = 1e-12


def to_dist(F):
    F = np.asarray(F, dtype=float)
    F = np.clip(F, 0.0, None)
    flat = F.flatten()
    flat = flat + EPS
    return flat / flat.sum()


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def main():
    sqa = json.load(open("results/sqa_30k/results.json"))
    qa = json.load(open("results/qa_advantage2_30k/results.json"))

    p_sqa = to_dist(sqa["F"])
    p_qa = to_dist(qa["F"])

    kl_sqa_qa = entropy(p_sqa, p_qa)
    kl_qa_sqa = entropy(p_qa, p_sqa)
    js = js_divergence(p_sqa, p_qa)

    print("Structure-aware comparison of predicted edge-usage distributions (F)")
    print(f"  KL(F_SQA || F_QA) = {kl_sqa_qa:.4f} nats")
    print(f"  KL(F_QA || F_SQA) = {kl_qa_sqa:.4f} nats")
    print(f"  Jensen-Shannon divergence (symmetric, bounded by ln 2 = {np.log(2):.4f}) = {js:.4f} nats")
    print(f"  RMSE(p') for reference: SQA = {sqa['rmse']:.6f}, QA = {qa['rmse']:.6f}")

    out = {
        "kl_sqa_qa_nats": float(kl_sqa_qa),
        "kl_qa_sqa_nats": float(kl_qa_sqa),
        "js_divergence_nats": float(js),
        "js_divergence_bound_nats": float(np.log(2)),
        "rmse_sqa": float(sqa["rmse"]),
        "rmse_qa": float(qa["rmse"]),
    }
    with open("results/kl_divergence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/kl_divergence.json")


if __name__ == "__main__":
    main()
