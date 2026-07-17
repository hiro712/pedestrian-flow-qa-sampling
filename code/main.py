import json
import numpy as np
from pathlib import Path

from predict.main import predict_flow_percent
from distance.main import get_distances
from gridsearch.main import grid_search_alpha_beta


def main() -> None:
    # load distance matrix
    distances, _ = get_distances()
    dist_array = np.array(distances, dtype=float)

    # load C_array from research/data.json
    data_path = Path(__file__).resolve().parent / "data.json"
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read data.json at {data_path}: {e}")
        return

    # assume data contains a key compatible with C_array structure; if it's a list or nested lists, convert directly
    try:
        C_array = np.array(data, dtype=float)
    except Exception:
        # fallback: if data contains a mapping with a specific key, try common keys
        if isinstance(data, dict):
            for key in ("C", "C_array", "cost", "matrix"):
                if key in data:
                    C_array = np.array(data[key], dtype=float)
                    break
            else:
                print("data.json format not recognized for C_array conversion")
                return
        else:
            print("data.json format not recognized for C_array conversion")
            return

    lambda_onehot = 13.0
    lambda_P = 5.0
    lambda_div = 1.0
    lambda_entry = 2.0
    lambda_move = 0.5
    seed = None
    num_reads = 100

    alpha = 0.3
    beta = 0.55
    # alpha = 3.5
    # beta = 1.65
    # alpha = 0
    # beta = 0.3
    # alpha = None
    # beta = None

    try:
        if alpha is None or beta is None:  # run grid search
            alphas_stage1 = [float(x) for x in np.arange(0.0, 6.0 + 1e-9, 0.5)]
            betas_stage1 = [float(x) for x in np.arange(0.3, 2.5 + 1e-9, 0.2)]

            a1, b1, loss1, rmse1, meta1 = grid_search_alpha_beta(
                C_history=C_array,
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
                history_csv="history.csv",
                stage_name="stage1",
            )
            print(f"[Stage1] best α={a1}, β={b1}, RMSE={rmse1:.6f}")

            a_win, a_step = 0.6, 0.1
            b_win, b_step = 0.6, 0.05

            alphas_stage2 = [
                float(x)
                for x in np.arange(max(0.0, a1 - a_win), a1 + a_win + 1e-9, a_step)
            ]
            betas_stage2 = [
                float(x)
                for x in np.arange(max(0.1, b1 - b_win), b1 + b_win + 1e-9, b_step)
            ]

            a2, b2, loss2, rmse2, meta2 = grid_search_alpha_beta(
                C_history=C_array,
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
                history_csv="history.csv",
                stage_name="stage2",
            )

            print(f"[Stage2] best α={a2}, β={b2}, RMSE={rmse2:.6f}")

            predict_flow_percent(
                C_history=C_array,
                distances=distances,
                lambda_onehot=lambda_onehot,
                lambda_P=lambda_P,
                lambda_div=lambda_div,
                lambda_entry=lambda_entry,
                lambda_move=lambda_move,
                seed=seed,
                num_reads=num_reads,
                alpha=a2,  # <- pass the fixed value here (not None)
                beta=b2,
            )

        else:  # run prediction directly
            predict_flow_percent(
                C_array,
                dist_array,
                lambda_onehot,
                lambda_P,
                lambda_div,
                lambda_entry,
                lambda_move,
                seed,
                num_reads,
                alpha,
                beta,
            )
    except Exception as e:
        print(f"Error during flow prediction: {e}")
        return


if __name__ == "__main__":
    main()
