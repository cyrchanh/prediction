import csv
import itertools
import torch
from scipy import stats

from src.models import MLP
from src.data import load_task_data, ALL_DIGIT_PAIRS
from src.training import train, accuracy
from src.signals import predict_safer_ordering


def run_actual_sequence(pair_first, pair_second, epochs=5):
    x1, y1 = load_task_data(pair_first)
    x2, y2 = load_task_data(pair_second)
    model = MLP()
    train(model, x1, y1, epochs=epochs)
    acc_before = accuracy(model, x1, y1)
    train(model, x2, y2, epochs=epochs)
    acc_after = accuracy(model, x1, y1)
    return acc_before - acc_after  # forgetting on the FIRST task


def evaluate_pair(pair_a, pair_b, epochs=5):
    x_a, y_a = load_task_data(pair_a)
    x_b, y_b = load_task_data(pair_b)

    torch.manual_seed(0)
    probe_a = train(MLP(), x_a, y_a, epochs=epochs)
    torch.manual_seed(0)
    probe_b = train(MLP(), x_b, y_b, epochs=epochs)

    predicted_safer, conflict_ba, conflict_ab = predict_safer_ordering(
        probe_a, x_a, y_a, x_b, y_b, probe_b
    )

    forgetting_a_then_b = run_actual_sequence(pair_a, pair_b, epochs=epochs)
    forgetting_b_then_a = run_actual_sequence(pair_b, pair_a, epochs=epochs)

    actual_safer = "B_then_A" if forgetting_b_then_a < forgetting_a_then_b else "A_then_B"

    return {
        "pair_a": pair_a, "pair_b": pair_b,
        "conflict_b_disrupts_a": conflict_ba,
        "conflict_a_disrupts_b": conflict_ab,
        "predicted_safer": predicted_safer,
        "forgetting_a_then_b": forgetting_a_then_b,
        "forgetting_b_then_a": forgetting_b_then_a,
        "actual_safer": actual_safer,
        "hit": predicted_safer == actual_safer,
    }


if __name__ == "__main__":
    task_combinations = list(itertools.combinations(ALL_DIGIT_PAIRS[:6], 2))  # 15 pairs from 6 tasks

    results = []
    for pair_a, pair_b in task_combinations:
        print(f"Testing {pair_a} vs {pair_b}...")
        result = evaluate_pair(pair_a, pair_b)
        results.append(result)
        print(f"  predicted={result['predicted_safer']}, actual={result['actual_safer']}, hit={result['hit']}")

    n_hits = sum(r["hit"] for r in results)
    n_total = len(results)
    hit_rate = n_hits / n_total

    # Binomial test against chance (p=0.5)
    binom_result = stats.binomtest(n_hits, n_total, p=0.5, alternative="greater")

    print(f"\n=== SUMMARY ===")
    print(f"Hit rate: {n_hits}/{n_total} = {hit_rate:.3f}")
    print(f"Binomial test p-value (vs chance): {binom_result.pvalue:.4f}")

    with open("results/exp2_hit_rate.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
