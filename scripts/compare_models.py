"""Compare the models that have been scored, on the same thirty scenarios.

    uv run python scripts/measure_trajectories.py --model z-ai/glm-5.3-flash
    uv run python scripts/compare_models.py

Reads whatever `measure_trajectories.py` has written and lines it up. Two things it does that a
bare table does not:

- **Failures are split by kind.** A run that died because the provider returned an error inside an
  HTTP 200 is not the same as a model that skipped the lookup, and averaging them into one
  "accuracy" hides which one you would be buying.
- **The comparison is paired.** Every model ran the same thirty cases, so the question is not
  whether two independent intervals overlap — it is how many cases they *disagree* on, which is
  McNemar's test. Independent intervals on n=30 call almost everything a tie.
"""

import json
import pathlib
from typing import Any

from agentdesk.metrics import Proportion, mcnemar_p_value

RESULTS = pathlib.Path("evals/results")
BASELINE = "anthropic/claude-sonnet-4.5"

# Blended cost per million tokens, from OpenRouter's price list on 2026-09-04. Kept here rather
# than fetched: a table that changes under you between two readings is not a comparison.
PRICES = {
    "anthropic/claude-sonnet-4.5": "3.00 / 15.00",
    "deepseek/deepseek-v4-flash": "0.089 / 0.177",
    "z-ai/glm-5.3-flash": "0.075 / 0.250",
    "qwen/qwen3.7-flash": "0.030 / 0.130",
    "google/gemini-2.5-flash-lite": "0.100 / 0.400",
}


def load() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("trajectories*.json")):
        data = json.loads(path.read_text())
        if data.get("engine", "native") != "native":
            continue
        runs[data.get("model", BASELINE)] = data
    return runs


def kinds(data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Passed, and the three ways of failing, which are not interchangeable."""
    failed = [run for run in data["runs"] if run["failures"]]
    upstream = sum(run["status"] == "failed" for run in failed)
    budget = sum(run["status"] == "budget_exceeded" for run in failed)
    return len(data["runs"]) - len(failed), upstream, budget, len(failed) - upstream - budget


def main() -> None:
    scored = load()
    if BASELINE not in scored:
        raise SystemExit(f"no baseline run for {BASELINE}; run the script without --model first")

    baseline = {run["id"]: not run["failures"] for run in scored[BASELINE]["runs"]}

    header = (
        f"{'model':30} {'price in/out':>14} {'pass':>6} {'up':>3} {'bud':>4} {'miss':>5} "
        f"{'median':>7} {'p95':>7} {'$/run':>8} {'vs baseline':>28}"
    )
    print(header)
    print("-" * len(header))

    for model, data in sorted(
        scored.items(), key=lambda item: -item[1]["summary"]["cost_usd_total"]
    ):
        passed, upstream, budget, miss = kinds(data)
        summary = data["summary"]
        cost = summary["cost_usd_total"] / len(data["runs"])

        verdict = "— baseline —"
        if model != BASELINE:
            outcomes = {run["id"]: not run["failures"] for run in data["runs"]}
            only_baseline = sum(
                1 for case, ok in baseline.items() if ok and not outcomes.get(case, False)
            )
            only_model = sum(
                1 for case, ok in outcomes.items() if ok and not baseline.get(case, False)
            )
            p = mcnemar_p_value(only_baseline, only_model)
            direction = "worse" if only_baseline > only_model else "better"
            verdict = (
                f"{only_baseline}/{only_model} discordant, p={p:.3f} "
                f"({direction if p < 0.05 else 'not significant'})"
            )

        print(
            f"{model:30} {PRICES.get(model, '?'):>14} "
            f"{passed:>3}/{len(data['runs']):<2} {upstream:>3} {budget:>4} {miss:>5} "
            f"{summary['latency_s']['median']:>6.1f}s {summary['latency_s']['p95']:>6.1f}s "
            f"{cost:>8.4f} {verdict:>28}"
        )

    print("\nup = upstream failure (provider error), bud = budget exceeded, miss = policy miss")
    print("Interval on the baseline's overall score:", Proportion(30, 30).format())


if __name__ == "__main__":
    main()
