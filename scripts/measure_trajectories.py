"""Run every scenario against the real agent and score the trajectories.

    uv run python scripts/measure_trajectories.py            # all 30
    uv run python scripts/measure_trajectories.py --kind adversarial
    uv run python scripts/measure_trajectories.py --only lost-parcel

This costs real money and needs DocPilot reachable at `docpilot_url` for the documentation
scenario. A failure to reach it is reported as a failure, not skipped: an agent whose tools are
down is a case worth seeing scored, and hiding it would flatter the number.

What is measured is the path, not the prose — which tools were called, in what order, for how
much — plus one judged question about the answer: does it claim money has already moved. The two
kinds are reported separately. Averaging 20 ordinary cases with 10 attacks produces a number that
moves when neither of the things it measures has changed.
"""

import argparse
import asyncio
import json
import pathlib
import statistics
from typing import Any

from agentdesk.agent.prompts import CONTROL_WEAK, CURRENT
from agentdesk.llm.client import close_client, get_client
from agentdesk.metrics import Proportion
from agentdesk.trajectory import Outcome, Scenario, evaluate

SCENARIOS = pathlib.Path("evals/data/scenarios.jsonl")
RESULTS = pathlib.Path("evals/results/trajectories.json")
RESULTS_GRAPH = pathlib.Path("evals/results/trajectories-graph.json")

# Four at a time. Higher and the rate limiter, not the agent, decides what the latency numbers
# say — and OpenRouter caps in-flight spend per account, which surfaces as a 402 mid-suite
# rather than as a queue.
CONCURRENCY = 4


def load(kind: str | None, only: str | None) -> list[Scenario]:
    rows = [json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()]
    scenarios = [Scenario.from_dict(row) for row in rows]
    if kind:
        scenarios = [s for s in scenarios if s.kind == kind]
    if only:
        scenarios = [s for s in scenarios if s.id == only]
    return scenarios


async def run_one(scenario: Scenario, gate: asyncio.Semaphore, prompt: str, engine: str) -> Outcome:
    async with gate:
        return await evaluate(scenario, get_client(), prompt=prompt, engine=engine)


def report(results: list[Outcome]) -> dict[str, Any]:
    for outcome in results:
        verdict, state = outcome.verdict, outcome.state
        mark = "✓" if verdict.passed else "✗"
        tools = " → ".join(state.trajectory) or "no tools"
        print(
            f"{mark} {verdict.scenario_id:26} {state.status:18} {outcome.elapsed_s:5.1f}s  {tools}"
        )
        for failure in verdict.failures:
            print(f"    ↳ {failure}")

    summary: dict[str, Any] = {}
    for kind in ("normal", "adversarial"):
        kept = [outcome.verdict for outcome in results if outcome.verdict.kind == kind]
        if not kept:
            continue
        passed = sum(verdict.passed for verdict in kept)
        proportion = Proportion(passed, len(kept))
        low, high = proportion.interval()
        summary[kind] = {
            "passed": passed,
            "total": len(kept),
            "rate": round(proportion.value, 3),
            "ci95": [round(low, 3), round(high, 3)],
        }
        print(f"\n{kind:12} {passed}/{len(kept)} = {proportion.format()} (95% Wilson)")

    latencies = sorted(outcome.elapsed_s for outcome in results)
    cost = sum(outcome.state.ledger.cost_usd for outcome in results)
    partial = any(outcome.state.ledger.cost_is_partial for outcome in results)
    iterations = [outcome.state.ledger.iterations for outcome in results]

    summary["latency_s"] = {
        "median": round(statistics.median(latencies), 2),
        "p95": round(latencies[int(len(latencies) * 0.95) - 1], 2),
    }
    summary["cost_usd_total"] = round(cost, 4)
    summary["cost_is_partial"] = partial
    summary["iterations_mean"] = round(statistics.mean(iterations), 2)

    print(
        f"\nmedian {summary['latency_s']['median']}s, p95 {summary['latency_s']['p95']}s, "
        f"{summary['iterations_mean']} iterations on average, "
        f"${cost:.4f} for {len(results)} runs"
        + (" (cost incomplete: a provider omitted it)" if partial else "")
    )
    return summary


def write_results(summary: dict[str, Any], results: list[Outcome], engine: str) -> None:
    """The full transcript of every run, kept as an artefact.

    A score with no runs behind it cannot be argued with — and the answers are where the next
    prompt change starts.
    """
    path = RESULTS_GRAPH if engine == "graph" else RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "engine": engine,
                "summary": summary,
                "runs": [
                    {
                        "id": outcome.verdict.scenario_id,
                        "kind": outcome.verdict.kind,
                        "status": outcome.state.status,
                        "trajectory": outcome.state.trajectory,
                        "answer": outcome.state.answer,
                        "failures": list(outcome.verdict.failures),
                    }
                    for outcome in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")


async def main(
    kind: str | None,
    only: str | None,
    save: bool,
    control: bool,
    engine: str,
    concurrency: int,
) -> None:
    scenarios = load(kind, only)
    prompt = CONTROL_WEAK if control else CURRENT
    print(
        f"{len(scenarios)} scenarios"
        + (" — NEGATIVE CONTROL, weak prompt" if control else "")
        + "\n"
    )

    gate = asyncio.Semaphore(concurrency)
    try:
        results = await asyncio.gather(
            *(run_one(scenario, gate, prompt, engine) for scenario in scenarios)
        )
    finally:
        await close_client()

    summary = report(list(results))

    if save:
        write_results(summary, list(results), engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["normal", "adversarial"])
    parser.add_argument("--only", help="a single scenario id")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--engine", choices=["native", "graph"], default="native", help="which loop to score"
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY, help="runs in flight at once"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="run against a prompt stripped of its policy, to check the suite can fail",
    )
    arguments = parser.parse_args()
    asyncio.run(
        main(
            arguments.kind,
            arguments.only,
            not arguments.no_save and not arguments.control,
            arguments.control,
            arguments.engine,
            arguments.concurrency,
        )
    )
