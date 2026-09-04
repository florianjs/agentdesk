"""Check the judge before believing it.

    uv run python scripts/calibrate_judge.py

A judge is a model, so it is a thing that can be wrong, and a score produced by an unchecked
judge is a number with nothing behind it. This runs both rubric versions over the same twelve
replies — five that must be flagged, seven that must not — and prints where each one lands.

The seven negatives are the point. A judge that flags everything scores perfect recall and is
useless. Two of them are the verbatim answers that v1 wrongly failed during the trajectory run:
replies reporting a refund that genuinely happened weeks ago. They are here rather than a
paraphrase because the two paraphrases written first *passed* v1 — an invented case tends to be
one the rubric already handles, and it certifies nothing.
"""

import asyncio
import json
import pathlib
from typing import Any

from agentdesk.judge import RUBRIC, RUBRIC_V1, judge_payment_claim
from agentdesk.llm.client import close_client, get_client
from agentdesk.metrics import Proportion

CASES = pathlib.Path("evals/data/judge_calibration.jsonl")


def load() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


async def run(rubric: str, label: str) -> None:
    client = get_client()
    rows = load()

    verdicts = await asyncio.gather(
        *(judge_payment_claim(row["reply"], client=client, rubric=rubric) for row in rows)
    )

    print(f"\n── {label} " + "─" * (60 - len(label)))
    false_positives = false_negatives = 0
    for row, verdict in zip(rows, verdicts, strict=True):
        correct = verdict.claims_money_moved == row["should_flag"]
        if not correct:
            if verdict.claims_money_moved:
                false_positives += 1
                kind = "false positive — flagged an accurate reply"
            else:
                false_negatives += 1
                kind = "FALSE NEGATIVE — missed a promise of money"
            print(f"  ✗ {row['id']:24} {kind}")
            print(f"      {row['reply'][:110]}")
        else:
            print(f"  ✓ {row['id']:24} {'flagged' if verdict.claims_money_moved else 'passed'}")

    agreement = Proportion(len(rows) - false_positives - false_negatives, len(rows))
    print(f"  agreement with the labels: {agreement.format()}")
    print(f"  false positives {false_positives}, false negatives {false_negatives}")


async def main() -> None:
    try:
        await run(RUBRIC_V1, "v1 (superseded)")
        await run(RUBRIC, "v2 (current)")
    finally:
        await close_client()


if __name__ == "__main__":
    asyncio.run(main())
