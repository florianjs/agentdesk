# AgentDesk

> A support agent that diagnoses, proposes an action (refund, escalation), and **waits for human
> approval** before acting.

A persistent agent loop bounded by four budgets, with human-in-the-loop as a first-class state. Its
tools are the two sibling services — **Triagely** (classification) and **DocPilot** (product
knowledge) — and the whole suite is also exposed as an **MCP server**, drivable from any MCP client.

## Status

⬜ Not started — v1 ships when the trajectory evals are green in CI.

| Area                                                           | Status |
| -------------------------------------------------------------- | ------ |
| Project setup: uv, ruff, mypy strict, pytest, CI               | ✅     |
| Tool registry with Pydantic schemas and structured tool errors | ⬜     |
| Agent loop with four budgets and clean wrap-up on overrun      | ⬜     |
| Run state persisted per iteration, resumable after a crash     | ⬜     |
| Human-in-the-loop: approve / reject endpoints                  | ⬜     |
| Trajectory eval suite, including adversarial scenarios         | ⬜     |
| State-graph execution with checkpoints and step streaming      | ⬜     |
| MCP server exposing the suite → **v1**                         | ⬜     |

## Stack

Python 3.12 · uv · FastAPI · Pydantic v2 · Postgres · LangGraph · MCP · pytest · ruff · mypy strict
· OpenRouter

## Getting started

```bash
uv sync
cp .env.example .env          # add your OpenRouter key
docker compose up -d          # Postgres (port 5433, so it can run alongside DocPilot)
uv run pytest
uv run ruff check . && uv run mypy
```

## Design principles

- **The model proposes, the code disposes.** `propose_refund` creates a request; it does not refund.
  The refund executes on `POST /v1/runs/{id}/approve`, never inside the model's turn.
- **No prompt guarantees behaviour — a state machine does.** Caps live in the schema
  (`amount_eur ≤ 500`), not in instructions.
- **An agent without budgets is a billing incident waiting to happen.** Iterations, tokens, cost and
  wall time are checked before every step.
- **Tool errors are information, not exceptions.** They go back into the conversation so the model
  can retry, work around, or escalate.
- **Clean termination is a first-class feature**, not an error path.

## Numbers

_Published on v1: clean-exit rate, iterations per run, relevant-tool rate and cost across the
trajectory eval suite, including adversarial scenarios._
