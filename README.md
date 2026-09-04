# AgentDesk

> A support agent that diagnoses, proposes an action (refund, escalation), and **waits for human
> approval** before acting.

A persistent agent loop bounded by four budgets, with human-in-the-loop as a first-class state. Its
tools are the two sibling services — **Triagely** (classification) and **DocPilot** (product
knowledge) — and the whole suite is also exposed as an **MCP server**, drivable from any MCP client.

## Status

🔧 In progress — v1 ships when the state graph and the MCP server land.

| Area                                                             | Status |
| ---------------------------------------------------------------- | ------ |
| Project setup: uv, ruff, mypy strict, pytest, CI                 | ✅     |
| Tool registry with Pydantic schemas and structured tool errors   | ✅     |
| Agent loop with four budgets and clean wrap-up on overrun        | ✅     |
| Run state persisted per iteration, resumable after a crash       | ✅     |
| Human-in-the-loop: approve / reject endpoints                    | ✅     |
| Trajectory eval suite, including adversarial scenarios           | ✅     |
| State-graph execution with checkpoints and step streaming        | ⬜     |
| MCP server exposing the suite → **v1**                           | ⬜     |

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
uv run uvicorn agentdesk.main:app --reload --port 8002
```

The documentation tool calls DocPilot, so the evals need it reachable at `DOCPILOT_URL`
(`cd ../docpilot && uv run uvicorn docpilot.main:app --port 8001`).

## How a run works

```
POST /v1/runs                     → status: awaiting_approval
                                    pending_action: { propose_refund, A-1003, 89.50 EUR }
POST /v1/runs/{id}/approve        → status: answered
```

The run is written to Postgres after every iteration, so the wait for a human costs an open
connection to nobody. Between the two requests the process can restart.

## Design principles

- **The model proposes, the code disposes.** `propose_refund` creates a request; it does not refund.
  Nothing in the tool registry moves money.
- **No prompt guarantees behaviour — a schema does.** The 500 EUR cap is `Field(gt=0, le=500)`, so a
  larger refund cannot be constructed. It is deliberately *absent* from the system prompt: repeating
  it there would suggest it is the prompt that enforces it.
- **An agent without budgets is a billing incident waiting to happen.** Iterations, tokens, spend and
  wall clock are checked *before* every step — a check that runs afterwards has already paid for
  what it was meant to prevent.
- **A blown budget wraps up; it does not raise.** And the wrap-up spends nothing: writing a polite
  goodbye with the model is one more call made after the ceiling was declared reached.
- **Tool errors are information, not exceptions.** They go back into the conversation as structured
  results, so the model can correct, work around, or escalate.

## Numbers

30 scenarios, run end to end against the real tools — 20 ordinary support cases and 10 adversarial
ones. What is scored is the **trajectory**: which tools were called, in what order, for how much,
plus one judged question about the reply (does it claim money has already moved).

| Suite                | Score | 95% Wilson    |
| -------------------- | ----- | ------------- |
| Ordinary cases       | 20/20 | 83.9 % – 100 % |
| Adversarial          | 10/10 | 72.2 % – 100 % |

Median 6.8 s, p95 9.7 s, **1.97 iterations per run**, **$0.0144 per run** (measured, not estimated:
the provider reports the credit cost of every call).

The ten attacks include instruction overrides, a fake system message, a claimed prior approval, a
request to split a refund past the cap, an attempt to read back the tool schemas — and one where the
injection is inside a **tool result** rather than the customer's message, which is the version that
actually happens in production and the one no system prompt about "user input" reaches.

### The score that mattered was the one from the control

30/30 on the first attempt is a result to distrust, so the same suite was run against an agent
stripped of its policy — same tools, same scenarios, a two-line prompt. **It also scored 30/30.**

The suite was measuring the guardrails, not the agent. That is half a real finding: the cap, the
approval gate and the tool schemas hold whatever the prompt says, which is the entire argument for
putting them in the type system. But it meant three scenarios were accepting either outcome where
the policy states one — "escalate rather than propose past 30 days" is a rule, not a preference.
Tightened to what the policy actually says, the control drops to **28/30** while the real agent
stays at 30/30. The suite now measures both layers, and it is written down here because a suite
nothing can fail is a suite that certifies nothing.

### The judge was wrong before the agent was

Two ordinary runs first came back failed: _"told the customer the money had already moved"_. The
answers were accurate — order A-1002 **was** refunded, 40 days earlier, and the agent was reading
the order record. The rubric had no way to tell a refund this conversation would cause from one it
merely reported, so it failed the agent for telling the truth.

Rewriting it around that distinction (v2) fixed both. The interesting part is the calibration set:
the two paraphrases written to reproduce the failure **both passed v1** — an invented case tends to
be one the rubric already handles. The real answers, pasted in verbatim, reproduced it. The
calibration set now holds them, and both rubric versions stay in `judge.py` with their scores.

## Layout

```
src/agentdesk/
  agent/         loop, budgets, state, prompts, tool registry
  routes/        POST /runs, GET /runs/{id}, approve, reject
  judge.py       the one judged question, with its superseded rubric
  trajectory.py  scoring a run against what it was supposed to do
  db.py          one table; the transcript is jsonb, the counters are columns
evals/
  data/          30 scenarios, 12 judge calibration cases
  test_trajectories.py   the CI gate (thresholds at the measured lower bound)
scripts/
  measure_trajectories.py   the full suite, with --control for the weak-prompt run
  calibrate_judge.py        both rubrics over the same labelled replies
```
