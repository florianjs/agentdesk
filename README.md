# AgentDesk

> A support agent that diagnoses, proposes an action (refund, escalation), and **waits for human
> approval** before acting.

A persistent agent loop bounded by four budgets, with human-in-the-loop as a first-class state. Its
tools are the two sibling services — **Triagely** (classification) and **DocPilot** (product
knowledge) — and the whole suite is also exposed as an **MCP server**, drivable from any MCP client.

## Status

**v1.** The trajectory evals are green in CI and the suite is exposed over MCP.

| Area                                                             | Status |
| ---------------------------------------------------------------- | ------ |
| Project setup: uv, ruff, mypy strict, pytest, CI                 | ✅     |
| Tool registry with Pydantic schemas and structured tool errors   | ✅     |
| Agent loop with four budgets and clean wrap-up on overrun        | ✅     |
| Run state persisted per iteration, resumable after a crash       | ✅     |
| Human-in-the-loop: approve / reject endpoints                    | ✅     |
| Trajectory eval suite, including adversarial scenarios           | ✅     |
| State-graph execution with checkpoints and step streaming        | ✅     |
| MCP server exposing the suite → **v1**                           | ✅     |

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

`POST /v1/runs/stream` returns the same run as server-sent events — one per token, plus one when
each tool starts and ends, so the customer sees "looking up your order" instead of a spinner.

## Two engines

The agent exists twice: a hand-written loop (`agent/loop.py`) and a LangGraph state machine
(`graph.py`), selected per request with `?engine=native|graph`. Both are scored by the same thirty
scenarios — a migration validated by a different eval is validated by nothing.

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	agent(agent)
	act(act)
	approval(approval)
	wrap_up(wrap_up)
	__end__([<p>__end__</p>]):::last
	__start__ --> agent;
	act -.-> __end__;
	act -.-> agent;
	act -.-> approval;
	act -.-> wrap_up;
	agent -.-> __end__;
	agent -.-> act;
	approval -.-> agent;
	approval -.-> wrap_up;
	wrap_up --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

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

### Hand-written loop vs LangGraph

Both engines, same thirty scenarios, same thresholds, two runs in flight at a time so the numbers
are comparable:

| Engine                | Ordinary | Adversarial | median | p95     | iterations | cost / 30 runs |
| --------------------- | -------- | ----------- | ------ | ------- | ---------- | -------------- |
| hand-written          | 20/20    | 10/10       | 6.69 s | 10.68 s | 1.93       | $0.42          |
| LangGraph             | 20/20    | 10/10       | 7.35 s | 12.37 s | 1.93       | $0.38          |

**29 of 30 runs took the identical tool path.** The one that differed (`already-refunded`:
escalate vs. answer) also differs between two runs of the *same* engine — it is the model, not the
framework.

**What the framework gave.** Three things, and only one of them is large. `interrupt()` turns
human-in-the-loop into a function call that returns the human's answer: the hand-written engine
needed a second entry point, a serialiser and a table to reach the same place. `astream_events`
publishes every node and tool transition for free, which is the difference between a customer
watching a spinner for eight seconds and watching progress. `draw_mermaid()` produces the diagram
above from the code, so it cannot drift.

**What it cost.** A second Postgres driver — the checkpointer speaks psycopg while the rest of the
service speaks asyncpg — so one process now holds two connection pools to one database. Correct
behaviour also moved into constructor keywords: `ToolNode`'s default is to re-raise, so the first
graph run died on the scenario where the order service fails, a bug the hand-written engine cannot
have because its errors are caught by construction. The same thing happened again with provider
failures, which escape the graph as tracebacks until a node catches them. Neither is hard to fix;
both are invisible until traffic finds them, which is a different property from code that reads
wrong on the page.

**And what it did not give.** The checkpointer is not a substitute for the `runs` table. It stores
opaque blobs keyed by thread id, and "which runs are waiting for a human" is a business question —
so both engines still write a row, and the graph engine keeps its checkpoint on top. The module
this work follows suggests deleting the hand-rolled persistence once the checkpointer is in; that
would have deleted the approval queue with it.

**Verdict: worth it here, unlike LCEL in the sibling project.** The deciding feature is
`interrupt()`: human-in-the-loop is this product's core, and the framework does it better than the
code it replaces. Streaming step events is a real second reason. Had the agent been a straight
loop with no human in the middle, the answer would have been the same as DocPilot's — the
abstraction would have replaced working, inspectable code with an equivalent that is harder to
read. Both engines stay in the repository, and the evals decide, not the enthusiasm.

## MCP server

The three services are exposed as one MCP server, so any MCP client — Claude Code, Claude
Desktop, another agent — can triage a ticket, search the docs, or hand a whole ticket to the
agent.

```bash
uv run mcp dev src/agentdesk/mcp_server.py        # the official inspector
claude mcp add supportly -- uv run --directory /path/to/agentdesk python -m agentdesk.mcp_server
```

| Surface                                    | What it does                                        |
| ------------------------------------------ | --------------------------------------------------- |
| `classify_ticket(subject, body)`           | → Triagely: category, urgency, sentiment, language   |
| `search_product_docs(query, collection)`   | → DocPilot: five passages with sources and scores    |
| `start_support_run(subject, body)`         | → the agent: answers, escalates, or proposes         |
| `supportly://runs/{run_id}` *(resource)*   | status and transcript of one run                     |

**There is deliberately no `approve_run` tool.** A run that proposes a refund comes back as
`awaiting_approval` with the amount attached, and the only way to approve it is a human on
`POST /v1/runs/{id}/approve`. A client that could approve its own proposals would be an agent
holding its own rubber stamp — and a client's guardrails are not this server's to trust. A test
asserts the tool list stays that way.

The suspension survives the process, too: a run started inside the MCP server and approved later
through the HTTP API resumes from its Postgres checkpoint, in a different process entirely.

Remote deployments use the streamable-HTTP transport behind a shared key — a remote MCP server is
a public API whatever the protocol calls it, and serving it without `MCP_API_KEY` is refused at
startup:

```bash
uv run python -m agentdesk.mcp_server --http --port 8003
claude mcp add supportly --transport http http://localhost:8003/mcp --header "X-API-Key: $MCP_API_KEY"
```

Output is kept small on purpose: five hits, 400-character excerpts, transcripts without the
system prompt. Everything returned here is spent from the client's context on every later turn.

## Layout

```
src/agentdesk/
  agent/         loop, budgets, state, prompts, tool registry
  routes/        POST /runs, GET /runs/{id}, approve, reject
  graph.py       the same agent as a LangGraph state machine
  mcp_server.py  three tools and one resource; no way to approve anything
  judge.py       the one judged question, with its superseded rubric
  trajectory.py  scoring a run against what it was supposed to do
  db.py          one table; the transcript is jsonb, the counters are columns
evals/
  data/          30 scenarios, 12 judge calibration cases
  test_trajectories.py   the CI gate (thresholds at the measured lower bound)
scripts/
  measure_trajectories.py   the full suite; --engine graph, --control for the weak prompt
  calibrate_judge.py        both rubrics over the same labelled replies
```
