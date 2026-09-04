"""Where runs live.

One table. A run is a transcript plus a set of counters, and both are written after every
iteration — an agent whose state exists only inside a request handler cannot be paused, and an
agent that cannot be paused cannot ask a human anything without holding the connection open for
as long as the person takes to answer.

The transcript is `jsonb` rather than a `messages` table. It is always read and written whole,
never queried by row, and normalising it would buy joins nobody needs while making "resend the
exact transcript" a reassembly job. The counters are real columns, because "which runs blew
their budget this week" is a question worth asking in SQL.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentdesk.agent.state import AgentState
from agentdesk.config import settings

_engine: AsyncEngine | None = None
_sessions: async_sessionmaker[AsyncSession] | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              text PRIMARY KEY,
    status          text NOT NULL,
    customer_email  text NOT NULL DEFAULT '',
    messages        jsonb NOT NULL,
    answer          text,
    pending_action  jsonb,
    stop_reason     text NOT NULL DEFAULT '',
    trajectory      jsonb NOT NULL DEFAULT '[]'::jsonb,
    engine          text NOT NULL DEFAULT 'native',
    budgets         jsonb NOT NULL,
    iterations      int NOT NULL DEFAULT 0,
    tokens          int NOT NULL DEFAULT 0,
    cost_usd        double precision NOT NULL DEFAULT 0,
    cost_is_partial boolean NOT NULL DEFAULT false,
    elapsed_s       double precision NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- The queue a human works from: everything suspended, oldest first.
CREATE INDEX IF NOT EXISTS runs_awaiting_idx ON runs (created_at)
    WHERE status = 'awaiting_approval';
"""

# One statement for both insert and update: a checkpoint runs on every iteration and must not
# have to know whether it is the first one.
UPSERT = """
INSERT INTO runs (id, status, customer_email, messages, answer, pending_action, stop_reason,
                  trajectory, engine, budgets, iterations, tokens, cost_usd, cost_is_partial,
                  elapsed_s)
VALUES (:id, :status, :customer_email, CAST(:messages AS jsonb), :answer,
        CAST(:pending_action AS jsonb), :stop_reason, CAST(:trajectory AS jsonb), :engine,
        CAST(:budgets AS jsonb), :iterations, :tokens, :cost_usd, :cost_is_partial, :elapsed_s)
ON CONFLICT (id) DO UPDATE SET
    status = EXCLUDED.status,
    messages = EXCLUDED.messages,
    answer = EXCLUDED.answer,
    pending_action = EXCLUDED.pending_action,
    stop_reason = EXCLUDED.stop_reason,
    trajectory = EXCLUDED.trajectory,
    engine = EXCLUDED.engine,
    iterations = EXCLUDED.iterations,
    tokens = EXCLUDED.tokens,
    cost_usd = EXCLUDED.cost_usd,
    cost_is_partial = EXCLUDED.cost_is_partial,
    elapsed_s = EXCLUDED.elapsed_s,
    updated_at = now()
"""

# `::text` on every jsonb column: asyncpg hands back a JSON string for jsonb, and the driver's
# exact behaviour is one more thing to remember at each call site. Casting in the query makes
# the row shape the same one `to_row` produced, so the round trip is symmetric by construction.
SELECT_RUN = """
SELECT id, status, customer_email, messages::text AS messages, answer,
       pending_action::text AS pending_action, stop_reason, trajectory::text AS trajectory,
       engine, budgets::text AS budgets, iterations, tokens, cost_usd, cost_is_partial
FROM runs WHERE id = :id
"""


def get_engine() -> AsyncEngine:
    global _engine, _sessions
    if _engine is None:
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        _sessions = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessions is not None
    return _sessions


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """One session, for one task.

    An `AsyncSession` is **not safe for concurrent use**. Sharing one across `asyncio.gather`
    raises `IllegalStateChangeError`, usually not at the overlapping query but later when the
    scope closes — which makes it look like a shutdown bug rather than a concurrency one.
    """
    async with get_sessionmaker()() as session:
        yield session


async def save_run(session: AsyncSession, state: AgentState) -> None:
    """Write the run as it currently stands, and commit.

    Committed per iteration on purpose: a checkpoint that is only flushed when the run finishes
    protects against nothing, since the crash it exists for is the one that stops the run.
    """
    await session.execute(text(UPSERT), state.to_row())
    await session.commit()


async def load_run(session: AsyncSession, run_id: str) -> AgentState | None:
    row = (await session.execute(text(SELECT_RUN), {"id": run_id})).mappings().first()
    return AgentState.from_row(dict(row)) if row else None


async def create_schema() -> None:
    """Idempotent schema creation — enough while the schema still moves; migrations take over
    before anything is deployed, because `CREATE TABLE IF NOT EXISTS` cannot express a column
    that changed type."""
    engine = get_engine()
    async with engine.begin() as connection:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                await connection.execute(text(statement))


async def dispose_engine() -> None:
    global _engine, _sessions
    if _engine is not None:
        await _engine.dispose()
        _engine, _sessions = None, None
