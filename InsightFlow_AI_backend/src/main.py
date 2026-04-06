"""
main.py
-------
FastAPI application for InsightFlow AI.

Exposes three endpoints required by the assessment spec:

    POST /run-pipeline   — Trigger the full multi-agent pipeline.
    GET  /get-insights   — Return the latest pipeline insights.
    GET  /query          — Answer a natural-language question against
                           the stored results.

Additional utility endpoints:
    GET  /health         — Liveness check for Cloud Run / load balancers.
    GET  /trace/{run_id} — Return the full audit log for a given run.
    GET  /sources        — Return source relevance rankings.
    GET  /records        — Return paginated list of processed records.

CORS
~~~~
Configured to allow requests from any origin in development.  In production,
set the ALLOWED_ORIGINS environment variable to a comma-separated list of
permitted frontend URLs (e.g. your Vercel deployment URL).

Running locally
~~~~~~~~~~~~~~~
    uvicorn main:app --reload --port 8080

Environment variables
~~~~~~~~~~~~~~~~~~~~~
    ANTHROPIC_API_KEY  — Required for LLM extraction.
    FMP_API_KEY        — Required for FMP data collection.
    BQ_PROJECT         — GCP project ID for BigQuery writes (optional).
    BQ_DATASET         — BigQuery dataset name (default: insightflow).
    ALLOWED_ORIGINS    — Comma-separated list of allowed CORS origins.
    LOG_LEVEL          — Logging verbosity (default: INFO).
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from logger import configure_logging, get_logger
from schema import PipelineState

from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory run store
# Stores the last N completed PipelineState objects keyed by run_id.
# In production this would be backed by Redis or Firestore.
# ---------------------------------------------------------------------------

_MAX_STORED_RUNS = 10
_runs: dict[str, PipelineState] = {}
_latest_run_id: str | None = None


def _store_run(state: PipelineState) -> None:
    """Persist a completed run in the in-memory store."""
    global _latest_run_id
    _runs[state.run_id] = state
    _latest_run_id = state.run_id
    # Evict oldest runs if over limit
    if len(_runs) > _MAX_STORED_RUNS:
        oldest = next(iter(_runs))
        del _runs[oldest]
        log.debug("Evicted oldest run from store", extra={"evicted_run_id": oldest})


# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------

def _parse_origins() -> list[str]:
    """
    Build the CORS allow-list from the ALLOWED_ORIGINS env var.

    Defaults to allowing all origins ("*") when not set — suitable for
    local development.  Always include localhost variants for developer
    convenience.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if not raw or raw.strip() == "*":
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    # Always allow localhost for developer testing
    for port in ("3000", "8080", "5173"):
        origins.append(f"http://localhost:{port}")
        origins.append(f"http://127.0.0.1:{port}")
    return list(dict.fromkeys(origins))  # deduplicate while preserving order


ALLOWED_ORIGINS = _parse_origins()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and perform any startup tasks."""
    configure_logging(run_id="startup")
    log.info(
        "InsightFlow AI starting",
        extra={
            "allowed_origins": ALLOWED_ORIGINS,
            "bq_project": os.getenv("BQ_PROJECT", "(not set)"),
        },
    )
    yield
    log.info("InsightFlow AI shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InsightFlow AI",
    description=(
        "Multi-agent CRE intelligence pipeline. "
        "Ingests PropertyWeek, JLL, Altus Group, CRE Lending data, and FMP "
        "to produce structured, explainable insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — must be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Run-Id"],
)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _ok(data: Any, run_id: str | None = None) -> JSONResponse:
    """Wrap a successful response with consistent envelope."""
    content = {"status": "ok", "data": data}
    headers = {"X-Run-Id": run_id} if run_id else {}
    return JSONResponse(content=content, headers=headers)


def _error(message: str, status_code: int = 400) -> JSONResponse:
    """Wrap an error response."""
    return JSONResponse(
        content={"status": "error", "message": message},
        status_code=status_code,
    )


def _get_run(run_id: str | None) -> PipelineState:
    """
    Retrieve a run by ID, or the latest run if run_id is None.

    Raises
    ------
    HTTPException(404)
        If no runs are stored or the requested run_id is not found.
    """
    if not _runs:
        raise HTTPException(status_code=404, detail="No pipeline runs found. POST /run-pipeline first.")

    if run_id:
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        return _runs[run_id]

    return _runs[_latest_run_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["utility"])
async def health():
    """
    Liveness probe for Cloud Run and load balancers.

    Returns 200 when the application is running.  Does not check external
    dependencies (BigQuery, Claude API) — use a readiness probe for that.
    """
    return {"status": "healthy", "version": app.version}


@app.post("/run-pipeline", tags=["pipeline"])
async def run_pipeline():
    """
    Trigger the full multi-agent CRE intelligence pipeline.

    This is a synchronous endpoint — it blocks until all agents have
    completed.  For production use with long-running pipelines, consider
    wrapping in a background task and polling /trace/{run_id}.

    Returns
    -------
    JSON object with:
    - run_id       : unique identifier for this run
    - started_at   : ISO timestamp
    - completed_at : ISO timestamp
    - total_records: number of records ingested
    - included     : records that passed validation
    - insights     : number of cross-source insights generated
    - audit_events : number of supervisor-level audit log entries
    """
    log.info("POST /run-pipeline received")

    try:
        # Import here to avoid circular imports at module load time
        from agents.supervisor import SupervisorAgent

        supervisor = SupervisorAgent()
        state = supervisor.run()
        _store_run(state)

        summary = {
            "run_id": state.run_id,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "total_records": len(state.records),
            "included": len(state.included_records()),
            "excluded": len(state.records) - len(state.included_records()),
            "insights": len(state.insights),
            "audit_events": len(state.audit_log),
        }
        log.info("Pipeline run complete", extra=summary)
        return _ok(summary, run_id=state.run_id)

    except Exception as exc:
        log.error("Pipeline run failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/get-insights", tags=["insights"])
async def get_insights(run_id: str | None = Query(None, description="Run ID; defaults to latest run.")):
    """
    Return all cross-source insights from the most recent (or specified) run.

    Each insight includes:
    - insight_id    : stable identifier (geo_activity, signal_alignment, etc.)
    - title         : human-readable title
    - description   : methodology and interpretation guide
    - data          : structured result data
    """
    log.info("GET /get-insights", extra={"run_id": run_id})
    state = _get_run(run_id)
    return _ok(
        {
            "run_id": state.run_id,
            "completed_at": state.completed_at,
            "insights": state.insights,
            "source_relevance": state.source_relevance,
        },
        run_id=state.run_id,
    )


@app.get("/query", tags=["insights"])
async def query(
    q: str = Query(..., description="Natural language question about the CRE data."),
    run_id: str | None = Query(None, description="Run ID; defaults to latest run."),
):
    """
    Answer a natural language question against the stored pipeline results.

    Supports questions such as:
    - "Which locations show the highest activity?"
    - "Where do signals and lending diverge?"
    - "What are the top investment markets?"
    - "Which records have low confidence?"

    The query is matched against insights and record aggregations using
    keyword rules.  For richer NL querying, point to a BigQuery dataset
    with a SQL interface.

    Returns
    -------
    JSON object with answer (text) and supporting data (list of records or
    insight excerpts).
    """
    log.info("GET /query", extra={"q": q, "run_id": run_id})
    state = _get_run(run_id)

    answer, data = _answer_query(q, state)

    return _ok({"run_id": state.run_id, "question": q, "answer": answer, "data": data}, run_id=state.run_id)


@app.get("/trace/{run_id}", tags=["utility"])
async def get_trace(run_id: str):
    """
    Return the full supervisor audit log for a specific pipeline run.

    The trace shows every agent phase, routing decision, and record-level
    event — satisfying the assessment's non-negotiable traceability requirement.
    """
    log.info("GET /trace", extra={"run_id": run_id})
    state = _get_run(run_id)
    return _ok(
        {
            "run_id": state.run_id,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "audit_log": state.audit_log,
        }
    )


@app.get("/sources", tags=["insights"])
async def get_sources(run_id: str | None = Query(None)):
    """
    Return the source relevance ranking from a pipeline run.

    Shows each source's ingestion-time weight, observed average confidence,
    and combined final rank score.
    """
    log.info("GET /sources", extra={"run_id": run_id})
    state = _get_run(run_id)

    ranked = sorted(
        [
            {"source": k, **v}
            for k, v in state.source_relevance.items()
        ],
        key=lambda x: x.get("final_rank_score", 0),
        reverse=True,
    )
    return _ok({"run_id": state.run_id, "sources": ranked}, run_id=state.run_id)


@app.get("/records", tags=["insights"])
async def get_records(
    run_id: str | None = Query(None),
    source: str | None = Query(None, description="Filter by source name."),
    signal_type: str | None = Query(None, description="Filter by signal_type."),
    location: str | None = Query(None, description="Filter by normalised_location."),
    included_only: bool = Query(True, description="Return only records that passed validation."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Return a paginated list of processed CRE records.

    Supports filtering by source, signal_type, and location.
    Pagination via page and page_size query parameters.
    """
    log.info(
        "GET /records",
        extra={"run_id": run_id, "source": source, "signal_type": signal_type},
    )
    state = _get_run(run_id)

    records = state.records
    if included_only:
        records = [r for r in records if r.included]
    if source:
        records = [r for r in records if r.source.lower() == source.lower()]
    if signal_type:
        records = [r for r in records if r.signal_type.lower() == signal_type.lower()]
    if location:
        records = [r for r in records if location.lower() in r.normalised_location.lower()]

    total = len(records)
    start = (page - 1) * page_size
    page_records = records[start: start + page_size]

    return _ok(
        {
            "run_id": state.run_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "records": [_serialise_record(r) for r in page_records],
        },
        run_id=state.run_id,
    )


# ---------------------------------------------------------------------------
# Query engine — keyword routing
# ---------------------------------------------------------------------------

def _answer_query(question: str, state: PipelineState) -> tuple[str, Any]:
    """
    Route a natural-language question to the appropriate data slice.

    Parameters
    ----------
    question:
        Raw question string from the user.
    state:
        Completed pipeline state.

    Returns
    -------
    tuple[str, Any]
        (answer_text, supporting_data)
    """
    q = question.lower()
    included = state.included_records()

    # --- "highest activity" / "top locations" ---
    if any(kw in q for kw in ("highest activity", "top location", "most active", "busiest")):
        from collections import Counter
        top = Counter(r.normalised_location for r in included).most_common(5)
        answer = (
            f"The top {len(top)} most active locations by record count are: "
            + ", ".join(f"{loc} ({cnt} records)" for loc, cnt in top)
            + "."
        )
        return answer, [{"location": l, "count": c} for l, c in top]

    # --- "diverge" / "signal vs lending" ---
    if any(kw in q for kw in ("diverge", "lending", "alignment", "mismatch")):
        for ins in state.insights:
            if ins.get("insight_id") == "signal_alignment":
                divergent = ins.get("divergent_locations", [])
                answer = (
                    f"Locations where news signals and lending activity diverge most: "
                    + (", ".join(divergent[:5]) if divergent else "None identified.")
                )
                return answer, ins.get("data", [])

    # --- "emerging" / "underrepresented" ---
    if any(kw in q for kw in ("emerging", "underrepresented", "overlooked", "growth")):
        for ins in state.insights:
            if ins.get("insight_id") == "emerging_markets":
                emerging = ins.get("emerging_markets", [])
                answer = (
                    "Potential emerging CRE markets (high news, low lending): "
                    + (", ".join(emerging[:5]) if emerging else "None identified.")
                )
                return answer, ins.get("data", {})

    # --- "low confidence" / "uncertain" ---
    if any(kw in q for kw in ("low confidence", "uncertain", "poor quality", "excluded")):
        excluded = [r for r in state.records if not r.included]
        answer = (
            f"{len(excluded)} records were excluded after validation. "
            + (
                "Top reasons: "
                + "; ".join(set(r.exclusion_reason or "" for r in excluded[:5] if r.exclusion_reason))
                if excluded else "No exclusions."
            )
        )
        return answer, [_serialise_record(r) for r in excluded[:10]]

    # --- "investment" / "lending" / any signal type ---
    for signal in ("investment", "lending", "market_trend", "regulatory", "valuation"):
        if signal.replace("_", " ") in q or signal in q:
            matching = [r for r in included if r.signal_type == signal]
            answer = (
                f"Found {len(matching)} records classified as '{signal}'. "
                + (
                    "Top locations: "
                    + ", ".join(r.normalised_location for r in matching[:5])
                    if matching else ""
                )
            )
            return answer, [_serialise_record(r) for r in matching[:10]]

    # --- Fallback: summary ---
    answer = (
        f"The latest pipeline run processed {len(state.records)} records "
        f"({len(included)} included after validation) and produced "
        f"{len(state.insights)} insights. "
        "Try asking: 'Which locations show the highest activity?' or "
        "'Where do signals and lending diverge?'"
    )
    return answer, [ins.get("title") for ins in state.insights]


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _serialise_record(rec) -> dict[str, Any]:
    """Convert an ArticleRecord to a JSON-safe dict, omitting raw_text."""
    return {
        "id": rec.id,
        "source": rec.source,
        "title": rec.title,
        "date": rec.date,
        "normalised_location": rec.normalised_location,
        "entities": rec.entities,
        "signal_type": rec.signal_type,
        "summary": rec.summary,
        "themes": rec.themes,
        "confidence": rec.confidence,
        "included": rec.included,
        "exclusion_reason": rec.exclusion_reason,
        "raw_url": rec.raw_url,
    }


# ---------------------------------------------------------------------------
# Entry point for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("ENV", "development") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
