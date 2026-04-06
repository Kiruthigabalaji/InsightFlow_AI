"""
schema.py
---------
Shared Pydantic models used across all agents and the FastAPI layer.

Every agent reads from and writes to PipelineState.  The audit_log field
is the non-negotiable traceability record required by the assessment spec.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Audit / tracing helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TraceEvent(BaseModel):
    """One atomic entry in an agent's execution log."""

    agent: str = Field(..., description="Name of the agent that produced this entry.")
    timestamp: str = Field(default_factory=utc_now)
    action: str = Field(..., description="Short verb describing what happened, e.g. 'fetched', 'classified'.")
    detail: str = Field(default="", description="Human-readable detail / reasoning.")
    input_summary: Optional[str] = Field(None, description="Brief digest of the input consumed.")
    output_summary: Optional[str] = Field(None, description="Brief digest of the output produced.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Per-article record
# ---------------------------------------------------------------------------


class ArticleRecord(BaseModel):
    """
    Canonical representation of a single piece of content after ingestion.

    All downstream agents mutate this record in-place (location, entities,
    signal_type, summary, confidence) while appending to agent_trace so the
    full decision history is preserved.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(..., description="Origin label: 'JLL', 'Altus', 'PropertyWeek', 'CRE_Lending', 'FMP'.")
    raw_url: Optional[str] = Field(None, description="URL the article was fetched from, if applicable.")
    title: str = Field(default="", description="Article or record headline.")
    raw_text: str = Field(default="", description="Raw / truncated body text before LLM processing.")
    date: Optional[str] = Field(None, description="Publication date as a raw string.")

    # Filled by NormalisationAgent
    normalised_location: str = Field(default="Unknown", description="Canonical city / region from cities.csv lookup.")

    # Filled by ExtractionAgent
    entities: list[str] = Field(default_factory=list, description="Named entities (companies, funds, lenders).")
    signal_type: str = Field(
        default="unknown",
        description="Classified signal: investment | lending | market_trend | regulatory | other.",
    )
    summary: str = Field(default="", description="LLM-generated one-sentence summary.")
    themes: list[str] = Field(default_factory=list, description="High-level themes extracted by LLM.")

    # Filled by ValidationAgent
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Composite confidence score 0-1.")
    needs_reprocessing: bool = Field(default=False, description="Set True by ValidationAgent if confidence < 0.5.")
    reprocessing_reason: Optional[str] = Field(None, description="Why the record was flagged for re-extraction.")

    # Relevance / inclusion decision (ValidationAgent / Supervisor)
    included: bool = Field(default=True, description="Whether this record contributes to final insights.")
    exclusion_reason: Optional[str] = Field(None, description="If excluded, the justification must be logged here.")

    # Full per-record trace
    agent_trace: list[TraceEvent] = Field(
        default_factory=list,
        description="Ordered list of trace events written by each agent that touched this record.",
    )

    def add_trace(self, **kwargs: Any) -> None:
        """Convenience wrapper so agents don't import TraceEvent directly."""
        self.agent_trace.append(TraceEvent(**kwargs))


# ---------------------------------------------------------------------------
# Pipeline-level state
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """
    Top-level object passed through the supervisor to every specialist agent.

    The audit_log captures supervisor-level decisions (phase transitions,
    re-routing events) while each ArticleRecord carries its own agent_trace
    for fine-grained per-record traceability.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(default_factory=utc_now)
    completed_at: Optional[str] = Field(None)

    records: list[ArticleRecord] = Field(default_factory=list)

    # Supervisor-level audit log
    audit_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered sequence of supervisor-level events (phase starts, re-routing, completion).",
    )

    # Source relevance assessments written by IngestionAgent
    source_relevance: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-source relevance scores and justifications produced by the ingestion agent.",
    )

    # Final cross-source insights written by ReportingAgent
    insights: list[dict[str, Any]] = Field(default_factory=list)

    def log(self, phase: str, action: str, detail: str = "", **extra: Any) -> None:
        """Append a supervisor-level audit event."""
        self.audit_log.append(
            {
                "phase": phase,
                "action": action,
                "detail": detail,
                "timestamp": utc_now(),
                **extra,
            }
        )

    def included_records(self) -> list[ArticleRecord]:
        """Return only records that passed validation."""
        return [r for r in self.records if r.included]
