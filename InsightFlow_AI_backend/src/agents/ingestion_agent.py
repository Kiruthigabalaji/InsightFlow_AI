"""
agents/ingestion_agent.py
-------------------------
IngestionAgent — first specialist in the supervisor pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
* Call every data collector and merge results into a unified list of
  ArticleRecord objects stored on PipelineState.
* Evaluate each source's relevance to Commercial Real Estate (CRE)
  intelligence and log a justification for inclusion / deprioritisation.
* Record a TraceEvent on every ArticleRecord it creates so downstream
  agents know where the record originated.

Inputs
~~~~~~
    PipelineState (empty records list)

Outputs
~~~~~~~
    PipelineState.records          — populated with raw ArticleRecord objects
    PipelineState.source_relevance — per-source relevance scores + justifications
    PipelineState.audit_log        — ingestion phase events appended
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

# Allow running from the src/ directory directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from schema import ArticleRecord, PipelineState, utc_now

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Source relevance definitions
# These weights are used by the analysis agent to prioritise signals.
# Any source with relevance < 0.4 is ingested but flagged as low-priority
# so the exclusion rationale is captured in the audit log.
# ---------------------------------------------------------------------------

SOURCE_RELEVANCE: dict[str, dict[str, Any]] = {
    "PropertyWeek": {
        "weight": 0.85,
        "justification": (
            "Primary UK CRE trade publication. High signal density for market "
            "trends, lending activity, and geographic investment flows."
        ),
    },
    "JLL": {
        "weight": 0.90,
        "justification": (
            "Tier-1 CRE advisory firm. Authoritative on capital markets, "
            "occupier demand, and cross-border investment."
        ),
    },
    "Altus": {
        "weight": 0.80,
        "justification": (
            "Specialist CRE data and valuation firm. Strong on appraisal, "
            "risk analytics, and portfolio data."
        ),
    },
    "CRE_Lending": {
        "weight": 1.00,
        "justification": (
            "Structured lending dataset — highest factual confidence. "
            "Ground truth for lender activity and geographic exposure."
        ),
    },
    "FMP": {
        "weight": 0.70,
        "justification": (
            "Financial market data for publicly listed CRE firms. Useful for "
            "aligning news signals with stock/company fundamentals."
        ),
    },
    "cities": {
        "weight": 0.60,
        "justification": (
            "Reference dataset only — used for location normalisation. "
            "Not a content source; ingested for lookup purposes."
        ),
    },
}


# ---------------------------------------------------------------------------
# Safe collector imports — each collector lives in data_collection/
# ---------------------------------------------------------------------------


def _safe_import(module_name: str, func_name: str):
    """
    Import a data-collection function, returning None on failure.

    This pattern lets the pipeline proceed even when a collector is broken
    or its dependencies (Playwright, openpyxl) are not installed, while
    logging a clear error for the operator.
    """
    try:
        import importlib

        mod = importlib.import_module(f"data_collection.{module_name}")
        return getattr(mod, func_name)
    except Exception as exc:
        log.warning(
            "Collector import failed — skipping source",
            extra={"module": module_name, "func": func_name, "error": str(exc)},
        )
        return None


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------


class IngestionAgent:
    """
    Orchestrates all six data collectors and populates PipelineState.

    Design decisions
    ~~~~~~~~~~~~~~~~
    * Sources are collected sequentially to avoid hammering external sites.
    * Each source is assessed for relevance before ingestion; low-weight
      sources are still ingested but marked in source_relevance so the
      reporting agent can weight them appropriately.
    * Failures in individual collectors are caught and logged; the pipeline
      continues with remaining sources.
    """

    AGENT_NAME = "IngestionAgent"

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the ingestion phase.

        Parameters
        ----------
        state:
            Shared pipeline state.  ``state.records`` is expected to be empty
            on entry and will be populated on exit.

        Returns
        -------
        PipelineState
            The same object, mutated in place and returned for chaining.
        """
        log.info("Ingestion phase started", extra={"run_id": state.run_id})
        state.log(phase="ingestion", action="started", detail="Beginning multi-source ingestion")

        # Register source relevance assessments
        state.source_relevance = SOURCE_RELEVANCE
        state.log(
            phase="ingestion",
            action="relevance_assessed",
            detail="Source weights assigned before collection",
            sources=list(SOURCE_RELEVANCE.keys()),
        )

        total_before = len(state.records)

        # --- 1. PropertyWeek RSS / scraper ---
        self._ingest_propertyweek(state)

        # --- 2. JLL ---
        self._ingest_jll(state)

        # --- 3. Altus Group ---
        self._ingest_altus(state)

        # --- 4. CRE Lending dataset ---
        self._ingest_cre(state)

        # --- 5. FMP API ---
        self._ingest_fmp(state)

        added = len(state.records) - total_before
        log.info(
            "Ingestion phase complete",
            extra={"run_id": state.run_id, "records_added": added},
        )
        state.log(
            phase="ingestion",
            action="completed",
            detail=f"Ingested {added} records across {len(SOURCE_RELEVANCE)} sources",
            records_added=added,
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers — one per source
    # ------------------------------------------------------------------

    def _ingest_propertyweek(self, state: PipelineState) -> None:
        """Fetch PropertyWeek news articles and add to state."""
        source = "PropertyWeek"
        log.info("Collecting PropertyWeek data", extra={"source": source})

        collect = _safe_import("data_collection_rss_feed", "collect_propertyweek_data")
        if collect is None:
            self._log_collector_skip(state, source, "Import failed")
            return

        try:
            raw: list[dict] = collect(limit=12)
            for item in raw:
                rec = self._make_record(item, source)
                state.records.append(rec)
            log.info("PropertyWeek collected", extra={"source": source, "count": len(raw)})
            state.log(phase="ingestion", action="source_collected", source=source, count=len(raw))
        except Exception as exc:
            log.error("PropertyWeek collection failed", extra={"source": source, "error": str(exc)})
            self._log_collector_skip(state, source, str(exc))

    def _ingest_jll(self, state: PipelineState) -> None:
        """Fetch JLL insights articles via Playwright scraper."""
        source = "JLL"
        log.info("Collecting JLL data", extra={"source": source})

        collect = _safe_import("data_collection_jll", "collect_jll_data")
        if collect is None:
            self._log_collector_skip(state, source, "Import failed")
            return

        try:
            raw: list[dict] = collect(limit=12)
            for item in raw:
                rec = self._make_record(item, source)
                state.records.append(rec)
            log.info("JLL collected", extra={"source": source, "count": len(raw)})
            state.log(phase="ingestion", action="source_collected", source=source, count=len(raw))
        except Exception as exc:
            log.error("JLL collection failed", extra={"source": source, "error": str(exc)})
            self._log_collector_skip(state, source, str(exc))

    def _ingest_altus(self, state: PipelineState) -> None:
        """Fetch Altus Group insights articles."""
        source = "Altus"
        log.info("Collecting Altus data", extra={"source": source})

        collect = _safe_import("data_collection_altus_group", "collect_altus_data")
        if collect is None:
            self._log_collector_skip(state, source, "Import failed")
            return

        try:
            raw: list[dict] = collect(limit=12)
            for item in raw:
                rec = self._make_record(item, source)
                state.records.append(rec)
            log.info("Altus collected", extra={"source": source, "count": len(raw)})
            state.log(phase="ingestion", action="source_collected", source=source, count=len(raw))
        except Exception as exc:
            log.error("Altus collection failed", extra={"source": source, "error": str(exc)})
            self._log_collector_skip(state, source, str(exc))

    def _ingest_cre(self, state: PipelineState) -> None:
        """Load CRE lending structured dataset from Excel."""
        source = "CRE_Lending"
        log.info("Loading CRE lending dataset", extra={"source": source})

        load_df = _safe_import("data_collection_cre", "load_cre_dataset")
        norm_cols = _safe_import("data_collection_cre", "normalize_columns")
        transform = _safe_import("data_collection_cre", "transform_cre_to_schema")

        if None in (load_df, norm_cols, transform):
            self._log_collector_skip(state, source, "Import failed")
            return

        try:
            df = load_df()
            if df is None:
                self._log_collector_skip(state, source, "Excel file not found")
                return
            df = norm_cols(df)
            raw: list[dict] = transform(df)
            for item in raw:
                rec = self._make_record(item, source)
                state.records.append(rec)
            log.info("CRE dataset loaded", extra={"source": source, "count": len(raw)})
            state.log(phase="ingestion", action="source_collected", source=source, count=len(raw))
        except Exception as exc:
            log.error("CRE ingestion failed", extra={"source": source, "error": str(exc)})
            self._log_collector_skip(state, source, str(exc))

    def _ingest_fmp(self, state: PipelineState) -> None:
        """Fetch FMP company profiles for major CRE firms."""
        source = "FMP"
        log.info("Collecting FMP data", extra={"source": source})

        collect = _safe_import("data_collection_fmp", "collect_fmp_data")
        if collect is None:
            self._log_collector_skip(state, source, "Import failed")
            return

        try:
            raw: list[dict] = collect()
            for item in raw:
                rec = self._make_record(item, source)
                state.records.append(rec)
            log.info("FMP collected", extra={"source": source, "count": len(raw)})
            state.log(phase="ingestion", action="source_collected", source=source, count=len(raw))
        except Exception as exc:
            log.error("FMP collection failed", extra={"source": source, "error": str(exc)})
            self._log_collector_skip(state, source, str(exc))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_record(raw: dict[str, Any], source: str) -> ArticleRecord:
        """
        Convert a raw dict from a collector into an ArticleRecord.

        The record gets an initial TraceEvent so downstream agents always
        know where data came from and when it was ingested.
        """
        rec = ArticleRecord(
            id=str(uuid.uuid4()),
            source=source,
            raw_url=raw.get("link"),
            title=raw.get("title", ""),
            raw_text=raw.get("content", raw.get("summary", "")),
            date=raw.get("date"),
            # Pre-populate location if the collector already resolved it
            normalised_location=raw.get("location", "Unknown"),
            # CRE lending records sometimes carry pre-parsed entities
            entities=raw.get("entities", []),
        )
        rec.add_trace(
            agent=IngestionAgent.AGENT_NAME,
            action="ingested",
            detail=f"Record created from {source} collector",
            input_summary=f"url={raw.get('link')}",
            output_summary=f"title='{rec.title[:60]}'",
        )
        return rec

    @staticmethod
    def _log_collector_skip(state: PipelineState, source: str, reason: str) -> None:
        """Record that a collector was skipped and why."""
        state.log(
            phase="ingestion",
            action="source_skipped",
            source=source,
            reason=reason,
        )
        log.warning("Source skipped", extra={"source": source, "reason": reason})
