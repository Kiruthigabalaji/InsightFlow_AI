"""
agents/supervisor.py
---------------------
SupervisorAgent — top-level orchestrator for the InsightFlow AI pipeline.

The Supervisor is responsible for:
* Initialising a PipelineState with a unique run_id.
* Calling each specialist agent in the correct sequence.
* Routing low-confidence records from ValidationAgent back to
  ExtractionAgent (the feedback loop).
* Logging every phase transition and routing decision to the audit log.
* Returning the completed PipelineState to the FastAPI layer.

Agent execution sequence
~~~~~~~~~~~~~~~~~~~~~~~~
    IngestionAgent
        ↓
    NormalisationAgent
        ↓
    ExtractionAgent  (first pass)
        ↓
    ValidationAgent  (first pass) ─── flags low-confidence ──→ ExtractionAgent (refinement)
        ↓                                                              ↓
    ValidationAgent  (second pass, re-evaluates refined records) ←────┘
        ↓
    ReportingAgent

The feedback loop is explicit: the Supervisor checks whether any records
were flagged after the first validation pass and, if so, calls
ExtractionAgent with refinement=True before calling ValidationAgent again.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import configure_logging, get_logger
from schema import PipelineState, utc_now

from agents.extraction_agent import ExtractionAgent
from agents.ingestion_agent import IngestionAgent
from agents.normalisation_agent import NormalisationAgent
from agents.reporting_agent import ReportingAgent
from agents.validation_agent import ValidationAgent

log = get_logger(__name__)


class SupervisorAgent:
    """
    Orchestrates the full multi-agent CRE intelligence pipeline.

    Usage
    -----
        supervisor = SupervisorAgent()
        state = supervisor.run()
        # state.insights  → list of CRE insights
        # state.audit_log → full trace of every agent decision
    """

    AGENT_NAME = "SupervisorAgent"

    def __init__(self) -> None:
        self._ingestion = IngestionAgent()
        self._normalisation = NormalisationAgent()
        self._extraction = ExtractionAgent()
        self._validation = ValidationAgent()
        self._reporting = ReportingAgent()

    def run(self) -> PipelineState:
        """
        Execute the complete pipeline from ingestion to reporting.

        Returns
        -------
        PipelineState
            Fully populated state including insights, source_relevance,
            records, and audit_log.
        """
        run_id = str(uuid.uuid4())
        configure_logging(run_id=run_id)

        log.info("Pipeline started", extra={"run_id": run_id})

        state = PipelineState(run_id=run_id)
        state.log(
            phase="supervisor",
            action="pipeline_started",
            detail="InsightFlow AI pipeline initialised",
            run_id=run_id,
        )

        # ── Phase 1: Ingestion ─────────────────────────────────────────
        state = self._run_phase("ingestion", self._ingestion.run, state)

        # ── Phase 2: Normalisation ────────────────────────────────────
        state = self._run_phase("normalisation", self._normalisation.run, state)

        # ── Phase 3: Extraction (first pass) ─────────────────────────
        state = self._run_phase("extraction_pass_1", lambda s: self._extraction.run(s, refinement=False), state)

        # ── Phase 4: Validation (first pass) ─────────────────────────
        state = self._run_phase("validation_pass_1", lambda s: self._validation.run(s, pass_number=1), state)

        # ── Feedback loop ─────────────────────────────────────────────
        validation_summary = self._validation.summary(state)
        reprocess_count = validation_summary["needs_reprocessing"]

        log.info(
            "Post-validation summary",
            extra={"run_id": run_id, **validation_summary},
        )
        state.log(
            phase="supervisor",
            action="feedback_loop_evaluated",
            detail=f"{reprocess_count} records flagged for re-processing",
            **validation_summary,
        )

        if reprocess_count > 0:
            log.info(
                "Routing flagged records to ExtractionAgent for refinement",
                extra={"run_id": run_id, "count": reprocess_count},
            )
            state.log(
                phase="supervisor",
                action="rerouting",
                detail=f"Re-routing {reprocess_count} low-confidence records to extraction refinement",
                records_rerouted=reprocess_count,
            )

            # ── Phase 3b: Extraction (refinement pass) ────────────────
            state = self._run_phase(
                "extraction_refinement",
                lambda s: self._extraction.run(s, refinement=True),
                state,
            )

            # ── Phase 4b: Validation (second pass) ───────────────────
            state = self._run_phase(
                "validation_pass_2",
                lambda s: self._validation.run(s, pass_number=2),
                state,
            )

            final_summary = self._validation.summary(state)
            log.info("Post-refinement summary", extra={"run_id": run_id, **final_summary})
            state.log(
                phase="supervisor",
                action="post_refinement_summary",
                **final_summary,
            )
        else:
            log.info("No re-processing required — skipping refinement pass", extra={"run_id": run_id})
            state.log(
                phase="supervisor",
                action="feedback_loop_skipped",
                detail="All records passed validation on first pass",
            )

        # ── Phase 5: Reporting ────────────────────────────────────────
        state = self._run_phase("reporting", self._reporting.run, state)

        state.log(
            phase="supervisor",
            action="pipeline_complete",
            detail="All phases executed successfully",
            run_id=run_id,
            completed_at=state.completed_at,
            total_records=len(state.records),
            included_records=len(state.included_records()),
            insights_generated=len(state.insights),
        )
        log.info(
            "Pipeline complete",
            extra={
                "run_id": run_id,
                "total_records": len(state.records),
                "included": len(state.included_records()),
                "insights": len(state.insights),
            },
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_phase(self, phase_name: str, fn, state: PipelineState) -> PipelineState:
        """
        Execute a single pipeline phase with error handling and timing.

        Parameters
        ----------
        phase_name:
            Human-readable label for the phase (used in audit log).
        fn:
            Callable that accepts and returns PipelineState.
        state:
            Current pipeline state.

        Returns
        -------
        PipelineState
            State returned from fn, or the unchanged state if fn raised.

        Notes
        -----
        Exceptions are caught and logged rather than propagated so that a
        single agent failure does not abort the entire pipeline.  The audit
        log records the failure, and downstream agents receive whatever
        partial state was produced up to that point.
        """
        log.info("Phase starting", extra={"phase": phase_name, "run_id": state.run_id})
        state.log(phase="supervisor", action="phase_start", phase_name=phase_name)

        try:
            result = fn(state)
            state.log(phase="supervisor", action="phase_complete", phase_name=phase_name)
            log.info("Phase complete", extra={"phase": phase_name})
            return result
        except Exception as exc:
            log.error(
                "Phase failed",
                extra={"phase": phase_name, "error": str(exc), "run_id": state.run_id},
                exc_info=True,
            )
            state.log(
                phase="supervisor",
                action="phase_failed",
                phase_name=phase_name,
                error=str(exc),
            )
            return state  # continue with partial state
