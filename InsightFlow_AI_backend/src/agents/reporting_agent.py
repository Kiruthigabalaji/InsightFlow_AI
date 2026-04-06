"""
agents/reporting_agent.py
--------------------------
ReportingAgent — fifth and final specialist in the supervisor pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
* Aggregate validated ArticleRecords into at least three meaningful
  cross-source CRE insights.
* Produce a source relevance ranking using the weights assigned during
  ingestion plus the observed average confidence per source.
* Write the final structured dataset to BigQuery (if credentials are
  available) and return the complete PipelineState for the API layer.

Insights produced
~~~~~~~~~~~~~~~~~
1. Geographic activity — which locations appear most frequently across
   all sources and what signals are dominant there.
2. Signal alignment — where external news (PropertyWeek / JLL / Altus)
   agrees or diverges from structured lending data (CRE_Lending).
3. Emerging vs underrepresented markets — locations with rising news
   coverage but low lending activity (and vice versa).

Inputs
~~~~~~
    PipelineState.records  — validated records from ValidationAgent

Outputs
~~~~~~~
    PipelineState.insights     — list of structured insight dicts
    PipelineState.completed_at — timestamp set
    PipelineState.audit_log    — reporting phase events appended
    BigQuery                   — records written (if BQ_PROJECT env set)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from schema import ArticleRecord, PipelineState, utc_now

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# BigQuery config (optional — pipeline runs without it)
# ---------------------------------------------------------------------------

BQ_PROJECT = os.getenv("BQ_PROJECT", "")
BQ_DATASET = os.getenv("BQ_DATASET", "insightflow")
BQ_TABLE = os.getenv("BQ_TABLE", "cre_records")
BQ_AUDIT_TABLE = os.getenv("BQ_AUDIT_TABLE", "audit_log")

NEWS_SOURCES = {"PropertyWeek", "JLL", "Altus"}
LENDING_SOURCES = {"CRE_Lending"}


# ---------------------------------------------------------------------------
# ReportingAgent
# ---------------------------------------------------------------------------


class ReportingAgent:
    """
    Aggregates validated records into CRE insights and writes to BigQuery.

    All insights are deterministic (computed from record data, not from an
    LLM) so they are fully reproducible and traceable without additional
    API calls.
    """

    AGENT_NAME = "ReportingAgent"

    def run(self, state: PipelineState) -> PipelineState:
        """
        Generate insights, rank sources, and persist data.

        Parameters
        ----------
        state:
            Shared pipeline state after validation.

        Returns
        -------
        PipelineState
            Completed state with insights populated and timestamps set.
        """
        log.info(
            "Reporting phase started",
            extra={"run_id": state.run_id, "included_records": len(state.included_records())},
        )
        state.log(phase="reporting", action="started", detail="Generating insights and writing output")

        included = state.included_records()

        if not included:
            log.warning("No included records — reporting skipped", extra={"run_id": state.run_id})
            state.log(phase="reporting", action="skipped", detail="No included records")
            state.completed_at = utc_now()
            return state

        # --- Generate insights ---
        insight_geo = self._geographic_activity_insight(included)
        insight_signal = self._signal_alignment_insight(included)
        insight_emerging = self._emerging_markets_insight(included)

        state.insights = [insight_geo, insight_signal, insight_emerging]

        for i, ins in enumerate(state.insights, start=1):
            log.info(
                "Insight generated",
                extra={"insight_number": i, "title": ins["title"]},
            )
            state.log(
                phase="reporting",
                action="insight_generated",
                insight_number=i,
                title=ins["title"],
                locations=ins.get("top_locations", []),
            )

        # --- Source relevance ranking ---
        self._rank_sources(state, included)

        # --- Persist to BigQuery ---
        self._write_bigquery(state, included)

        state.completed_at = utc_now()
        state.log(
            phase="reporting",
            action="completed",
            detail="Pipeline complete",
            insights_generated=len(state.insights),
        )
        log.info(
            "Reporting phase complete",
            extra={"run_id": state.run_id, "insights": len(state.insights)},
        )
        return state

    # ------------------------------------------------------------------
    # Insight 1: Geographic activity
    # ------------------------------------------------------------------

    def _geographic_activity_insight(self, records: list[ArticleRecord]) -> dict[str, Any]:
        """
        Identify which locations generate the most CRE signal activity.

        Aggregates record counts per normalised location and computes the
        dominant signal type at each location.
        """
        location_counts: Counter = Counter()
        location_signals: dict[str, Counter] = defaultdict(Counter)
        location_sources: dict[str, set] = defaultdict(set)

        for rec in records:
            loc = rec.normalised_location
            location_counts[loc] += 1
            location_signals[loc][rec.signal_type] += 1
            location_sources[loc].add(rec.source)

        top_locations = location_counts.most_common(10)

        summary_rows = []
        for loc, count in top_locations:
            dominant_signal = location_signals[loc].most_common(1)[0][0]
            sources = sorted(location_sources[loc])
            summary_rows.append(
                {
                    "location": loc,
                    "record_count": count,
                    "dominant_signal": dominant_signal,
                    "sources": sources,
                }
            )

        return {
            "insight_id": "geo_activity",
            "title": "Geographic CRE Activity Distribution",
            "description": (
                "Locations ranked by total signal volume across all ingested sources. "
                "Dominant signal type indicates whether a market is primarily driven by "
                "investment, lending, or market trend reporting."
            ),
            "top_locations": [r["location"] for r in summary_rows[:5]],
            "data": summary_rows,
            "methodology": "Record count aggregation per normalised_location with dominant signal_type per location.",
        }

    # ------------------------------------------------------------------
    # Insight 2: Signal alignment (news vs lending)
    # ------------------------------------------------------------------

    def _signal_alignment_insight(self, records: list[ArticleRecord]) -> dict[str, Any]:
        """
        Compare news-source signal distribution against lending data signals.

        Locations where news volume is high but lending activity is low (or
        vice versa) represent interesting market divergences worth flagging.
        """
        news_locations: Counter = Counter()
        lending_locations: Counter = Counter()

        for rec in records:
            loc = rec.normalised_location
            if rec.source in NEWS_SOURCES:
                news_locations[loc] += 1
            elif rec.source in LENDING_SOURCES:
                lending_locations[loc] += 1

        all_locations = set(news_locations.keys()) | set(lending_locations.keys())

        alignment_rows = []
        for loc in sorted(all_locations):
            news_count = news_locations.get(loc, 0)
            lending_count = lending_locations.get(loc, 0)
            total = news_count + lending_count
            if total == 0:
                continue
            news_pct = round(news_count / total * 100, 1)
            status = self._alignment_status(news_count, lending_count)
            alignment_rows.append(
                {
                    "location": loc,
                    "news_signals": news_count,
                    "lending_signals": lending_count,
                    "news_pct": news_pct,
                    "alignment_status": status,
                }
            )

        alignment_rows.sort(key=lambda x: x["news_signals"] + x["lending_signals"], reverse=True)

        divergent = [r for r in alignment_rows if r["alignment_status"] != "aligned"]

        return {
            "insight_id": "signal_alignment",
            "title": "News Signal vs Lending Activity Alignment",
            "description": (
                "Identifies where external media coverage aligns with or diverges from "
                "actual lending transactions. High news / low lending may indicate "
                "emerging or speculative markets; low news / high lending may indicate "
                "mature, less-publicised markets."
            ),
            "divergent_locations": [r["location"] for r in divergent[:5]],
            "data": alignment_rows[:20],
            "methodology": (
                "News sources: PropertyWeek, JLL, Altus. "
                "Lending source: CRE_Lending dataset. "
                "Status: aligned (within 30% of each other), news_led, lending_led."
            ),
        }

    @staticmethod
    def _alignment_status(news: int, lending: int) -> str:
        """Classify alignment between news and lending signals."""
        if news == 0 and lending == 0:
            return "no_data"
        if news == 0:
            return "lending_led"
        if lending == 0:
            return "news_led"
        ratio = news / (news + lending)
        if ratio > 0.65:
            return "news_led"
        if ratio < 0.35:
            return "lending_led"
        return "aligned"

    # ------------------------------------------------------------------
    # Insight 3: Emerging vs underrepresented markets
    # ------------------------------------------------------------------

    def _emerging_markets_insight(self, records: list[ArticleRecord]) -> dict[str, Any]:
        """
        Surface locations that appear in recent news but have low lending data
        (emerging) or vice versa (underrepresented in media).
        """
        news_records = [r for r in records if r.source in NEWS_SOURCES]
        lending_records = [r for r in records if r.source in LENDING_SOURCES]

        news_locs = Counter(r.normalised_location for r in news_records)
        lending_locs = Counter(r.normalised_location for r in lending_records)

        news_only = sorted(
            [loc for loc in news_locs if loc not in lending_locs and loc != "Unknown"],
            key=lambda l: -news_locs[l],
        )
        lending_only = sorted(
            [loc for loc in lending_locs if loc not in news_locs and loc != "Unknown"],
            key=lambda l: -lending_locs[l],
        )

        return {
            "insight_id": "emerging_markets",
            "title": "Emerging and Underrepresented CRE Markets",
            "description": (
                "Emerging markets: locations generating news coverage but absent from "
                "lending data — potential growth markets or early-cycle activity. "
                "Underrepresented markets: locations with active lending but limited "
                "media coverage — potentially overlooked by the market."
            ),
            "emerging_markets": news_only[:8],
            "underrepresented_markets": lending_only[:8],
            "data": {
                "emerging": [{"location": l, "news_count": news_locs[l]} for l in news_only[:8]],
                "underrepresented": [{"location": l, "lending_count": lending_locs[l]} for l in lending_only[:8]],
            },
            "methodology": (
                "Locations present in news sources but absent from CRE_Lending = emerging. "
                "Locations present in CRE_Lending but absent from news = underrepresented."
            ),
        }

    # ------------------------------------------------------------------
    # Source relevance ranking
    # ------------------------------------------------------------------

    def _rank_sources(self, state: PipelineState, records: list[ArticleRecord]) -> None:
        """
        Combine ingestion-time source weights with observed avg confidence
        to produce a final source relevance ranking.
        """
        source_conf: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            source_conf[rec.source].append(rec.confidence)

        for source, weight_info in state.source_relevance.items():
            avg_conf = (
                sum(source_conf[source]) / len(source_conf[source])
                if source_conf[source] else 0.0
            )
            state.source_relevance[source]["avg_observed_confidence"] = round(avg_conf, 4)
            state.source_relevance[source]["final_rank_score"] = round(
                0.5 * weight_info["weight"] + 0.5 * avg_conf, 4
            )

        ranked = sorted(
            state.source_relevance.items(),
            key=lambda x: x[1].get("final_rank_score", 0),
            reverse=True,
        )
        state.log(
            phase="reporting",
            action="sources_ranked",
            ranking=[{"source": k, "score": v.get("final_rank_score")} for k, v in ranked],
        )
        log.info("Source ranking complete", extra={"ranking": [(k, v.get("final_rank_score")) for k, v in ranked]})

    # ------------------------------------------------------------------
    # BigQuery persistence
    # ------------------------------------------------------------------

    def _write_bigquery(self, state: PipelineState, records: list[ArticleRecord]) -> None:
        """
        Write records and audit log to BigQuery if BQ_PROJECT is configured.

        Gracefully skips if the google-cloud-bigquery package is not installed
        or credentials are not available.
        """
        if not BQ_PROJECT:
            log.info("BQ_PROJECT not set — skipping BigQuery write")
            state.log(phase="reporting", action="bigquery_skipped", reason="BQ_PROJECT not set")
            return

        try:
            from google.cloud import bigquery  # type: ignore

            client = bigquery.Client(project=BQ_PROJECT)
            table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
            audit_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_AUDIT_TABLE}"

            rows = [self._record_to_bq_row(r, state.run_id) for r in records]
            errors = client.insert_rows_json(table_ref, rows)
            if errors:
                log.error("BigQuery insert errors", extra={"errors": errors[:3]})
            else:
                log.info("Records written to BigQuery", extra={"table": table_ref, "count": len(rows)})

            audit_rows = [{"run_id": state.run_id, "event": json.dumps(e)} for e in state.audit_log]
            client.insert_rows_json(audit_ref, audit_rows)
            log.info("Audit log written to BigQuery", extra={"table": audit_ref, "count": len(audit_rows)})

            state.log(
                phase="reporting",
                action="bigquery_written",
                records_table=table_ref,
                records_count=len(rows),
                audit_table=audit_ref,
            )

        except ImportError:
            log.warning("google-cloud-bigquery not installed — skipping BQ write")
            state.log(phase="reporting", action="bigquery_skipped", reason="package not installed")
        except Exception as exc:
            log.error("BigQuery write failed", extra={"error": str(exc)})
            state.log(phase="reporting", action="bigquery_failed", error=str(exc))

    @staticmethod
    def _record_to_bq_row(rec: ArticleRecord, run_id: str) -> dict[str, Any]:
        """Serialise an ArticleRecord to a BigQuery-compatible dict."""
        return {
            "run_id": run_id,
            "record_id": rec.id,
            "source": rec.source,
            "raw_url": rec.raw_url,
            "title": rec.title,
            "date": rec.date or "",
            "normalised_location": rec.normalised_location,
            "entities": json.dumps(rec.entities),
            "signal_type": rec.signal_type,
            "summary": rec.summary,
            "themes": json.dumps(rec.themes),
            "confidence": rec.confidence,
            "included": rec.included,
            "exclusion_reason": rec.exclusion_reason or "",
        }
