"""
agents/extraction_agent.py
---------------------------
ExtractionAgent — third specialist in the supervisor pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
* Use the Google Gemini API (free tier) to classify, summarise, and extract
  entities from every included ArticleRecord.
* Process records in configurable batches to avoid brute-force per-article
  LLM calls (assessment requirement: selective / batched usage).
* Populate signal_type, summary, entities, themes, and confidence on each
  record.
* Append a TraceEvent per record with the LLM's raw decision so the
  assessor can audit every classification.

LLM backend
~~~~~~~~~~~
Uses google-generativeai (Gemini 1.5 Flash) via GEMINI_API_KEY.
Gemini 1.5 Flash free-tier limits:
    • 15 RPM  (requests per minute)
    • 1 million TPM (tokens per minute)
    • 1,500 RPD (requests per day)
The agent respects this by sleeping between batches when needed.

LLM usage strategy
~~~~~~~~~~~~~~~~~~~
Records are batched BATCH_SIZE at a time.  Each batch produces one API
call with a structured JSON prompt.  This means ~60 records → ~12 API calls
(BATCH_SIZE=5) rather than 60 individual calls.  Low-confidence records
are NOT re-called here; the ValidationAgent flags them and the Supervisor
re-routes them for a second pass with a more focused prompt.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
    GEMINI_API_KEY          — Required. Free-tier key from aistudio.google.com
    GEMINI_MODEL            — Optional. Default: gemini-1.5-flash
    EXTRACTION_BATCH_SIZE   — Optional. Default: 5
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from schema import ArticleRecord, PipelineState

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BATCH_SIZE: int = int(os.getenv("EXTRACTION_BATCH_SIZE", "5"))
MAX_RETRIES: int = 3
RETRY_DELAY: float = 5.0   # seconds — Gemini free tier: 15 RPM, so be polite
BATCH_SLEEP: float = 4.0   # seconds between batches to stay under rate limits

VALID_SIGNAL_TYPES: set[str] = {
    "investment",
    "lending",
    "market_trend",
    "regulatory",
    "occupier_demand",
    "valuation",
    "other",
    "unknown",
}

# ---------------------------------------------------------------------------
# Prompt templates
# Gemini uses a single combined prompt (system + user merged) because the
# google-generativeai SDK's generate_content() method takes a single string
# or list of content parts.  We prepend the system instructions at the top.
# ---------------------------------------------------------------------------

FIRST_PASS_PROMPT_TEMPLATE = """\
You are a specialist Commercial Real Estate (CRE) intelligence analyst.
Your task is to analyse a batch of CRE articles / records and return
structured JSON intelligence for each one.

For each record return EXACTLY these fields:
- signal_type: one of [investment, lending, market_trend, regulatory,
  occupier_demand, valuation, other]
- summary: one sentence (30 words max) capturing the key CRE insight
- entities: list of named organisations, funds, or lenders mentioned
- themes: list of 1-3 high-level themes (e.g. "office demand", "ESG")
- confidence: float 0.0-1.0 reflecting your certainty in the classification
  (use 0.7 or above only when the content clearly maps to a CRE signal)

CRITICAL: Respond ONLY with a valid JSON array. No markdown fences, no prose,
no explanation — just the raw JSON array starting with [ and ending with ].
The array must contain exactly {n} objects in the same order as the input.

Records to analyse:
{records_json}
"""

REFINEMENT_PROMPT_TEMPLATE = """\
You are a specialist CRE analyst performing a second-pass review.
The following {n} records received low confidence scores on the first pass.
Re-analyse them carefully using all available context. Return improved
classifications using the same schema as before.

Fields required per record:
- signal_type: one of [investment, lending, market_trend, regulatory,
  occupier_demand, valuation, other]
- summary: one sentence (30 words max)
- entities: list of named organisations / funds / lenders
- themes: list of 1-3 high-level themes
- confidence: float 0.0-1.0

CRITICAL: Respond ONLY with a valid JSON array — no markdown, no prose.
Exactly {n} objects in the same order as the input.

Records to re-analyse:
{records_json}
"""


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------


class ExtractionAgent:
    """
    Classifies and summarises ArticleRecords using the Google Gemini API.

    Batched processing
    ------------------
    Records are processed BATCH_SIZE at a time.  On first pass, all
    included records are processed.  On refinement pass (called by the
    Supervisor after ValidationAgent flags records), only the flagged
    subset is sent with the refinement prompt.

    Rate limiting
    -------------
    The free tier allows 15 requests per minute.  The agent sleeps
    BATCH_SLEEP seconds between batches and uses exponential back-off
    on 429 (rate limit) responses.

    Graceful degradation
    --------------------
    If the API key is absent or a batch call fails after MAX_RETRIES,
    the affected records receive confidence=0.0 and signal_type='unknown'.
    They will be caught by ValidationAgent and logged appropriately.
    """

    AGENT_NAME = "ExtractionAgent"

    def __init__(self) -> None:
        self._model = self._init_client()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState, refinement: bool = False) -> PipelineState:
        """
        Run extraction on all included records (or flagged records if refinement=True).

        Parameters
        ----------
        state:
            Shared pipeline state.
        refinement:
            If True, only process records where needs_reprocessing=True,
            using the refinement prompt.

        Returns
        -------
        PipelineState
            Mutated in place and returned for chaining.
        """
        pass_label = "refinement" if refinement else "first_pass"

        log.info(
            "Extraction phase started",
            extra={"run_id": state.run_id, "pass": pass_label, "model": MODEL},
        )
        state.log(
            phase="extraction",
            action="started",
            detail=f"Starting {pass_label} extraction with Gemini ({MODEL})",
            pass_type=pass_label,
            llm_model=MODEL,
        )

        targets = [
            r for r in state.records
            if r.included and (r.needs_reprocessing if refinement else True)
        ]

        log.info(
            "Records selected for extraction",
            extra={"count": len(targets), "pass": pass_label},
        )

        if not targets:
            log.info("No records to extract — skipping", extra={"pass": pass_label})
            state.log(phase="extraction", action="skipped", detail="No target records")
            return state

        batches = [targets[i: i + BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
        successful_batches = failed_batches = 0

        for idx, batch in enumerate(batches, start=1):
            log.debug(
                "Processing batch",
                extra={"batch": idx, "total_batches": len(batches), "size": len(batch)},
            )
            try:
                results = self._call_gemini(batch, refinement=refinement)
                self._apply_results(batch, results, pass_label)
                successful_batches += 1
            except Exception as exc:
                log.error(
                    "Batch extraction failed",
                    extra={"batch": idx, "error": str(exc)},
                )
                failed_batches += 1
                for rec in batch:
                    rec.confidence = 0.0
                    rec.signal_type = "unknown"
                    rec.summary = ""
                    rec.add_trace(
                        agent=self.AGENT_NAME,
                        action="extraction_failed",
                        detail=f"Batch {idx} error: {exc}",
                        confidence=0.0,
                    )

            # Respect Gemini free-tier rate limits between batches
            if idx < len(batches):
                log.debug("Rate-limit sleep", extra={"sleep_seconds": BATCH_SLEEP})
                time.sleep(BATCH_SLEEP)

        state.log(
            phase="extraction",
            action="completed",
            pass_type=pass_label,
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            records_processed=len(targets),
        )
        log.info(
            "Extraction phase complete",
            extra={
                "run_id": state.run_id,
                "pass": pass_label,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
            },
        )
        return state

    # ------------------------------------------------------------------
    # Gemini API interaction
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        batch: list[ArticleRecord],
        refinement: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Send a batch to Gemini and return parsed JSON results.

        Parameters
        ----------
        batch:
            Records to classify.
        refinement:
            If True, use the refinement prompt template.

        Returns
        -------
        list[dict]
            One dict per record with keys: signal_type, summary, entities,
            themes, confidence.

        Raises
        ------
        RuntimeError
            If the API call fails after MAX_RETRIES or the response cannot
            be parsed as a JSON array of the expected length.
        """
        if not self._model:
            raise RuntimeError("Gemini client not initialised — check GEMINI_API_KEY")

        records_payload = [
            {
                "id": r.id,
                "source": r.source,
                "title": r.title,
                # Truncate to ~800 chars to stay within token budget per batch
                "text": r.raw_text[:800],
                "location": r.normalised_location,
            }
            for r in batch
        ]

        template = REFINEMENT_PROMPT_TEMPLATE if refinement else FIRST_PASS_PROMPT_TEMPLATE
        prompt = template.format(
            n=len(batch),
            records_json=json.dumps(records_payload, indent=2),
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.debug(
                    "Calling Gemini API",
                    extra={"attempt": attempt, "batch_size": len(batch), "model": MODEL},
                )
                response = self._model.generate_content(prompt)
                raw_text = response.text.strip()

                log.debug(
                    "Gemini response received",
                    extra={"response_length": len(raw_text), "attempt": attempt},
                )

                parsed = self._parse_response(raw_text, expected_count=len(batch))
                return parsed

            except Exception as exc:
                err_str = str(exc)
                # Detect rate-limit errors and back off longer
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                sleep = (RETRY_DELAY * attempt * 3) if is_rate_limit else (RETRY_DELAY * attempt)

                log.warning(
                    "Gemini API attempt failed",
                    extra={
                        "attempt": attempt,
                        "max_retries": MAX_RETRIES,
                        "error": err_str,
                        "rate_limited": is_rate_limit,
                        "retry_sleep": sleep,
                    },
                )

                if attempt < MAX_RETRIES:
                    time.sleep(sleep)
                else:
                    raise RuntimeError(
                        f"Gemini API failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, expected_count: int) -> list[dict[str, Any]]:
        """
        Parse the Gemini response into a list of classification dicts.

        Gemini sometimes wraps JSON in markdown fences despite instructions;
        this method strips them robustly before parsing.

        Parameters
        ----------
        raw:
            Raw text response from Gemini.
        expected_count:
            Number of records in the batch — used to validate array length.

        Returns
        -------
        list[dict]
            Validated list of classification dicts.

        Raises
        ------
        RuntimeError
            On JSON parse failure, wrong type, wrong length, or missing keys.
        """
        clean = raw

        # Strip markdown fences: ```json ... ``` or ``` ... ```
        if clean.startswith("```"):
            lines = clean.split("\n")
            # Remove first line (```json or ```) and last line (```)
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            clean = "\n".join(inner_lines).strip()

        # Sometimes Gemini adds a trailing comment after the JSON — strip it
        # by finding the last ] and truncating
        last_bracket = clean.rfind("]")
        if last_bracket != -1 and last_bracket < len(clean) - 1:
            clean = clean[: last_bracket + 1]

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini response is not valid JSON: {exc}\nRaw (first 400 chars): {raw[:400]}"
            ) from exc

        if not isinstance(parsed, list):
            raise RuntimeError(f"Expected JSON array, got {type(parsed).__name__}")

        if len(parsed) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} results, got {len(parsed)}. "
                f"Raw (first 200 chars): {raw[:200]}"
            )

        required = {"signal_type", "summary", "entities", "themes", "confidence"}
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise RuntimeError(f"Result {i} is not a dict: {type(item)}")
            missing = required - set(item.keys())
            if missing:
                raise RuntimeError(f"Result {i} missing required keys: {missing}")

        return parsed

    # ------------------------------------------------------------------
    # Result application
    # ------------------------------------------------------------------

    def _apply_results(
        self,
        batch: list[ArticleRecord],
        results: list[dict[str, Any]],
        pass_label: str,
    ) -> None:
        """
        Write Gemini classification results back onto ArticleRecords.

        Parameters
        ----------
        batch:
            The records that were classified.
        results:
            Parsed classification dicts from Gemini (same order as batch).
        pass_label:
            'first_pass' or 'refinement' — used in trace event labels.
        """
        for rec, result in zip(batch, results):
            # Sanitise and validate signal_type
            raw_signal = str(result.get("signal_type", "unknown")).lower().strip()
            signal = raw_signal if raw_signal in VALID_SIGNAL_TYPES else "other"

            if raw_signal not in VALID_SIGNAL_TYPES:
                log.debug(
                    "Non-standard signal type from Gemini — defaulting to 'other'",
                    extra={"raw_signal": raw_signal, "record_id": rec.id},
                )

            # Clamp confidence to [0.0, 1.0]
            conf = float(result.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))

            rec.signal_type = signal
            rec.summary = str(result.get("summary", "")).strip()
            rec.confidence = conf
            rec.themes = [str(t) for t in result.get("themes", [])]

            # Merge Gemini entities with any pre-existing entities from collectors
            llm_entities: list[str] = [str(e) for e in result.get("entities", [])]
            existing_lower = {e.lower() for e in rec.entities}
            for ent in llm_entities:
                if ent.strip() and ent.lower() not in existing_lower:
                    rec.entities.append(ent.strip())
                    existing_lower.add(ent.lower())

            rec.add_trace(
                agent=self.AGENT_NAME,
                action=f"classified_{pass_label}",
                detail=f"Gemini: signal={signal}, confidence={conf:.2f}",
                input_summary=rec.title[:60],
                output_summary=rec.summary[:80],
                confidence=conf,
            )

            log.debug(
                "Record classified",
                extra={
                    "record_id": rec.id,
                    "signal": signal,
                    "confidence": conf,
                    "pass": pass_label,
                    "model": MODEL,
                },
            )

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_client():
        """
        Initialise the Gemini GenerativeModel client.

        Returns None (with a warning log) if GEMINI_API_KEY is not set or
        if the google-generativeai package is not installed.

        Returns
        -------
        google.generativeai.GenerativeModel | None
        """
        if not GEMINI_API_KEY:
            log.warning(
                "GEMINI_API_KEY not set — extraction will be degraded. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
            return None

        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=GEMINI_API_KEY)

            # Generation config: low temperature for deterministic classification,
            # JSON output enforced by prompt.
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,        # near-deterministic for classification
                max_output_tokens=2048, # enough for a batch of 5 detailed records
            )

            model = genai.GenerativeModel(
                model_name=MODEL,
                generation_config=generation_config,
            )

            log.info(
                "Gemini client initialised",
                extra={"model": MODEL},
            )
            return model

        except ImportError:
            log.error(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
            return None
        except Exception as exc:
            log.error(
                "Failed to initialise Gemini client",
                extra={"error": str(exc)},
            )
            return None
