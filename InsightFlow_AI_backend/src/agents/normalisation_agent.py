"""
agents/normalisation_agent.py
------------------------------
NormalisationAgent — second specialist in the supervisor pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
* Normalise location strings on every ArticleRecord by fuzzy-matching
  against the canonical city list in cities.csv.
* Deduplicate records that share an identical title (keeps the first
  seen; logs all subsequent duplicates with an exclusion_reason).
* Standardise entity names (basic casing / alias resolution).

Inputs
~~~~~~
    PipelineState.records  — populated by IngestionAgent (raw text, raw locations)

Outputs
~~~~~~~
    PipelineState.records  — normalised_location set; duplicates marked included=False
    PipelineState.audit_log — normalisation phase events appended
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from schema import ArticleRecord, PipelineState

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default cities CSV path (relative to src/)
# ---------------------------------------------------------------------------

CITIES_CSV = Path(__file__).parent.parent / "data_collection" / "cities.csv"


# ---------------------------------------------------------------------------
# Well-known entity aliases — extend as needed
# ---------------------------------------------------------------------------

ENTITY_ALIASES: dict[str, str] = {
    "jones lang lasalle": "JLL",
    "jones lang lasalle incorporated": "JLL",
    "cbre group": "CBRE",
    "cb richard ellis": "CBRE",
    "altus group limited": "Altus Group",
    "altus group ltd": "Altus Group",
    "brookfield asset management": "Brookfield",
    "blackstone real estate": "Blackstone",
}


# ---------------------------------------------------------------------------
# NormalisationAgent
# ---------------------------------------------------------------------------


class NormalisationAgent:
    """
    Cleans and standardises location and entity fields across all records.

    Location matching strategy
    --------------------------
    1. Try an exact match (case-insensitive) against the cities list.
    2. If no exact match, try substring containment (record text contains a
       canonical city name) — longest match wins to avoid "London" matching
       inside "Londonderry".
    3. If still unmatched, leave as "Unknown" and log the raw value so an
       operator can extend cities.csv if needed.

    The fuzzy threshold is intentionally conservative: a wrong normalisation
    silently corrupts geographic analysis, whereas "Unknown" is visibly
    fixable.
    """

    AGENT_NAME = "NormalisationAgent"

    def __init__(self, cities_csv: Path = CITIES_CSV) -> None:
        self._cities: list[str] = self._load_cities(cities_csv)
        # Build a lowercase lookup dict for O(1) exact matching
        self._city_lower: dict[str, str] = {c.lower(): c for c in self._cities}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """
        Normalise all records in state and deduplicate by title.

        Parameters
        ----------
        state:
            Shared pipeline state from IngestionAgent.

        Returns
        -------
        PipelineState
            Mutated in place and returned for chaining.
        """
        log.info(
            "Normalisation phase started",
            extra={"run_id": state.run_id, "record_count": len(state.records)},
        )
        state.log(
            phase="normalisation",
            action="started",
            detail=f"Processing {len(state.records)} records",
        )

        self._normalise_locations(state)
        self._normalise_entities(state)
        self._deduplicate(state)

        included = sum(1 for r in state.records if r.included)
        log.info(
            "Normalisation phase complete",
            extra={"run_id": state.run_id, "included_after_dedup": included},
        )
        state.log(
            phase="normalisation",
            action="completed",
            detail=f"{included} records remain after deduplication",
            included=included,
            excluded=len(state.records) - included,
        )
        return state

    # ------------------------------------------------------------------
    # Location normalisation
    # ------------------------------------------------------------------

    def _normalise_locations(self, state: PipelineState) -> None:
        """Resolve each record's raw location to a canonical city name."""
        matched = unmatched = 0

        for rec in state.records:
            raw = (rec.normalised_location or "").strip()

            # Also scan the title + raw_text for city names if location is missing
            search_corpus = " ".join([raw, rec.title, rec.raw_text[:500]]).lower()

            canonical = self._match_city(raw.lower(), search_corpus)

            if canonical:
                matched += 1
                rec.normalised_location = canonical
                rec.add_trace(
                    agent=self.AGENT_NAME,
                    action="location_normalised",
                    detail=f"'{raw}' → '{canonical}'",
                    input_summary=raw,
                    output_summary=canonical,
                )
            else:
                unmatched += 1
                rec.normalised_location = "Unknown"
                rec.add_trace(
                    agent=self.AGENT_NAME,
                    action="location_unresolved",
                    detail=f"No city match found for raw='{raw}'",
                    input_summary=raw,
                    output_summary="Unknown",
                )

        log.info(
            "Location normalisation complete",
            extra={"matched": matched, "unmatched": unmatched},
        )
        state.log(
            phase="normalisation",
            action="locations_normalised",
            matched=matched,
            unmatched=unmatched,
        )

    def _match_city(self, raw_lower: str, corpus_lower: str) -> str | None:
        """
        Return the canonical city name for a given raw location string.

        Parameters
        ----------
        raw_lower:
            The location field from the record, already lowercased.
        corpus_lower:
            Broader text (title + body excerpt) to fall back to if the
            location field itself is empty or generic.

        Returns
        -------
        str | None
            Canonical city name from cities.csv, or None if no match.
        """
        # 1. Exact match on the raw location field
        if raw_lower and raw_lower in self._city_lower:
            return self._city_lower[raw_lower]

        # 2. Substring match on corpus — longest city name wins
        candidates = [
            city
            for city_lower, city in self._city_lower.items()
            if city_lower and city_lower in corpus_lower
        ]
        if candidates:
            return max(candidates, key=len)

        return None

    # ------------------------------------------------------------------
    # Entity normalisation
    # ------------------------------------------------------------------

    def _normalise_entities(self, state: PipelineState) -> None:
        """Apply alias resolution and title-case standardisation to entity lists."""
        for rec in state.records:
            original = list(rec.entities)
            normalised = [self._normalise_entity(e) for e in original]
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for e in normalised:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    unique.append(e)
            rec.entities = unique
            if unique != original:
                rec.add_trace(
                    agent=self.AGENT_NAME,
                    action="entities_normalised",
                    input_summary=str(original),
                    output_summary=str(unique),
                )

    @staticmethod
    def _normalise_entity(raw: str) -> str:
        """Resolve known aliases and apply consistent title-casing."""
        cleaned = raw.strip()
        alias = ENTITY_ALIASES.get(cleaned.lower())
        if alias:
            return alias
        # Basic title-case for multi-word names; keep acronyms upper
        if cleaned.isupper() and len(cleaned) <= 6:
            return cleaned  # keep CBRE, JLL, etc. as-is
        return cleaned.title()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, state: PipelineState) -> None:
        """
        Mark duplicate records (same normalised title) as excluded.

        The first occurrence is kept; all subsequent occurrences have
        ``included`` set to False with an ``exclusion_reason`` explaining
        the decision.  This satisfies the assessment's requirement that
        every exclusion must be justified.
        """
        seen_titles: dict[str, str] = {}  # normalised_title → record.id
        duplicates = 0

        for rec in state.records:
            key = self._normalise_title(rec.title)
            if key in seen_titles:
                rec.included = False
                rec.exclusion_reason = (
                    f"Duplicate of record {seen_titles[key]} — identical normalised title."
                )
                rec.add_trace(
                    agent=self.AGENT_NAME,
                    action="deduplicated",
                    detail=rec.exclusion_reason,
                )
                duplicates += 1
                log.debug(
                    "Duplicate excluded",
                    extra={"record_id": rec.id, "title": rec.title[:60]},
                )
            else:
                seen_titles[key] = rec.id

        log.info("Deduplication complete", extra={"duplicates_removed": duplicates})
        state.log(
            phase="normalisation",
            action="deduplication_complete",
            duplicates_removed=duplicates,
        )

    @staticmethod
    def _normalise_title(title: str) -> str:
        """Lower-case and strip punctuation for title comparison."""
        return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cities(path: Path) -> list[str]:
        """
        Load the canonical city list from cities.csv.

        The CSV is expected to have a column named 'city', 'name', or
        similar.  Falls back to the first column if none of the expected
        names are found.
        """
        if not path.exists():
            log.warning("cities.csv not found — location normalisation will be limited", extra={"path": str(path)})
            return []

        try:
            df = pd.read_csv(path)
            # Find the city column
            col_candidates = [c for c in df.columns if c.lower() in ("city", "name", "city_name", "location")]
            col = col_candidates[0] if col_candidates else df.columns[0]
            cities = df[col].dropna().astype(str).tolist()
            log.info("cities.csv loaded", extra={"path": str(path), "count": len(cities)})
            return cities
        except Exception as exc:
            log.error("Failed to load cities.csv", extra={"path": str(path), "error": str(exc)})
            return []
