"""Service for resolving WebQSP entity IDs into readable names."""

from __future__ import annotations

import json
from pathlib import Path

from helpers.constants import WEBQSP_ENTITY_NAMES_MAPPING_PATH
from helpers.logging_config import get_logger
from pipeline.preparation.models.webqsp_local_graph import WebQSPEntityMappingSummary
from pipeline.services import AbstractService

logger = get_logger(__name__)


class WebQSPEntityNameMappingService(AbstractService):
    """Resolve Freebase-style WebQSP entity IDs to human-readable names."""

    max_unmapped_samples = 20

    def __init__(self, mapping_path: Path | None = None):
        self.mapping_path = mapping_path or Path(WEBQSP_ENTITY_NAMES_MAPPING_PATH)
        self.entity_names: dict[str, str] | None = None
        self.entity_name_counts: dict[str, int] | None = None
        self.reset_summary()

    def reset_summary(self) -> None:
        """Reset per-processing-run mapping counters."""
        self.total_entity_references = 0
        self.mapped_entity_references = 0
        self.disambiguated_entity_references = 0
        self.unmapped_mid_entity_references = 0
        self.mapped_mids: set[str] = set()
        self.disambiguated_mids: set[str] = set()
        self.unmapped_mids: set[str] = set()

    def resolve_entity(self, entity: str) -> str:
        """Return the readable entity name when the mapping contains the entity ID."""
        self.total_entity_references += 1
        normalized_entity = self.normalize_entity_id(entity)
        entity_names = self.get_entity_names()
        if normalized_entity in entity_names:
            self.mapped_entity_references += 1
            self.mapped_mids.add(normalized_entity)
            entity_name = entity_names[normalized_entity]
            if self.is_ambiguous_entity_name(entity_name):
                self.disambiguated_entity_references += 1
                self.disambiguated_mids.add(normalized_entity)
                return f"{entity_name} [{normalized_entity}]"

            return entity_name

        if self.is_mid_like_entity(normalized_entity):
            self.unmapped_mid_entity_references += 1
            self.unmapped_mids.add(normalized_entity)

        return entity

    def build_summary(self) -> WebQSPEntityMappingSummary:
        """Build a typed summary for logs and persisted metadata."""
        return WebQSPEntityMappingSummary(
            total_entity_references=self.total_entity_references,
            mapped_entity_references=self.mapped_entity_references,
            disambiguated_entity_references=self.disambiguated_entity_references,
            unmapped_mid_entity_references=self.unmapped_mid_entity_references,
            unique_mapped_mid_count=len(self.mapped_mids),
            unique_disambiguated_mid_count=len(self.disambiguated_mids),
            unique_unmapped_mid_count=len(self.unmapped_mids),
            unmapped_mid_samples=sorted(self.unmapped_mids)[: self.max_unmapped_samples],
        )

    @staticmethod
    def normalize_entity_id(entity: str) -> str:
        """Normalize supported WebQSP MID spellings to the mapping key format."""
        normalized_entity = entity.strip()
        if normalized_entity.startswith("/m/"):
            return f"m.{normalized_entity.removeprefix('/m/')}"

        return normalized_entity

    @staticmethod
    def is_mid_like_entity(entity: str) -> bool:
        """Return whether the entity looks like an unresolved Freebase MID."""
        return entity.startswith("m.")

    def get_entity_names(self) -> dict[str, str]:
        """Load and return the entity-name mapping on first use."""
        if self.entity_names is None:
            self.entity_names = self._load_entity_names()

        return self.entity_names

    def get_entity_name_counts(self) -> dict[str, int]:
        """Return how many mapped IDs point to each readable entity name."""
        if self.entity_name_counts is None:
            entity_name_counts: dict[str, int] = {}
            for entity_name in self.get_entity_names().values():
                entity_name_counts[entity_name] = entity_name_counts.get(entity_name, 0) + 1

            self.entity_name_counts = entity_name_counts

        return self.entity_name_counts

    def is_ambiguous_entity_name(self, entity_name: str) -> bool:
        """Return whether several MIDs map to the same readable entity name."""
        return self.get_entity_name_counts()[entity_name] > 1

    def _load_entity_names(self) -> dict[str, str]:
        raw_mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        entity_names = {
            str(entity_id): str(entity_name)
            for entity_id, entity_name in raw_mapping.items()
        }
        logger.info(
            f"Loaded WebQSP entity-name mapping: entries={len(entity_names)} "
            f"path={self.mapping_path}"
        )
        return entity_names
