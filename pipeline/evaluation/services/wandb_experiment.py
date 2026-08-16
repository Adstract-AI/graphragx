"""Shared resumable Weights & Biases experiment lifecycle."""

from __future__ import annotations

import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from helpers.env_variables import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT
from helpers.logging_config import get_logger
from pipeline.exceptions import PipelineException

logger = get_logger(__name__)


class WandbTrackingMetadata(BaseModel):
    """Persisted W&B lineage and latest integration status."""

    status: Literal["logged", "failed", "skipped"]
    run_id: str | None = None
    run_name: str | None = None
    run_url: str | None = None
    project: str | None = None
    entity: str | None = None
    error_message: str | None = None


class WandbRunIdentifierService:
    """Allocate dataset-wide sequential names for newly created W&B runs."""

    def allocate(self, run_root: Path) -> str:
        """Create and return the next ``number_timestamp`` run identifier."""
        run_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        while True:
            run_number = max(
                (
                    self._extract_run_number(path.name)
                    for path in run_root.iterdir()
                    if path.is_dir()
                ),
                default=0,
            ) + 1
            run_name = f"{run_number}_{timestamp}"
            try:
                (run_root / run_name).mkdir(exist_ok=False)
                return run_name
            except FileExistsError:
                continue

    @staticmethod
    def _extract_run_number(name: str) -> int:
        match = re.match(r"^(\d+)_", name)
        return int(match.group(1)) if match else 0


class WandbExperimentCoordinator:
    """Own one W&B run that can be shared and resumed across pipeline stages."""

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        enabled: bool = True,
        run_root: Path | None = None,
        run_identifier_service: WandbRunIdentifierService | None = None,
    ) -> None:
        self.requested_project = project
        self.requested_entity = entity
        self.project = project or WANDB_PROJECT
        self.entity = entity if entity is not None else WANDB_ENTITY
        self.mode = mode or WANDB_MODE
        self.enabled = enabled
        self.run_root = run_root
        self.run_identifier_service = (
            run_identifier_service or WandbRunIdentifierService()
        )
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._metadata = WandbTrackingMetadata(status="skipped")
        self._metadata_paths: set[Path] = set()

    @property
    def metadata(self) -> WandbTrackingMetadata:
        """Return the latest immutable tracking metadata snapshot."""
        return self._metadata.model_copy(deep=True)

    @property
    def has_active_run(self) -> bool:
        """Return whether this coordinator currently owns an initialized run."""
        return self._run is not None

    def ensure_run(
        self,
        *,
        source_config_path: Path | None = None,
    ) -> WandbTrackingMetadata:
        """Initialize or resume the shared run using optional persisted lineage."""
        if not self.enabled:
            return self.metadata
        if self._run is not None:
            return self.metadata

        lineage = self._load_lineage(source_config_path)
        if lineage is not None:
            self._validate_lineage(lineage)
            self.project = lineage.project or self.project
            self.entity = lineage.entity if lineage.entity is not None else self.entity
        run_name = lineage.run_name if lineage is not None else None
        if lineage is None:
            if self.run_root is None:
                raise PipelineException(
                    "Creating a W&B run requires a configured run identifier root."
                )
            run_name = self.run_identifier_service.allocate(self.run_root)
            logger.info(f"Allocated W&B run identifier: {run_name}")
        try:
            self._wandb = importlib.import_module("wandb")
            init_kwargs: dict[str, Any] = {
                "project": self.project,
                "entity": self.entity,
                "mode": self.mode,
                "name": run_name,
                "job_type": "graphragx-pipeline",
            }
            if lineage is not None and lineage.run_id:
                init_kwargs.update({"id": lineage.run_id, "resume": "allow"})
            self._run = self._wandb.init(**init_kwargs)
            if hasattr(self._run, "define_metric"):
                self._run.define_metric("Training/global_step")
                self._run.define_metric(
                    "Training/loss",
                    step_metric="Training/global_step",
                )
            run_id = str(getattr(self._run, "id", "")) or None
            run_url = str(getattr(self._run, "url", "")) or None
            self._metadata = WandbTrackingMetadata(
                status="logged",
                run_id=run_id,
                run_name=run_name,
                run_url=run_url,
                project=self.project,
                entity=self.entity,
            )
        except Exception as error:
            logger.warning(f"WandB experiment initialization failed: {error}")
            self._metadata = WandbTrackingMetadata(
                status="failed",
                run_name=run_name,
                project=self.project,
                entity=self.entity,
                error_message=str(error),
            )
        return self.metadata

    def log(
        self,
        payload: dict[str, float | int | str],
        *,
        source_config_path: Path | None = None,
        step: int | None = None,
    ) -> None:
        """Best-effort log scalar stage data to the active experiment."""
        self.ensure_run(source_config_path=source_config_path)
        if self._run is None:
            return
        try:
            if step is None:
                self._run.log(payload)
            else:
                self._run.log(payload, step=step)
        except Exception as error:
            self._record_failure("WandB metric logging failed", error)

    def log_training_progress(self, payload: dict[str, float | int]) -> None:
        """Log one live training progress event using optimizer progress as step."""
        global_step = int(payload["global_step"])
        self.log(
            {
                "Training/loss": float(payload["loss"]),
                "Training/epoch": int(payload["epoch"]),
                "Training/instance": int(payload["instance"]),
                "Training/global_step": global_step,
            },
        )

    def log_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        paths: list[Path],
        source_config_path: Path | None = None,
    ) -> None:
        """Best-effort upload a collection of existing local artifact files."""
        self.ensure_run(source_config_path=source_config_path)
        if self._run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            for path in paths:
                if path.exists():
                    artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as error:
            self._record_failure("WandB artifact logging failed", error)

    def persist_metadata(self, path: Path) -> None:
        """Merge the latest W&B tracking metadata into a JSON config artifact."""
        self._metadata_paths.add(path)
        self._write_metadata(path)

    def _write_metadata(self, path: Path) -> None:
        """Write tracking metadata without changing the registered path set."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("config root must be an object")
            payload["wandb"] = self.metadata.model_dump(mode="json")
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            logger.warning(f"Could not persist WandB metadata to {path}: {error}")

    def finish(self) -> None:
        """Finish the active W&B run without changing pipeline success semantics."""
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as error:
            self._record_failure("WandB experiment finalization failed", error)
            for path in self._metadata_paths:
                self._write_metadata(path)
        finally:
            self._run = None

    def _load_lineage(self, path: Path | None) -> WandbTrackingMetadata | None:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tracking = payload.get("wandb") if isinstance(payload, dict) else None
            if not isinstance(tracking, dict) or not tracking.get("run_id"):
                return None
            return WandbTrackingMetadata.model_validate(tracking)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            logger.warning(f"Ignoring invalid WandB lineage in {path}: {error}")
            return None

    def _validate_lineage(self, lineage: WandbTrackingMetadata) -> None:
        conflicts: list[str] = []
        if self.requested_project is not None and lineage.project != self.requested_project:
            conflicts.append("project")
        if self.requested_entity is not None and lineage.entity != self.requested_entity:
            conflicts.append("entity")
        if conflicts:
            raise PipelineException(
                "WandB options conflict with persisted experiment lineage: "
                + ", ".join(conflicts)
            )

    def _record_failure(self, message: str, error: Exception) -> None:
        logger.warning(f"{message}: {error}")
        self._metadata = self._metadata.model_copy(
            update={"status": "failed", "error_message": str(error)}
        )
