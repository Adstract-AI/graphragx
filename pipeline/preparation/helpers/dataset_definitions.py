"""Single source of truth for built-in dataset definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class DatasetDefinition(BaseModel):
    """Typed definition of a built-in pipeline dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Stable identifier of the dataset.")
    display_name: str = Field(..., description="Human-readable dataset name.")
    dataset_family: str = Field(..., description="High-level dataset family.")
    task_domain: str = Field(..., description="Reasoning task or domain classification.")
    description: str = Field(..., description="Short dataset description.")
    supported: bool = Field(..., description="Whether the dataset is currently supported.")


class DatasetLoaderDefinition(BaseModel):
    """Typed loader configuration for a built-in pipeline dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Stable dataset identifier.")
    hugging_face_dataset_name: str = Field(
        ...,
        description="Hugging Face dataset repository name.",
    )
    cache_root: Path = Field(..., description="Local cache directory for downloaded data.")

WEBQSP_DATASET_ID: Final[str] = "WebQSP"
DATASET_CACHE_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data"
)

PIPELINE_DATASETS: Final[dict[str, DatasetDefinition]] = {
    WEBQSP_DATASET_ID: DatasetDefinition(
        dataset_id=WEBQSP_DATASET_ID,
        display_name="WebQSP",
        dataset_family="question_answering",
        task_domain="knowledge_graph_question_answering",
        description=(
            "A question-answering benchmark where each example contains a "
            "question, topic entities, gold answer entities, and a local graph."
        ),
        supported=True,
    ),
}

DATASET_LOADERS: Final[dict[str, DatasetLoaderDefinition]] = {
    WEBQSP_DATASET_ID: DatasetLoaderDefinition(
        dataset_id=WEBQSP_DATASET_ID,
        hugging_face_dataset_name="ml1996/webqsp",
        cache_root=DATASET_CACHE_ROOT / "webqsp",
    ),
}
