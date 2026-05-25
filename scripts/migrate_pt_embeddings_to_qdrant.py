"""One-time migration from legacy .pt embedding caches into Qdrant."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "webqsp"
DEFAULT_MODEL_ID = "text-embedding-3-small"

logger = logging.getLogger("migrate_pt_embeddings_to_qdrant")


def setup_logging() -> None:
    """Configure script-local logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def invert_vocabulary(vocabulary: dict[str, int]) -> dict[int, str]:
    """Return an id-to-text vocabulary mapping."""
    return {int(text_id): text for text, text_id in vocabulary.items()}


def load_json_mapping(path: Path) -> dict[str, int]:
    """Load a persisted text vocabulary."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required vocabulary file not found: {path}. "
            "Run the WebQSP local graph preparation step first, or pass "
            "--data-root pointing to the WebQSP data directory."
        )
    logger.info("Loading vocabulary: path=%s", path)
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Loaded vocabulary: path=%s entries=%s", path, len(vocabulary))
    return vocabulary


def load_pt_embeddings(path: Path) -> dict[int, list[float]]:
    """Load a legacy torch embedding dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Legacy embedding file not found: {path}")

    import torch

    logger.info("Loading legacy .pt embeddings: path=%s", path)
    loaded = torch.load(path, weights_only=False)
    embeddings = {int(text_id): list(vector) for text_id, vector in loaded.items()}
    logger.info(
        "Loaded legacy .pt embeddings: path=%s entries=%s",
        path,
        len(embeddings),
    )
    return embeddings


def migrate_kind(
    service: WebQSPEmbeddingCacheService,
    cache: TextEmbeddingCache,
    legacy_embeddings: dict[int, list[float]],
    id_to_text: dict[int, str],
    preprocess: bool,
) -> dict[str, int]:
    """Migrate one cache kind into Qdrant."""
    logger.info(
        "Preparing migration records: kind=%s legacy_embeddings=%s vocabulary=%s",
        cache.cache_kind,
        len(legacy_embeddings),
        len(id_to_text),
    )
    records: list[tuple[str, str, list[float]]] = []
    skipped_missing_text = 0

    for text_id, vector in legacy_embeddings.items():
        original_text = id_to_text.get(text_id)
        if original_text is None:
            skipped_missing_text += 1
            continue

        embedding_input = (
            service.preprocess_relation_text(original_text) if preprocess else original_text
        )
        records.append((original_text, embedding_input, vector))

    original_texts = [record[0] for record in records]
    logger.info(
        "Checking existing Qdrant points: kind=%s collection=%s records=%s",
        cache.cache_kind,
        cache.collection_name,
        len(original_texts),
    )
    existing_ids = service._existing_point_ids(cache, original_texts)
    missing_records = [
        record
        for record in records
        if service.point_id(cache=cache, text=record[0]) not in existing_ids
    ]
    logger.info(
        "Qdrant point check finished: kind=%s existing=%s missing=%s skipped_missing_text=%s",
        cache.cache_kind,
        len(existing_ids),
        len(missing_records),
        skipped_missing_text,
    )

    if missing_records:
        logger.info(
            "Upserting Qdrant embeddings: kind=%s collection=%s count=%s",
            cache.cache_kind,
            cache.collection_name,
            len(missing_records),
        )
        service.upsert_embedding_records(
            cache=cache,
            records=missing_records,
        )
        logger.info(
            "Finished Qdrant upsert: kind=%s collection=%s count=%s",
            cache.cache_kind,
            cache.collection_name,
            len(missing_records),
        )
    else:
        logger.info(
            "No Qdrant upsert needed: kind=%s all records already present",
            cache.cache_kind,
        )

    return {
        "legacy_embeddings": len(legacy_embeddings),
        "matched_texts": len(original_texts),
        "already_present": len(existing_ids),
        "migrated": len(missing_records),
        "skipped_missing_text": skipped_missing_text,
    }


def migrate_embeddings(
    data_root: Path,
    model_id: str,
    dataset_id: str,
    service: WebQSPEmbeddingCacheService | None = None,
) -> dict[str, dict[str, int]]:
    """Migrate all legacy WebQSP embedding files for one embedding model."""
    service = service or WebQSPEmbeddingCacheService()
    data_root = data_root.expanduser().resolve()
    processed_root = data_root / "processed"
    legacy_root = data_root / "embeddings" / model_id
    logger.info(
        "Starting legacy embedding migration: dataset=%s model=%s data_root=%s legacy_root=%s",
        dataset_id,
        model_id,
        data_root,
        legacy_root,
    )

    nodes_vocab = load_json_mapping(processed_root / "nodes.json")
    relations_vocab = load_json_mapping(processed_root / "relations.json")
    questions_vocab = load_json_mapping(processed_root / "questions.json")

    cache_specs = [
        (
            "nodes",
            service.load_node_cache(
                cache_root=data_root,
                model_id=model_id,
                vocabulary=nodes_vocab,
                dataset_id=dataset_id,
            ),
            legacy_root / "nodes.pt",
            invert_vocabulary(nodes_vocab),
            False,
        ),
        (
            "relations",
            service.load_relation_cache(
                cache_root=data_root,
                model_id=model_id,
                vocabulary=relations_vocab,
                dataset_id=dataset_id,
            ),
            legacy_root / "relations.pt",
            invert_vocabulary(relations_vocab),
            True,
        ),
        (
            "questions",
            service.load_question_cache(
                cache_root=data_root,
                model_id=model_id,
                vocabulary=questions_vocab,
                dataset_id=dataset_id,
            ),
            legacy_root / "questions.pt",
            invert_vocabulary(questions_vocab),
            False,
        ),
    ]

    results: dict[str, dict[str, int]] = {}
    for cache_kind, cache, embedding_path, id_to_text, preprocess in cache_specs:
        logger.info(
            "Starting cache-kind migration: kind=%s collection=%s legacy_path=%s",
            cache_kind,
            cache.collection_name,
            embedding_path,
        )
        if not embedding_path.exists():
            logger.warning(
                "Skipping cache-kind migration because legacy file is missing: "
                "kind=%s path=%s",
                cache_kind,
                embedding_path,
            )
            results[cache_kind] = {
                "legacy_embeddings": 0,
                "matched_texts": 0,
                "already_present": 0,
                "migrated": 0,
                "skipped_missing_text": 0,
            }
            continue

        results[cache_kind] = migrate_kind(
            service=service,
            cache=cache,
            legacy_embeddings=load_pt_embeddings(embedding_path),
            id_to_text=id_to_text,
            preprocess=preprocess,
        )
        logger.info(
            "Finished cache-kind migration: kind=%s result=%s",
            cache_kind,
            results[cache_kind],
        )

    logger.info("Finished legacy embedding migration: results=%s", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate legacy WebQSP .pt embedding caches into Qdrant."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "Path to the WebQSP data directory. Defaults to the project-root "
            "data/webqsp directory, regardless of the current working directory."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-id", default="WebQSP")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    results = migrate_embeddings(
        data_root=Path(args.data_root),
        model_id=args.model_id,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
