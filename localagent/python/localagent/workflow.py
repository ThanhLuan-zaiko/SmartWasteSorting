from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STEP1_REQUIRED_MESSAGE = (
    "Complete Step 1 first. Run `run-all` so the dataset manifest and summary exist."
)
STEP2_REQUIRED_MESSAGE = (
    "Complete Step 2 first. Review clusters, save the review, and run "
    "`promote-cluster-labels` before using Step 3."
)
CLUSTER_REQUIRED_MESSAGE = "Run `cluster` first so cluster assignments exist in the manifest."
EMBED_REQUIRED_MESSAGE = "Run `embed` first so image similarity vectors exist."
PROMOTE_REQUIRED_MESSAGE = (
    "Review at least one cluster, save the review, then run `promote-cluster-labels`."
)


def load_dataset_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dataset summary at {summary_path} must be a JSON object.")
    return payload


def manifest_summary_complete(manifest_path: Path, summary_path: Path) -> bool:
    return manifest_path.exists() and summary_path.exists()


def embedding_artifact_exists(
    summary: dict[str, Any] | None,
    *,
    embeddings_path: Path,
) -> bool:
    if isinstance(summary, dict):
        value = summary.get("embedding_artifact_exists")
        if isinstance(value, bool):
            return value
    return embeddings_path.exists()


def cluster_ready(
    summary: dict[str, Any] | None,
    *,
    cluster_summary_path: Path,
) -> bool:
    if isinstance(summary, dict):
        summary_exists = summary.get("cluster_summary_exists")
        clustered_files = summary.get("clustered_files")
        if isinstance(summary_exists, bool) and summary_exists:
            return True
        if isinstance(clustered_files, int) and clustered_files > 0:
            return True
    return cluster_summary_path.exists()


def accepted_cluster_review_labels(summary: dict[str, Any] | None) -> int:
    if not isinstance(summary, dict):
        return 0
    source_counts = summary.get("accepted_label_source_counts")
    if not isinstance(source_counts, dict):
        return 0
    value = source_counts.get("cluster_review")
    return value if isinstance(value, int) else 0


def summary_path_from_manifest(manifest_path: Path) -> Path:
    artifact_root = manifest_path.parent.parent
    return artifact_root / "reports" / "summary.json"
