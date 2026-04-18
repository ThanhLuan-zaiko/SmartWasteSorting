from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from localagent.utils import TerminalProgressBar
from localagent.vision import build_training_transforms, load_rgb_image


@dataclass(slots=True)
class EmbeddingArtifact:
    sample_ids: np.ndarray
    relative_paths: np.ndarray
    vectors: np.ndarray
    extractor: str
    image_size: int
    fallback_reason: str | None = None

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            sample_ids=self.sample_ids.astype(str),
            relative_paths=self.relative_paths.astype(str),
            vectors=self.vectors.astype(np.float32),
            extractor=np.array(self.extractor),
            image_size=np.array(self.image_size, dtype=np.int64),
            fallback_reason=np.array(self.fallback_reason or ""),
        )
        return output_path

    @classmethod
    def load(cls, input_path: Path) -> EmbeddingArtifact:
        if not input_path.exists():
            raise FileNotFoundError(f"Embedding artifact does not exist: {input_path}")
        with np.load(input_path, allow_pickle=False) as payload:
            return cls(
                sample_ids=np.asarray(payload["sample_ids"]).astype(str),
                relative_paths=np.asarray(payload["relative_paths"]).astype(str),
                vectors=np.asarray(payload["vectors"], dtype=np.float32),
                extractor=str(np.asarray(payload["extractor"]).item()),
                image_size=int(np.asarray(payload["image_size"]).item()),
                fallback_reason=(
                    str(np.asarray(payload["fallback_reason"]).item()).strip() or None
                ),
            )


@dataclass(slots=True)
class ClusterArtifact:
    assignments: np.ndarray
    distances: np.ndarray
    cluster_sizes: np.ndarray
    outliers: np.ndarray
    cluster_count: int

    def assignment_for(self, index: int) -> dict[str, int | float | bool]:
        cluster_id = int(self.assignments[index])
        return {
            "cluster_id": cluster_id,
            "cluster_distance": float(self.distances[index]),
            "cluster_size": int(self.cluster_sizes[cluster_id]),
            "is_cluster_outlier": bool(self.outliers[index]),
        }


def extract_embeddings(
    records: list[dict[str, Any]],
    *,
    image_size: int = 224,
    show_progress: bool = True,
) -> tuple[EmbeddingArtifact, dict[str, Any]]:
    if not records:
        raise ValueError("Cannot build embeddings for an empty dataset.")

    fallback_reason: str | None = None
    try:
        vectors = _extract_pretrained_embeddings(
            records,
            image_size=image_size,
            show_progress=show_progress,
        )
        extractor = "resnet18_imagenet"
    except Exception as error:
        fallback_reason = f"{type(error).__name__}: {error}"
        vectors = _extract_handcrafted_embeddings(records, show_progress=show_progress)
        extractor = "handcrafted_histogram"

    artifact = EmbeddingArtifact(
        sample_ids=np.asarray([str(record["sample_id"]) for record in records], dtype=str),
        relative_paths=np.asarray([str(record["relative_path"]) for record in records], dtype=str),
        vectors=_normalize_vectors(vectors),
        extractor=extractor,
        image_size=image_size,
        fallback_reason=fallback_reason,
    )
    summary = {
        "num_samples": int(len(records)),
        "vector_dim": int(artifact.vectors.shape[1]),
        "extractor": extractor,
        "image_size": image_size,
        "fallback_reason": fallback_reason,
    }
    return artifact, summary


def cluster_embeddings(
    vectors: np.ndarray,
    *,
    seed: int = 42,
    requested_clusters: int | None = None,
) -> tuple[ClusterArtifact, dict[str, Any]]:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embedding matrix.")

    normalized_vectors = _normalize_vectors(vectors)
    num_samples = int(normalized_vectors.shape[0])
    cluster_count = _resolve_cluster_count(num_samples, requested_clusters=requested_clusters)
    assignments, distances = _spherical_kmeans(
        normalized_vectors,
        cluster_count=cluster_count,
        seed=seed,
    )
    cluster_sizes = np.bincount(assignments, minlength=cluster_count).astype(np.int64)
    outliers = _detect_cluster_outliers(assignments, distances, cluster_count=cluster_count)
    largest_clusters = sorted(
        (
            {
                "cluster_id": int(cluster_id),
                "size": int(cluster_sizes[cluster_id]),
                "outliers": int(np.count_nonzero(outliers[assignments == cluster_id])),
            }
            for cluster_id in range(cluster_count)
        ),
        key=lambda item: (-item["size"], item["cluster_id"]),
    )[:10]
    summary = {
        "num_samples": num_samples,
        "cluster_count": cluster_count,
        "outlier_count": int(np.count_nonzero(outliers)),
        "cluster_size_stats": {
            "min": int(np.min(cluster_sizes)) if cluster_sizes.size else 0,
            "median": float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
            "max": int(np.max(cluster_sizes)) if cluster_sizes.size else 0,
        },
        "largest_clusters": largest_clusters,
    }
    artifact = ClusterArtifact(
        assignments=assignments.astype(np.int64),
        distances=distances.astype(np.float32),
        cluster_sizes=cluster_sizes,
        outliers=outliers,
        cluster_count=cluster_count,
    )
    return artifact, summary


def _extract_pretrained_embeddings(
    records: list[dict[str, Any]],
    *,
    image_size: int,
    show_progress: bool,
) -> np.ndarray:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = build_training_transforms(image_size, normalization_preset="imagenet")
    progress = TerminalProgressBar(
        total=len(records),
        description="embed dataset",
        enabled=show_progress,
    )
    vectors: list[np.ndarray] = []
    batch_tensors: list[Any] = []
    batch_size = 16

    def flush() -> None:
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu().numpy()
        for embedding in embeddings:
            vectors.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        batch_tensors.clear()

    for record in records:
        rgb_image = load_rgb_image(Path(str(record["image_path"])))
        batch_tensors.append(transform(rgb_image))
        if len(batch_tensors) >= batch_size:
            flush()
        progress.advance(postfix=str(record["file_name"]))
    flush()
    progress.close(summary=f"Embedded {len(vectors)} images with ResNet18 features")
    return np.vstack(vectors).astype(np.float32)


def _extract_handcrafted_embeddings(
    records: list[dict[str, Any]],
    *,
    show_progress: bool,
) -> np.ndarray:
    progress = TerminalProgressBar(
        total=len(records),
        description="embed fallback",
        enabled=show_progress,
    )
    vectors: list[np.ndarray] = []
    for record in records:
        rgb_image = load_rgb_image(Path(str(record["image_path"])))
        vectors.append(_handcrafted_embedding(rgb_image))
        progress.advance(postfix=str(record["file_name"]))
    progress.close(summary=f"Embedded {len(vectors)} images with handcrafted features")
    return np.vstack(vectors).astype(np.float32)


def _handcrafted_embedding(rgb_image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb_image, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    thumbnail = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumbnail = thumbnail.reshape(-1) / 255.0

    histograms: list[np.ndarray] = []
    for channel_index in range(3):
        histogram, _ = np.histogram(
            resized[:, :, channel_index],
            bins=8,
            range=(0, 256),
            density=True,
        )
        histograms.append(histogram.astype(np.float32))

    edges = cv2.Canny(gray, 60, 160)
    edge_histogram, _ = np.histogram(edges, bins=8, range=(0, 256), density=True)
    features = np.concatenate([thumbnail, *histograms, edge_histogram.astype(np.float32)])
    return features.astype(np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (vectors / norms).astype(np.float32)


def _resolve_cluster_count(
    num_samples: int,
    *,
    requested_clusters: int | None,
) -> int:
    if num_samples <= 1:
        return 1
    if requested_clusters is not None:
        return max(2, min(int(requested_clusters), num_samples))
    heuristic = int(round(math.sqrt(max(num_samples / 3.0, 4.0))))
    return max(2, min(num_samples, max(3, min(32, heuristic))))


def _spherical_kmeans(
    vectors: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_samples = int(vectors.shape[0])
    indices = rng.choice(num_samples, size=cluster_count, replace=False)
    centroids = vectors[indices].copy()
    assignments = np.zeros(num_samples, dtype=np.int64)

    for _ in range(max_iterations):
        similarities = vectors @ centroids.T
        next_assignments = np.argmax(similarities, axis=1).astype(np.int64)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_id in range(cluster_count):
            members = vectors[assignments == cluster_id]
            if members.size == 0:
                centroids[cluster_id] = vectors[int(rng.integers(0, num_samples))]
                continue
            centroid = members.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[cluster_id] = centroid if norm <= 1e-8 else centroid / norm

    final_similarities = vectors @ centroids.T
    final_assignments = np.argmax(final_similarities, axis=1).astype(np.int64)
    best_similarity = final_similarities[np.arange(num_samples), final_assignments]
    distances = (1.0 - best_similarity).astype(np.float32)
    return final_assignments, distances


def _detect_cluster_outliers(
    assignments: np.ndarray,
    distances: np.ndarray,
    *,
    cluster_count: int,
) -> np.ndarray:
    outliers = np.zeros_like(assignments, dtype=bool)
    for cluster_id in range(cluster_count):
        member_mask = assignments == cluster_id
        member_distances = distances[member_mask]
        if member_distances.size <= 3:
            continue
        median_distance = float(np.median(member_distances))
        mad = float(np.median(np.abs(member_distances - median_distance)))
        if mad > 1e-8:
            threshold = median_distance + (2.5 * mad)
        else:
            threshold = float(np.quantile(member_distances, 0.9))
        outliers[member_mask] = member_distances > threshold
    return outliers
