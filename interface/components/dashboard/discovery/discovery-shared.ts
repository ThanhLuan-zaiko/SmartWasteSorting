import { API_PREFIX, type ClusterReviewCluster, type ClusterReviewStatus } from "@/lib/localagent";

export const EMBEDDING_REQUIRED_ACTIONS = new Set([
  "cluster",
  "export-cluster-review",
  "promote-cluster-labels",
]);

export const CLUSTER_REQUIRED_ACTIONS = new Set([
  "export-cluster-review",
  "promote-cluster-labels",
]);

export function buildDatasetImageUrl(relativePath: string): string {
  return `${API_PREFIX}/dataset/image?relative_path=${encodeURIComponent(relativePath)}`;
}

export function cloneReviewClusters(clusters: ClusterReviewCluster[]): ClusterReviewCluster[] {
  return clusters.map((cluster) => ({
    ...cluster,
    representatives: cluster.representatives.map((representative) => ({ ...representative })),
  }));
}

export function statusTone(status: ClusterReviewStatus, isDark: boolean): string {
  if (status === "labeled") {
    return isDark
      ? "border-emerald-800 bg-emerald-950 text-emerald-300"
      : "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (status === "excluded") {
    return isDark
      ? "border-rose-900 bg-rose-950 text-rose-300"
      : "border-rose-200 bg-rose-50 text-rose-700";
  }
  return isDark
    ? "border-zinc-800 bg-zinc-900 text-zinc-400"
    : "border-zinc-200 bg-zinc-100 text-zinc-500";
}
