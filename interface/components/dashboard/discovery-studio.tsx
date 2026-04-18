"use client";

import { useEffect, useMemo, useState } from "react";

import { DiscoveryControls } from "@/components/dashboard/discovery/discovery-controls";
import { DiscoveryOverview } from "@/components/dashboard/discovery/discovery-overview";
import {
  CLUSTER_REQUIRED_ACTIONS,
  cloneReviewClusters,
  EMBEDDING_REQUIRED_ACTIONS,
} from "@/components/dashboard/discovery/discovery-shared";
import { ClusterReviewSection } from "@/components/dashboard/discovery/cluster-review-section";
import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { Panel } from "@/components/ui/primitives";
import {
  asObject,
  isActiveJobStatus,
  type ClusterReviewCluster,
  type ClusterReviewStatus,
} from "@/lib/localagent";

export function DiscoveryStudio() {
  const { isDark } = useThemeMode();
  const {
    runDetail,
    jobs,
    pipelineForm,
    pipelineCatalog,
    isSubmitting,
    clusterReview,
    clusterReviewError,
    isClusterReviewLoading,
    isClusterReviewSaving,
    setPipelineField,
    reloadClusterReview,
    saveClusterReview,
    submitPipeline,
  } = useLocalAgent();

  const [draftClusters, setDraftClusters] = useState<ClusterReviewCluster[]>([]);
  const [isDirty, setIsDirty] = useState(false);
  const [selectedClusterIds, setSelectedClusterIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    setDraftClusters(clusterReview ? cloneReviewClusters(clusterReview.clusters) : []);
    setIsDirty(false);
    setSelectedClusterIds(new Set());
  }, [clusterReview]);

  const fieldClass = [
    "min-h-12 rounded-2xl border px-4 py-3 text-sm outline-none transition",
    isDark
      ? "border-zinc-800 bg-zinc-950 text-zinc-100 placeholder:text-zinc-500"
      : "border-zinc-200 bg-zinc-50 text-zinc-900 placeholder:text-zinc-400",
  ].join(" ");
  const labelClass = "text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500";

  const dashboardSummary = asObject(runDetail?.dashboard_summary);
  const status = asObject(dashboardSummary?.status);
  const datasetSummary = asObject(dashboardSummary?.dataset_summary);
  const reviewStatusCounts = asObject(datasetSummary?.review_status_counts);
  const trainableLabelCounts = asObject(datasetSummary?.trainable_label_counts);
  const acceptedLabelSourceCounts = asObject(datasetSummary?.accepted_label_source_counts);
  const datasetReady = status?.dataset_ready === true;
  const embeddingExists = datasetSummary?.embedding_artifact_exists === true;
  const effectiveTrainingMode =
    typeof datasetSummary?.effective_training_mode === "string"
      ? datasetSummary.effective_training_mode
      : "weak_inferred";
  const clusteredFiles =
    typeof datasetSummary?.clustered_files === "number" ? datasetSummary.clustered_files : 0;
  const clusterOutliers =
    typeof datasetSummary?.cluster_outlier_files === "number"
      ? datasetSummary.cluster_outlier_files
      : 0;
  const clusterReady = datasetSummary?.cluster_summary_exists === true || clusteredFiles > 0;
  const acceptedSourcesSummary = Object.entries(acceptedLabelSourceCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");
  const trainableLabelsSummary = Object.entries(trainableLabelCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");
  const manifestReviewSummary = Object.entries(reviewStatusCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");

  const activeDatasetJob = useMemo(
    () =>
      jobs.find(
        (job) => job.job_type === "dataset_pipeline" && isActiveJobStatus(job.status),
      ) ?? null,
    [jobs],
  );
  const reviewedClusterCount = draftClusters.filter(
    (cluster) => cluster.status !== "unlabeled",
  ).length;
  const labeledClusterCount = draftClusters.filter(
    (cluster) => cluster.status === "labeled",
  ).length;
  const excludedClusterCount = draftClusters.filter(
    (cluster) => cluster.status === "excluded",
  ).length;
  const invalidLabeledClusters = draftClusters.filter(
    (cluster) => cluster.status === "labeled" && !cluster.label.trim(),
  ).length;

  function updateCluster(
    clusterId: number,
    updater: (cluster: ClusterReviewCluster) => ClusterReviewCluster,
  ) {
    setDraftClusters((current) =>
      current.map((cluster) =>
        cluster.cluster_id === clusterId ? updater(cluster) : cluster,
      ),
    );
    setIsDirty(true);
  }

  function toggleClusterSelection(clusterId: number) {
    setSelectedClusterIds((current) => {
      const next = new Set(current);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
  }

  function selectAllClusters() {
    setSelectedClusterIds(new Set(draftClusters.map((cluster) => cluster.cluster_id)));
  }

  function clearClusterSelection() {
    setSelectedClusterIds(new Set());
  }

  function handleStatusChange(clusterId: number, nextStatus: ClusterReviewStatus) {
    updateCluster(clusterId, (cluster) => ({
      ...cluster,
      status: nextStatus,
      label: nextStatus === "labeled" ? cluster.label : "",
    }));
  }

  function handleLabelChange(clusterId: number, nextLabel: string) {
    updateCluster(clusterId, (cluster) => ({
      ...cluster,
      label: nextLabel,
      status:
        nextLabel.trim() && cluster.status === "unlabeled" ? "labeled" : cluster.status,
    }));
  }

  function handleNotesChange(clusterId: number, nextNotes: string) {
    updateCluster(clusterId, (cluster) => ({
      ...cluster,
      notes: nextNotes,
    }));
  }

  function handleApplyBulkReview(payload: {
    label: string;
    notes: string;
    status: ClusterReviewStatus;
  }) {
    if (selectedClusterIds.size === 0) {
      return;
    }
    const normalizedLabel = payload.status === "labeled" ? payload.label : "";
    setDraftClusters((current) =>
      current.map((cluster) =>
        selectedClusterIds.has(cluster.cluster_id)
          ? {
              ...cluster,
              label: normalizedLabel,
              notes: payload.notes,
              status: payload.status,
            }
          : cluster,
      ),
    );
    setIsDirty(true);
  }

  async function handleSaveReview() {
    if (activeDatasetJob || invalidLabeledClusters > 0) {
      return;
    }
    await saveClusterReview({
      review_file: pipelineForm.review_file,
      clusters: draftClusters.map((cluster) => ({
        cluster_id: cluster.cluster_id,
        cluster_size: cluster.cluster_size,
        outlier_count: cluster.outlier_count,
        representative_sample_ids: cluster.representative_sample_ids,
        representative_paths: cluster.representative_paths,
        label: cluster.label,
        status: cluster.status,
        notes: cluster.notes,
      })),
    });
  }

  function actionBlockReason(command: string): string | null {
    if (!datasetReady) {
      return "Run the dataset pipeline first so the manifest exists before discovery steps.";
    }
    if (EMBEDDING_REQUIRED_ACTIONS.has(command) && !embeddingExists) {
      return "Run `embed` first to build image similarity vectors.";
    }
    if (CLUSTER_REQUIRED_ACTIONS.has(command) && !clusterReady) {
      return "Run `cluster` first so cluster assignments exist in the manifest.";
    }
    if (command === "promote-cluster-labels" && isClusterReviewLoading) {
      return "Wait for cluster review state to finish loading.";
    }
    if (command === "promote-cluster-labels" && isDirty) {
      return "Save the cluster review draft before promoting labels into the manifest.";
    }
    if (command === "promote-cluster-labels" && reviewedClusterCount === 0) {
      return "Review at least one cluster below first: set Decision to labeled or excluded, click Save review, then promote labels into the manifest.";
    }
    return null;
  }

  const saveDisabledReason = activeDatasetJob
    ? "Wait for the active dataset pipeline job to finish before saving review changes."
    : invalidLabeledClusters > 0
      ? "Every cluster marked labeled needs a non-empty label."
      : null;

  return (
    <Panel
      title="Step 2: Discovery workflow"
      description="Use embeddings and clusters to review visually similar images together, save draft decisions straight to the cluster review artifact, then promote accepted labels back into the manifest."
    >
      <DiscoveryOverview
        acceptedSourcesSummary={acceptedSourcesSummary}
        clusterOutliers={clusterOutliers}
        clusteredFiles={clusteredFiles}
        effectiveTrainingMode={effectiveTrainingMode}
        isDark={isDark}
        manifestReviewSummary={manifestReviewSummary}
        trainableLabelsSummary={trainableLabelsSummary}
      />

      <DiscoveryControls
        actionBlockReason={actionBlockReason}
        fieldClass={fieldClass}
        isDark={isDark}
        isSubmitting={isSubmitting}
        labelClass={labelClass}
        pipelineCatalog={pipelineCatalog}
        pipelineForm={pipelineForm}
        setPipelineField={setPipelineField}
        submitPipeline={submitPipeline}
      />

      <ClusterReviewSection
        activeDatasetJob={activeDatasetJob}
        clusterReady={clusterReady}
        clusterReview={clusterReview}
        clusterReviewError={clusterReviewError}
        draftClusters={draftClusters}
        excludedClusterCount={excludedClusterCount}
        fieldClass={fieldClass}
        invalidLabeledClusters={invalidLabeledClusters}
        isClusterReviewLoading={isClusterReviewLoading}
        isClusterReviewSaving={isClusterReviewSaving}
        isDark={isDark}
        isDirty={isDirty}
        labelClass={labelClass}
        labeledClusterCount={labeledClusterCount}
        onApplyBulk={handleApplyBulkReview}
        onLabelChange={handleLabelChange}
        onNotesChange={handleNotesChange}
        onClearSelection={clearClusterSelection}
        onReload={reloadClusterReview}
        onSave={handleSaveReview}
        onSelectAll={selectAllClusters}
        onStatusChange={handleStatusChange}
        onToggleSelection={toggleClusterSelection}
        reviewedClusterCount={reviewedClusterCount}
        saveDisabledReason={saveDisabledReason}
        selectedClusterIds={selectedClusterIds}
      />
    </Panel>
  );
}
