"use client";

import { useState } from "react";
import { HiOutlineArrowPath, HiSparkles } from "react-icons/hi2";

import { ClusterReviewBulkActions } from "@/components/dashboard/discovery/cluster-review-bulk-actions";
import { ClusterReviewCard } from "@/components/dashboard/discovery/cluster-review-card";
import { ClusterReviewQueue } from "@/components/dashboard/discovery/cluster-review-queue";
import { EmptyState, StatusBadge } from "@/components/ui/primitives";
import type {
  ClusterReviewCluster,
  ClusterReviewResponse,
  ClusterReviewStatus,
  JobRecord,
} from "@/lib/localagent";

type ClusterReviewSectionProps = {
  activeDatasetJob: JobRecord | null;
  clusterReady: boolean;
  clusterReview: ClusterReviewResponse | null;
  clusterReviewError: string | null;
  draftClusters: ClusterReviewCluster[];
  excludedClusterCount: number;
  fieldClass: string;
  invalidLabeledClusters: number;
  isClusterReviewLoading: boolean;
  isClusterReviewSaving: boolean;
  isDark: boolean;
  isDirty: boolean;
  labelClass: string;
  labeledClusterCount: number;
  onApplyBulk: (payload: {
    label: string;
    notes: string;
    status: ClusterReviewStatus;
  }) => void;
  onApplyBulkLabelList: (labels: string[]) => void;
  onApplyMajorityLabels: () => void;
  onLabelChange: (clusterId: number, nextLabel: string) => void;
  onNotesChange: (clusterId: number, nextNotes: string) => void;
  onReload: () => Promise<void>;
  onSave: () => Promise<void>;
  onStatusChange: (clusterId: number, nextStatus: "labeled" | "unlabeled" | "excluded") => void;
  onToggleSelection: (clusterId: number) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  reviewedClusterCount: number;
  saveDisabledReason: string | null;
  selectedClusterOrder: number[];
  selectedMajorityLabelCount: number;
  selectedClusterIds: Set<number>;
};

export function ClusterReviewSection({
  activeDatasetJob,
  clusterReady,
  clusterReview,
  clusterReviewError,
  draftClusters,
  excludedClusterCount,
  fieldClass,
  invalidLabeledClusters,
  isClusterReviewLoading,
  isClusterReviewSaving,
  isDark,
  isDirty,
  labelClass,
  labeledClusterCount,
  onApplyBulk,
  onApplyBulkLabelList,
  onApplyMajorityLabels,
  onLabelChange,
  onNotesChange,
  onClearSelection,
  onReload,
  onSave,
  onSelectAll,
  onStatusChange,
  onToggleSelection,
  reviewedClusterCount,
  saveDisabledReason,
  selectedClusterOrder,
  selectedMajorityLabelCount,
  selectedClusterIds,
}: ClusterReviewSectionProps) {
  const [viewMode, setViewMode] = useState<"grid" | "queue">("grid");
  const [activeQueueClusterId, setActiveQueueClusterId] = useState<number | null>(null);

  const queueClusters =
    selectedClusterIds.size > 0
      ? draftClusters.filter((cluster) => selectedClusterIds.has(cluster.cluster_id))
      : draftClusters;
  const pendingQueueCount = queueClusters.filter(
    (cluster) => cluster.status === "unlabeled",
  ).length;
  const resolvedActiveQueueClusterId =
    activeQueueClusterId !== null &&
    queueClusters.some((cluster) => cluster.cluster_id === activeQueueClusterId)
      ? activeQueueClusterId
      : (queueClusters.find((cluster) => cluster.status === "unlabeled") ?? queueClusters[0])
          ?.cluster_id ?? null;
  const activeQueueCluster =
    queueClusters.find((cluster) => cluster.cluster_id === resolvedActiveQueueClusterId) ?? null;
  const activeQueueIndex = activeQueueCluster
    ? queueClusters.findIndex((cluster) => cluster.cluster_id === activeQueueCluster.cluster_id)
    : -1;
  const queueScopeLabel =
    selectedClusterIds.size > 0
      ? `${selectedClusterIds.size} selected cluster${
          selectedClusterIds.size === 1 ? "" : "s"
        }`
      : `${draftClusters.length} cluster${draftClusters.length === 1 ? "" : "s"}`;

  function moveQueue(offset: number) {
    if (queueClusters.length === 0) {
      return;
    }
    const currentIndex = activeQueueIndex >= 0 ? activeQueueIndex : 0;
    const nextIndex = (currentIndex + offset + queueClusters.length) % queueClusters.length;
    setActiveQueueClusterId(queueClusters[nextIndex].cluster_id);
  }

  function goToNextPending() {
    if (queueClusters.length === 0 || pendingQueueCount === 0) {
      return;
    }
    const startIndex = activeQueueIndex >= 0 ? activeQueueIndex + 1 : 0;
    for (let offset = 0; offset < queueClusters.length; offset += 1) {
      const nextIndex = (startIndex + offset) % queueClusters.length;
      if (queueClusters[nextIndex].status === "unlabeled") {
        setActiveQueueClusterId(queueClusters[nextIndex].cluster_id);
        return;
      }
    }
  }

  return (
    <div className="mt-7">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold">Representative clusters</p>
          {clusterReview ? (
            <p className="text-sm text-zinc-500">
              {clusterReview.cluster_count} clusters loaded from{" "}
              <span className="font-medium">{clusterReview.review_file}</span>. Draft review:{" "}
              <span className="font-medium">
                {labeledClusterCount} labeled, {excludedClusterCount} excluded,{" "}
                {draftClusters.length - reviewedClusterCount} pending
              </span>
              .
              {clusterReview.stale_reset_count > 0
                ? ` Reset ${clusterReview.stale_reset_count} stale cluster reviews while loading.`
                : ""}
            </p>
          ) : (
            <p className="text-sm text-zinc-500">
              Load cluster review state from the manifest and draft review CSV.
            </p>
          )}
        </div>
        <div className="flex flex-wrap gap-3">
          {clusterReview && draftClusters.length > 0 ? (
            <div
              className={[
                "inline-flex rounded-full border p-1",
                isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-zinc-100",
              ].join(" ")}
            >
              {(["grid", "queue"] as const).map((mode) => {
                const isActive = viewMode === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setViewMode(mode)}
                    className={[
                      "rounded-full px-4 py-2 text-sm font-semibold transition",
                      isActive
                        ? isDark
                          ? "bg-white text-black"
                          : "bg-zinc-950 text-white"
                        : isDark
                          ? "text-zinc-300 hover:bg-zinc-900"
                          : "text-zinc-700 hover:bg-white",
                    ].join(" ")}
                  >
                    {mode === "grid" ? "Grid view" : "Queue mode"}
                  </button>
                );
              })}
            </div>
          ) : null}
          <button
            type="button"
            onClick={() => void onReload()}
            disabled={isClusterReviewLoading || isClusterReviewSaving}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark
                ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              isClusterReviewLoading || isClusterReviewSaving ? "opacity-60" : "",
            ].join(" ")}
          >
            <HiOutlineArrowPath />
            {isClusterReviewLoading ? "Reloading..." : "Reload review"}
          </button>
          <button
            type="button"
            title={saveDisabledReason ?? undefined}
            onClick={() => void onSave()}
            disabled={
              !clusterReview ||
              !isDirty ||
              isClusterReviewSaving ||
              isClusterReviewLoading ||
              saveDisabledReason !== null
            }
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-white text-black hover:bg-zinc-200" : "bg-zinc-950 text-white hover:bg-zinc-800",
              !clusterReview ||
              !isDirty ||
              isClusterReviewSaving ||
              isClusterReviewLoading ||
              saveDisabledReason !== null
                ? "opacity-60"
                : "",
            ].join(" ")}
          >
            <HiSparkles />
            {isClusterReviewSaving ? "Saving review..." : "Save review"}
          </button>
        </div>
      </div>

      {isDirty ? (
        <p className="mt-3 text-xs text-amber-500">
          Draft changes are unsaved. `promote-cluster-labels` reads the saved review file, not
          the in-memory edits shown here.
        </p>
      ) : null}
      {invalidLabeledClusters > 0 ? (
        <p className="mt-2 text-xs text-rose-500">
          {invalidLabeledClusters} labeled cluster
          {invalidLabeledClusters === 1 ? " is" : "s are"} missing a usable label.
        </p>
      ) : null}
      {selectedClusterIds.size > 0 && !isDirty && reviewedClusterCount === 0 ? (
        <p className="mt-2 text-xs text-amber-500">
          {selectedClusterIds.size} cluster{selectedClusterIds.size === 1 ? "" : "s"} selected.
          Selection only marks targets for the bulk editor. Click{" "}
          <span className="font-semibold">
            Apply to {selectedClusterIds.size} cluster
            {selectedClusterIds.size === 1 ? "" : "s"}
          </span>{" "}
          to change their decision, then save the review.
        </p>
      ) : null}
      {clusterReview && draftClusters.length > 0 ? (
        <p className="mt-2 text-xs text-zinc-500">
          You do not need to label all {draftClusters.length} clusters here. In practice, label a
          few representative clusters across at least 2 classes, save the review, promote those
          labels, then use Step 3 to expand coverage.
        </p>
      ) : null}
      {clusterReview && draftClusters.length > 0 && viewMode === "queue" ? (
        <p className="mt-2 text-xs text-zinc-500">
          Queue mode reviews one cluster at a time and uses the current selection as its scope.
          Clear selection to queue across every cluster, or keep a subset selected to work through
          only that slice.
        </p>
      ) : null}
      {clusterReviewError ? (
        <div
          className={[
            "mt-4 rounded-[1.2rem] border px-4 py-3 text-sm leading-6",
            isDark ? "border-rose-900 bg-rose-950/40 text-rose-200" : "border-rose-200 bg-rose-50 text-rose-700",
          ].join(" ")}
        >
          {clusterReviewError}
        </div>
      ) : null}

      {isClusterReviewLoading && !clusterReview ? (
        <div className="mt-4">
          <EmptyState
            title="Loading cluster review"
            description="Fetching the current cluster draft state from the localagent backend."
          />
        </div>
      ) : null}

      {!clusterReview && !isClusterReviewLoading && !clusterReady ? (
        <div className="mt-4">
          <EmptyState
            title="Cluster review unavailable"
            description="Run `embed` and `cluster` first so the manifest contains cluster assignments to review."
          />
        </div>
      ) : null}

      {clusterReview && draftClusters.length > 0 ? (
        <ClusterReviewBulkActions
          fieldClass={fieldClass}
          isDark={isDark}
          labelClass={labelClass}
          onApply={onApplyBulk}
          onApplyLabelList={onApplyBulkLabelList}
          onApplyMajorityLabels={onApplyMajorityLabels}
          onClearSelection={onClearSelection}
          onSelectAll={onSelectAll}
          selectedClusterOrder={selectedClusterOrder}
          selectedMajorityLabelCount={selectedMajorityLabelCount}
          selectedCount={selectedClusterIds.size}
          totalCount={draftClusters.length}
        />
      ) : null}

      {clusterReview && draftClusters.length > 0 ? (
        viewMode === "queue" && activeQueueCluster ? (
          <ClusterReviewQueue
            activeCluster={activeQueueCluster}
            currentIndex={activeQueueIndex}
            fieldClass={fieldClass}
            isDark={isDark}
            labelClass={labelClass}
            onGoNext={() => moveQueue(1)}
            onGoNextPending={goToNextPending}
            onGoPrevious={() => moveQueue(-1)}
            onJumpToCluster={setActiveQueueClusterId}
            onLabelChange={onLabelChange}
            onNotesChange={onNotesChange}
            onStatusChange={onStatusChange}
            onToggleSelection={onToggleSelection}
            pendingCount={pendingQueueCount}
            queueClusters={queueClusters}
            queueScopeLabel={queueScopeLabel}
            selected={selectedClusterIds.has(activeQueueCluster.cluster_id)}
            totalCount={queueClusters.length}
          />
        ) : (
          <div className="mt-4 grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
            {draftClusters.map((cluster) => (
              <ClusterReviewCard
                key={`cluster-${cluster.cluster_id}`}
                cluster={cluster}
                fieldClass={fieldClass}
                isDark={isDark}
                labelClass={labelClass}
                onToggleSelection={onToggleSelection}
                onLabelChange={onLabelChange}
                onNotesChange={onNotesChange}
                onStatusChange={onStatusChange}
                selected={selectedClusterIds.has(cluster.cluster_id)}
              />
            ))}
          </div>
        )
      ) : null}

      {clusterReview && draftClusters.length === 0 && !isClusterReviewLoading ? (
        <div className="mt-4">
          <EmptyState
            title="No clusters available"
            description="The manifest loaded successfully, but it does not contain any clustered rows yet."
          />
        </div>
      ) : null}

      {activeDatasetJob ? (
        <div className="mt-4 flex items-center gap-3">
          <StatusBadge status={activeDatasetJob.status} />
          <p className="text-sm text-zinc-500">
            Review saves are locked while dataset job{" "}
            <span className="font-medium">{activeDatasetJob.job_id}</span> is active.
          </p>
        </div>
      ) : null}
    </div>
  );
}
