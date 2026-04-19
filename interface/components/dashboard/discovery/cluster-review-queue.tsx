"use client";

import {
  HiOutlineChevronLeft,
  HiOutlineChevronRight,
} from "react-icons/hi2";

import { ClusterReviewCard } from "@/components/dashboard/discovery/cluster-review-card";
import { statusTone } from "@/components/dashboard/discovery/discovery-shared";
import type { ClusterReviewCluster, ClusterReviewStatus } from "@/lib/localagent";

type ClusterReviewQueueProps = {
  activeCluster: ClusterReviewCluster;
  currentIndex: number;
  fieldClass: string;
  isDark: boolean;
  labelClass: string;
  onGoNext: () => void;
  onGoNextPending: () => void;
  onGoPrevious: () => void;
  onJumpToCluster: (clusterId: number) => void;
  onLabelChange: (clusterId: number, nextLabel: string) => void;
  onNotesChange: (clusterId: number, nextNotes: string) => void;
  onStatusChange: (clusterId: number, nextStatus: ClusterReviewStatus) => void;
  onToggleSelection: (clusterId: number) => void;
  pendingCount: number;
  queueClusters: ClusterReviewCluster[];
  queueScopeLabel: string;
  selected: boolean;
  totalCount: number;
};

export function ClusterReviewQueue({
  activeCluster,
  currentIndex,
  fieldClass,
  isDark,
  labelClass,
  onGoNext,
  onGoNextPending,
  onGoPrevious,
  onJumpToCluster,
  onLabelChange,
  onNotesChange,
  onStatusChange,
  onToggleSelection,
  pendingCount,
  queueClusters,
  queueScopeLabel,
  selected,
  totalCount,
}: ClusterReviewQueueProps) {
  const progressPercent =
    totalCount > 0 ? Math.min(100, ((currentIndex + 1) / totalCount) * 100) : 0;

  return (
    <div
      className={[
        "mt-4 rounded-[1.5rem] border p-4",
        isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
      ].join(" ")}
    >
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <p className="text-sm font-semibold">Cluster Queue Mode</p>
          <p className="text-sm text-zinc-500">
            Reviewing cluster {currentIndex + 1} of {totalCount} from {queueScopeLabel}.{" "}
            {pendingCount} pending cluster{pendingCount === 1 ? "" : "s"} remain in this queue.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onGoPrevious}
            disabled={totalCount <= 1}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800" : "bg-white text-zinc-800 hover:bg-zinc-100",
              totalCount <= 1 ? "opacity-60" : "",
            ].join(" ")}
          >
            <HiOutlineChevronLeft />
            Previous
          </button>
          <button
            type="button"
            onClick={onGoNextPending}
            disabled={pendingCount === 0}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800" : "bg-white text-zinc-800 hover:bg-zinc-100",
              pendingCount === 0 ? "opacity-60" : "",
            ].join(" ")}
          >
            Next pending
          </button>
          <button
            type="button"
            onClick={onGoNext}
            disabled={totalCount <= 1}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-white text-black hover:bg-zinc-200" : "bg-zinc-950 text-white hover:bg-zinc-800",
              totalCount <= 1 ? "opacity-60" : "",
            ].join(" ")}
          >
            Next
            <HiOutlineChevronRight />
          </button>
        </div>
      </div>

      <div className="mt-4">
        <div
          className={[
            "h-2 overflow-hidden rounded-full",
            isDark ? "bg-zinc-900" : "bg-white",
          ].join(" ")}
        >
          <div
            className={isDark ? "h-full bg-white" : "h-full bg-zinc-950"}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex gap-2 overflow-x-auto pb-1">
        {queueClusters.map((cluster, index) => {
          const isActive = cluster.cluster_id === activeCluster.cluster_id;
          return (
            <button
              key={`queue-cluster-${cluster.cluster_id}`}
              type="button"
              onClick={() => onJumpToCluster(cluster.cluster_id)}
              className={[
                "min-w-28 rounded-[1.1rem] border px-3 py-3 text-left transition",
                isActive
                  ? isDark
                    ? "border-white bg-white text-black"
                    : "border-zinc-950 bg-zinc-950 text-white"
                  : isDark
                    ? "border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                    : "border-zinc-200 bg-white text-zinc-800 hover:bg-zinc-100",
              ].join(" ")}
            >
              <p className="text-xs font-semibold uppercase tracking-[0.18em]">
                {index + 1}
              </p>
              <p className="mt-1 text-sm font-semibold">Cluster {cluster.cluster_id}</p>
              <p className="mt-1 text-xs opacity-70">
                {cluster.cluster_size} samples
              </p>
              <span
                className={[
                  "mt-2 inline-flex rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]",
                  statusTone(cluster.status, isDark),
                ].join(" ")}
              >
                {cluster.status}
              </span>
            </button>
          );
        })}
      </div>

      <ClusterReviewCard
        cluster={activeCluster}
        fieldClass={fieldClass}
        isDark={isDark}
        labelClass={labelClass}
        onToggleSelection={onToggleSelection}
        onLabelChange={onLabelChange}
        onNotesChange={onNotesChange}
        onStatusChange={onStatusChange}
        selected={selected}
      />
    </div>
  );
}
