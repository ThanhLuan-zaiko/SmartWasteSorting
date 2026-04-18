"use client";

import Image from "next/image";

import {
  buildDatasetImageUrl,
  statusTone,
} from "@/components/dashboard/discovery/discovery-shared";
import type { ClusterReviewCluster, ClusterReviewStatus } from "@/lib/localagent";

type ClusterReviewCardProps = {
  cluster: ClusterReviewCluster;
  fieldClass: string;
  isDark: boolean;
  labelClass: string;
  onToggleSelection: (clusterId: number) => void;
  onLabelChange: (clusterId: number, nextLabel: string) => void;
  onNotesChange: (clusterId: number, nextNotes: string) => void;
  onStatusChange: (clusterId: number, nextStatus: ClusterReviewStatus) => void;
  selected: boolean;
};

export function ClusterReviewCard({
  cluster,
  fieldClass,
  isDark,
  labelClass,
  onToggleSelection,
  onLabelChange,
  onNotesChange,
  onStatusChange,
  selected,
}: ClusterReviewCardProps) {
  const majorityLabel =
    typeof cluster.current_majority_label === "string" &&
    cluster.current_majority_label.length > 0
      ? cluster.current_majority_label
      : null;

  return (
    <div
      className={[
        "rounded-[1.5rem] border p-4",
        isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-white",
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <label className="mb-3 inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-zinc-500">
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggleSelection(cluster.cluster_id)}
              className="size-4 rounded border-zinc-300"
            />
            {selected ? "Selected" : "Select cluster"}
          </label>
          <p className="text-sm font-semibold">Cluster {cluster.cluster_id}</p>
          <p className="text-xs text-zinc-500">
            {cluster.cluster_size} samples | {cluster.outlier_count} outliers
          </p>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <span
            className={[
              "rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]",
              majorityLabel
                ? isDark
                  ? "bg-emerald-950 text-emerald-300"
                  : "bg-emerald-50 text-emerald-700"
                : isDark
                  ? "bg-zinc-900 text-zinc-400"
                  : "bg-zinc-100 text-zinc-500",
            ].join(" ")}
          >
            {majorityLabel ?? "No accepted label"}
          </span>
          <span
            className={[
              "rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]",
              statusTone(cluster.status, isDark),
            ].join(" ")}
          >
            {cluster.status}
          </span>
        </div>
      </div>

      <div className="mt-4 grid gap-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-2">
            <span className={labelClass}>Cluster label</span>
            <input
              className={fieldClass}
              value={cluster.label}
              placeholder="glass"
              onChange={(event) => onLabelChange(cluster.cluster_id, event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-2">
            <span className={labelClass}>Decision</span>
            <select
              className={fieldClass}
              value={cluster.status}
              onChange={(event) =>
                onStatusChange(
                  cluster.cluster_id,
                  event.target.value as ClusterReviewStatus,
                )
              }
            >
              <option value="unlabeled">unlabeled</option>
              <option value="labeled">labeled</option>
              <option value="excluded">excluded</option>
            </select>
          </label>
        </div>

        <label className="flex flex-col gap-2">
          <span className={labelClass}>Notes</span>
          <input
            className={fieldClass}
            value={cluster.notes}
            placeholder="optional context for this cluster"
            onChange={(event) => onNotesChange(cluster.cluster_id, event.target.value)}
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-zinc-500">
        <span>Fingerprint: {cluster.representative_sample_ids}</span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
        {cluster.representatives.map((representative) => (
          <div key={representative.sample_id} className="min-w-0">
            <div
              className={[
                "aspect-square overflow-hidden rounded-[1.2rem] border",
                isDark ? "border-zinc-800 bg-zinc-900" : "border-zinc-200 bg-zinc-50",
              ].join(" ")}
            >
              {representative.relative_path ? (
                <div className="relative h-full w-full">
                  <Image
                    src={buildDatasetImageUrl(representative.relative_path)}
                    alt={representative.relative_path}
                    fill
                    unoptimized
                    sizes="(min-width: 1536px) 180px, (min-width: 1024px) 220px, 45vw"
                    className="object-cover"
                  />
                </div>
              ) : null}
            </div>
            <p className="mt-2 break-all text-[11px] text-zinc-500">
              {representative.relative_path}
            </p>
            <p className="mt-1 text-xs font-semibold">
              {representative.label}{" "}
              <span className="font-normal text-zinc-500">
                via {representative.label_source}
              </span>
            </p>
            <p className="mt-1 text-[11px] text-zinc-500">{representative.review_status}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
