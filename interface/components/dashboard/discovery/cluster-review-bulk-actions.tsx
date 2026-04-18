"use client";

import { useState } from "react";
import { HiMiniQueueList } from "react-icons/hi2";

import type { ClusterReviewStatus } from "@/lib/localagent";

type ClusterReviewBulkActionsProps = {
  fieldClass: string;
  isDark: boolean;
  labelClass: string;
  onApply: (payload: {
    label: string;
    notes: string;
    status: ClusterReviewStatus;
  }) => void;
  onClearSelection: () => void;
  onSelectAll: () => void;
  selectedCount: number;
  totalCount: number;
};

export function ClusterReviewBulkActions({
  fieldClass,
  isDark,
  labelClass,
  onApply,
  onClearSelection,
  onSelectAll,
  selectedCount,
  totalCount,
}: ClusterReviewBulkActionsProps) {
  const [bulkLabel, setBulkLabel] = useState("");
  const [bulkStatus, setBulkStatus] = useState<ClusterReviewStatus>("labeled");
  const [bulkNotes, setBulkNotes] = useState("");

  const applyDisabled =
    selectedCount === 0 || (bulkStatus === "labeled" && !bulkLabel.trim());

  function handleApply() {
    if (applyDisabled) {
      return;
    }
    onApply({
      label: bulkLabel,
      notes: bulkNotes,
      status: bulkStatus,
    });
  }

  return (
    <div
      className={[
        "mt-4 rounded-[1.3rem] border p-4",
        isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
      ].join(" ")}
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <span
              className={[
                "rounded-2xl p-3 text-lg",
                isDark ? "bg-zinc-900 text-emerald-300" : "bg-emerald-50 text-emerald-700",
              ].join(" ")}
            >
              <HiMiniQueueList />
            </span>
            <div>
              <p className="text-sm font-semibold">Bulk cluster actions</p>
              <p className="text-sm text-zinc-500">
                Select clusters, assign one label/decision in bulk, then save and promote.
              </p>
            </div>
          </div>
          <p className="mt-3 text-xs text-zinc-500">
            To use <span className="font-semibold">promote-cluster-labels</span>, bulk-assign a
            label with <span className="font-semibold">Decision = labeled</span>, click{" "}
            <span className="font-semibold">Save review</span>, then promote the saved review
            into the manifest.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onSelectAll}
            disabled={selectedCount === totalCount || totalCount === 0}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800" : "bg-white text-zinc-800 hover:bg-zinc-100",
              selectedCount === totalCount || totalCount === 0 ? "opacity-60" : "",
            ].join(" ")}
          >
            Select all clusters
          </button>
          <button
            type="button"
            onClick={onClearSelection}
            disabled={selectedCount === 0}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800" : "bg-white text-zinc-800 hover:bg-zinc-100",
              selectedCount === 0 ? "opacity-60" : "",
            ].join(" ")}
          >
            Clear selection
          </button>
        </div>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1fr_1fr_auto]">
        <label className="flex flex-col gap-2">
          <span className={labelClass}>Bulk label</span>
          <input
            className={fieldClass}
            value={bulkLabel}
            placeholder="glass"
            onChange={(event) => setBulkLabel(event.target.value)}
          />
        </label>
        <label className="flex flex-col gap-2">
          <span className={labelClass}>Bulk decision</span>
          <select
            className={fieldClass}
            value={bulkStatus}
            onChange={(event) => setBulkStatus(event.target.value as ClusterReviewStatus)}
          >
            <option value="labeled">labeled</option>
            <option value="excluded">excluded</option>
            <option value="unlabeled">unlabeled</option>
          </select>
        </label>
        <label className="flex flex-col gap-2">
          <span className={labelClass}>Bulk notes</span>
          <input
            className={fieldClass}
            value={bulkNotes}
            placeholder="optional context for selected clusters"
            onChange={(event) => setBulkNotes(event.target.value)}
          />
        </label>
        <div className="flex flex-col gap-2">
          <span className={labelClass}>Selected</span>
          <button
            type="button"
            onClick={handleApply}
            disabled={applyDisabled}
            className={[
              "min-h-12 rounded-2xl px-4 py-3 text-sm font-semibold transition",
              isDark ? "bg-white text-black hover:bg-zinc-200" : "bg-zinc-950 text-white hover:bg-zinc-800",
              applyDisabled ? "opacity-60" : "",
            ].join(" ")}
          >
            Apply to {selectedCount} cluster{selectedCount === 1 ? "" : "s"}
          </button>
        </div>
      </div>

      <div className="mt-3 text-xs text-zinc-500">
        {selectedCount === 0
          ? "Select at least one cluster card below before applying a bulk label or decision."
          : bulkStatus === "labeled" && !bulkLabel.trim()
            ? "Enter a label before applying the labeled decision to selected clusters."
            : `Ready to update ${selectedCount} selected cluster${selectedCount === 1 ? "" : "s"}.`}
      </div>
    </div>
  );
}
