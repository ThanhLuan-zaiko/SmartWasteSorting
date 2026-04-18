"use client";

import Image from "next/image";
import { HiBeaker, HiOutlineSquares2X2, HiSparkles } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { Panel } from "@/components/ui/primitives";
import { API_PREFIX, asArray, asObject, DISCOVERY_ACTIONS } from "@/lib/localagent";

const EMBEDDING_REQUIRED_ACTIONS = new Set(["cluster", "export-cluster-review", "promote-cluster-labels"]);
const CLUSTER_REQUIRED_ACTIONS = new Set(["export-cluster-review", "promote-cluster-labels"]);

function buildDatasetImageUrl(relativePath: string): string {
  return `${API_PREFIX}/dataset/image?relative_path=${encodeURIComponent(relativePath)}`;
}

export function DiscoveryStudio() {
  const { isDark } = useThemeMode();
  const {
    runDetail,
    pipelineForm,
    pipelineCatalog,
    isSubmitting,
    setPipelineField,
    submitPipeline,
  } = useLocalAgent();

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
  const clusterPreviewTotal =
    typeof datasetSummary?.cluster_preview_total === "number"
      ? datasetSummary.cluster_preview_total
      : 0;
  const clusterPreviewTruncated = datasetSummary?.cluster_preview_truncated === true;
  const clusterPreviews = asArray(datasetSummary?.cluster_previews).flatMap((value) => {
    const preview = asObject(value);
    return preview ? [preview] : [];
  });
  const acceptedSourcesSummary = Object.entries(acceptedLabelSourceCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");
  const trainableLabelsSummary = Object.entries(trainableLabelCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");
  const reviewSummary = Object.entries(reviewStatusCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");

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
    return null;
  }

  return (
    <Panel
      title="Step 2: Discovery workflow"
      description="Use embeddings and clusters to review visually similar images together, then promote accepted cluster labels back into the manifest."
    >
      <div
        className={[
          "rounded-[1.2rem] border px-4 py-3 text-sm leading-6",
          isDark
            ? "border-zinc-800 bg-zinc-950 text-zinc-300"
            : "border-zinc-200 bg-zinc-50 text-zinc-700",
        ].join(" ")}
      >
        <p>
          Effective training mode: <span className="font-semibold">{effectiveTrainingMode}</span>.
          {effectiveTrainingMode === "accepted_labels_only"
            ? " Filename hints are no longer treated as train labels because accepted labels already exist."
            : " The manifest is still relying on weak filename hints until you accept labels from review or pseudo-labeling."}
        </p>
        {trainableLabelsSummary ? (
          <p className="mt-2">
            Trainable labels: <span className="font-semibold">{trainableLabelsSummary}</span>
          </p>
        ) : null}
        <p className="mt-2">
          Clustered files: <span className="font-semibold">{clusteredFiles}</span>. Outliers:{" "}
          <span className="font-semibold">{clusterOutliers}</span>.
        </p>
        {acceptedSourcesSummary ? (
          <p className="mt-2">
            Accepted label sources: <span className="font-semibold">{acceptedSourcesSummary}</span>
          </p>
        ) : null}
        {reviewSummary ? (
          <p className="mt-2">
            Review status: <span className="font-semibold">{reviewSummary}</span>
          </p>
        ) : null}
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="review_output">
              Review export
            </label>
            <input
              id="review_output"
              className={fieldClass}
              value={pipelineForm.review_output}
              onChange={(event) => setPipelineField("review_output", event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="review_file">
              Review import
            </label>
            <input
              id="review_file"
              className={fieldClass}
              value={pipelineForm.review_file}
              onChange={(event) => setPipelineField("review_file", event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="num_clusters">
              Num clusters
            </label>
            <input
              id="num_clusters"
              className={fieldClass}
              placeholder="auto"
              value={pipelineForm.num_clusters}
              onChange={(event) => setPipelineField("num_clusters", event.target.value)}
            />
          </div>
        </div>

        <div
          className={[
            "rounded-[1.5rem] border p-4",
            isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
          ].join(" ")}
        >
          <div className="flex items-center gap-3">
            <span
              className={[
                "rounded-2xl p-3 text-lg",
                isDark ? "bg-zinc-900 text-emerald-300" : "bg-emerald-50 text-emerald-700",
              ].join(" ")}
            >
              <HiOutlineSquares2X2 />
            </span>
            <div>
              <p className="text-sm font-semibold">Discovery commands</p>
              <p className="text-sm text-zinc-500">
                {pipelineCatalog.dataset_commands
                  .filter((command) =>
                    DISCOVERY_ACTIONS.includes(
                      command as (typeof DISCOVERY_ACTIONS)[number],
                    ),
                  )
                  .join(", ")}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {DISCOVERY_ACTIONS.map((command) => {
          const blockReason = actionBlockReason(command);
          const isDisabled = isSubmitting !== null || blockReason !== null;
          const isPrimary = command === "embed" || command === "cluster";

          return (
            <button
              key={command}
              type="button"
              disabled={isDisabled}
              title={blockReason ?? undefined}
              onClick={() => void submitPipeline(command)}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
                isPrimary
                  ? isDark
                    ? "bg-white text-black hover:bg-zinc-200"
                    : "bg-zinc-950 text-white hover:bg-zinc-800"
                  : isDark
                    ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                isDisabled ? "opacity-60" : "",
              ].join(" ")}
            >
              {isPrimary ? <HiSparkles /> : <HiBeaker />}
              {isSubmitting === command ? `Starting ${command}...` : command}
            </button>
          );
        })}
      </div>

      {clusterPreviews.length > 0 ? (
        <div className="mt-7">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-sm font-semibold">Representative clusters</p>
              <p className="text-sm text-zinc-500">
                {clusterPreviewTruncated
                  ? `Showing ${clusterPreviews.length} of ${clusterPreviewTotal} clusters ranked by size.`
                  : `${clusterPreviewTotal} clustered groups ready for review.`}
              </p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
            {clusterPreviews.map((clusterPreview) => {
              const clusterId =
                typeof clusterPreview.cluster_id === "number" ? clusterPreview.cluster_id : "N/A";
              const clusterSize =
                typeof clusterPreview.cluster_size === "number" ? clusterPreview.cluster_size : 0;
              const outlierCount =
                typeof clusterPreview.outlier_count === "number" ? clusterPreview.outlier_count : 0;
              const majorityLabel =
                typeof clusterPreview.current_majority_label === "string" &&
                clusterPreview.current_majority_label.length > 0
                  ? clusterPreview.current_majority_label
                  : null;
              const reviewCounts = asObject(clusterPreview.review_status_counts);
              const representatives = asArray(clusterPreview.representatives).flatMap((value) => {
                const representative = asObject(value);
                return representative ? [representative] : [];
              });
              const reviewCountsSummary = Object.entries(reviewCounts ?? {})
                .map(([label, count]) => `${label}: ${count}`)
                .join(", ");

              return (
                <div
                  key={`cluster-${clusterId}`}
                  className={[
                    "rounded-[1.5rem] border p-4",
                    isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-white",
                  ].join(" ")}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold">Cluster {clusterId}</p>
                      <p className="text-xs text-zinc-500">
                        {clusterSize} samples • {outlierCount} outliers
                      </p>
                    </div>
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
                  </div>

                  {reviewCountsSummary ? (
                    <p className="mt-3 text-xs text-zinc-500">Review: {reviewCountsSummary}</p>
                  ) : null}

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    {representatives.map((representative) => {
                      const relativePath =
                        typeof representative.relative_path === "string"
                          ? representative.relative_path
                          : "";
                      const label =
                        typeof representative.label === "string" ? representative.label : "unknown";
                      const labelSource =
                        typeof representative.label_source === "string"
                          ? representative.label_source
                          : "unknown";
                      const reviewStatus =
                        typeof representative.review_status === "string"
                          ? representative.review_status
                          : "unreviewed";

                      return (
                        <div key={relativePath} className="min-w-0">
                          <div
                            className={[
                              "aspect-square overflow-hidden rounded-[1.2rem] border",
                              isDark ? "border-zinc-800 bg-zinc-900" : "border-zinc-200 bg-zinc-50",
                            ].join(" ")}
                          >
                            {relativePath ? (
                              <div className="relative h-full w-full">
                                <Image
                                  src={buildDatasetImageUrl(relativePath)}
                                  alt={relativePath}
                                  fill
                                  unoptimized
                                  sizes="(min-width: 1536px) 180px, (min-width: 1024px) 220px, 45vw"
                                  className="object-cover"
                                />
                              </div>
                            ) : null}
                          </div>
                          <p className="mt-2 break-all text-[11px] text-zinc-500">{relativePath}</p>
                          <p className="mt-1 text-xs font-semibold">
                            {label} <span className="font-normal text-zinc-500">via {labelSource}</span>
                          </p>
                          <p className="text-[11px] text-zinc-500">{reviewStatus}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}
