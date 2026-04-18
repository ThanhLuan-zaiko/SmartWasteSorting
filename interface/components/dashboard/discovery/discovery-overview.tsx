type DiscoveryOverviewProps = {
  acceptedSourcesSummary: string;
  clusterOutliers: number;
  clusteredFiles: number;
  effectiveTrainingMode: string;
  isDark: boolean;
  manifestReviewSummary: string;
  trainableLabelsSummary: string;
};

export function DiscoveryOverview({
  acceptedSourcesSummary,
  clusterOutliers,
  clusteredFiles,
  effectiveTrainingMode,
  isDark,
  manifestReviewSummary,
  trainableLabelsSummary,
}: DiscoveryOverviewProps) {
  return (
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
      {manifestReviewSummary ? (
        <p className="mt-2">
          Manifest review status: <span className="font-semibold">{manifestReviewSummary}</span>
        </p>
      ) : null}
    </div>
  );
}
