"use client";

import { useEffect, useEffectEvent } from "react";
import { HiArrowPath, HiRocketLaunch } from "react-icons/hi2";

import { TrainingHistoryChart } from "@/components/charts/training-history-chart";
import { BenchmarkComparisonChart } from "@/components/charts/benchmark-comparison-chart";
import { ClassMetricsTable } from "@/components/charts/class-metrics-table";
import { ConfusionMatrix } from "@/components/charts/confusion-matrix";
import { SummaryStrip } from "@/components/dashboard/summary-strip";
import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { PageIntro, Panel } from "@/components/ui/primitives";
import { asObject, formatMetric } from "@/lib/localagent";

export function RunDetailView({ experiment }: { experiment: string }) {
  const {
    runDetail,
    comparison,
    compareExperiment,
    ensureRunLoaded,
    setCompareExperiment,
    runs,
  } = useLocalAgent();
  const { isDark } = useThemeMode();

  const dashboard = asObject(runDetail?.dashboard_summary);
  const cards = asObject(dashboard?.cards);
  const benchmark = asObject(dashboard?.benchmark);
  const evaluation = asObject(benchmark?.evaluation ?? runDetail?.evaluation ?? null);
  const modelManifest = asObject(runDetail?.model_manifest);
  const jobHistory = runDetail?.job_history ?? [];
  const ensureCurrentRunLoaded = useEffectEvent(async () => {
    await ensureRunLoaded(experiment);
  });

  useEffect(() => {
    void ensureCurrentRunLoaded();
  }, [experiment]);

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Run detail"
        title={experiment}
        description="Experiment-specific metrics, charts, confusion matrix, model manifest, and related job history."
        actions={
          <>
            <button
              type="button"
              onClick={() => void ensureRunLoaded(experiment)}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              <HiArrowPath />
              Reload run
            </button>
            <select
              className={[
                "min-h-11 rounded-full border px-4 py-2 text-sm font-semibold outline-none transition",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-200"
                  : "border-zinc-200 bg-white text-zinc-700",
              ].join(" ")}
              value={compareExperiment}
              onChange={(event) => setCompareExperiment(event.target.value)}
            >
              <option value="">Compare against</option>
              {runs
                .filter((run) => run.experiment_name !== experiment)
                .map((run) => (
                  <option key={run.experiment_name} value={run.experiment_name}>
                    {run.experiment_name}
                  </option>
                ))}
            </select>
          </>
        }
      />

      <SummaryStrip cards={cards} />

      <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <TrainingHistoryChart training={asObject(runDetail?.training)} />
        <Panel
          title="Model manifest"
          description="Export metadata consumed by runtime and API layers."
        >
          <div className="grid gap-4 md:grid-cols-2">
            <ManifestStat label="Model name" value={modelManifest?.model_name} />
            <ManifestStat label="Image size" value={modelManifest?.image_size} />
            <ManifestStat label="Labels path" value={modelManifest?.labels_path} />
            <ManifestStat label="ONNX path" value={modelManifest?.onnx_path} />
            <ManifestStat
              label="Normalization preset"
              value={asObject(modelManifest?.normalization)?.preset}
            />
            <ManifestStat
              label="ONNX opset"
              value={asObject(modelManifest?.onnx)?.opset}
            />
          </div>
        </Panel>
      </div>

      <BenchmarkComparisonChart
        comparison={comparison}
        leftExperiment={experiment}
        rightExperiment={compareExperiment}
      />

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <ConfusionMatrix evaluation={evaluation} />
        <ClassMetricsTable evaluation={evaluation} />
      </div>

      <Panel
        title="Related jobs"
        description="Jobs recorded for this experiment in the local Actix control plane."
      >
        <div className="grid gap-4 xl:grid-cols-2">
          {jobHistory.length === 0 ? (
            <p className="text-sm text-zinc-500">
              No recorded jobs yet for this experiment.
            </p>
          ) : (
            jobHistory.map((job) => (
              <article
                key={job.job_id}
                className={[
                  "rounded-[1.4rem] border p-4",
                  isDark
                    ? "border-zinc-800 bg-zinc-950"
                    : "border-zinc-200 bg-zinc-50",
                ].join(" ")}
              >
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="text-lg font-bold">{job.job_type}</h3>
                    <p className="mt-1 text-sm text-zinc-500">{job.command}</p>
                  </div>
                  <HiRocketLaunch className="text-xl text-emerald-500" />
                </div>
                <p className="mt-4 text-xs uppercase tracking-[0.22em] text-zinc-500">
                  {job.created_at}
                </p>
              </article>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
}

function ManifestStat({ label, value }: { label: string; value: unknown }) {
  const { isDark } = useThemeMode();

  return (
    <article
      className={[
        "rounded-[1.35rem] border p-4",
        isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-zinc-50",
      ].join(" ")}
    >
      <div className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500">
        {label}
      </div>
      <div className="mt-2 text-lg font-bold break-all">{formatMetric(value)}</div>
    </article>
  );
}
