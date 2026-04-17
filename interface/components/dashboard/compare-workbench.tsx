"use client";

import { useEffect } from "react";

import { BenchmarkComparisonChart } from "@/components/charts/benchmark-comparison-chart";
import { SystemNotices } from "@/components/dashboard/system-notices";
import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { MetricCard, PageIntro, Panel } from "@/components/ui/primitives";

export function CompareWorkbench({
  initialLeft,
  initialRight,
}: {
  initialLeft?: string;
  initialRight?: string;
}) {
  const { isDark } = useThemeMode();
  const {
    runs,
    selectedExperiment,
    setSelectedExperiment,
    compareExperiment,
    setCompareExperiment,
    comparison,
  } = useLocalAgent();

  useEffect(() => {
    if (
      initialLeft &&
      runs.some((run) => run.experiment_name === initialLeft) &&
      initialLeft !== selectedExperiment
    ) {
      setSelectedExperiment(initialLeft);
    }
  }, [initialLeft, runs, selectedExperiment, setSelectedExperiment]);

  useEffect(() => {
    if (
      initialRight &&
      runs.some((run) => run.experiment_name === initialRight) &&
      initialRight !== compareExperiment
    ) {
      setCompareExperiment(initialRight);
    }
  }, [compareExperiment, initialRight, runs, setCompareExperiment]);

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Compare"
        title="Benchmark deltas across experiments"
        description="Render persisted benchmark artifacts as interactive charts and delta cards, without rerunning the pipeline just to compare two saved runs."
      />

      <SystemNotices />

      <Panel
        title="Comparison controls"
        description="Choose the left and right experiments to drive the benchmark chart and delta cards below."
      >
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label
              htmlFor="left_experiment"
              className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500"
            >
              Left experiment
            </label>
            <select
              id="left_experiment"
              value={selectedExperiment}
              onChange={(event) => setSelectedExperiment(event.target.value)}
              className={[
                "min-h-12 rounded-2xl border px-4 py-3 text-sm outline-none transition",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-200"
                  : "border-zinc-200 bg-zinc-50 text-zinc-900",
              ].join(" ")}
            >
              <option value="">Choose an experiment</option>
              {runs.map((run) => (
                <option key={run.experiment_name} value={run.experiment_name}>
                  {run.experiment_name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-2">
            <label
              htmlFor="right_experiment"
              className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500"
            >
              Right experiment
            </label>
            <select
              id="right_experiment"
              value={compareExperiment}
              onChange={(event) => setCompareExperiment(event.target.value)}
              className={[
                "min-h-12 rounded-2xl border px-4 py-3 text-sm outline-none transition",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-200"
                  : "border-zinc-200 bg-zinc-50 text-zinc-900",
              ].join(" ")}
            >
              <option value="">Choose an experiment</option>
              {runs
                .filter((run) => run.experiment_name !== selectedExperiment)
                .map((run) => (
                  <option key={run.experiment_name} value={run.experiment_name}>
                    {run.experiment_name}
                  </option>
                ))}
            </select>
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <MetricCard label="Duration delta" value={comparison?.duration_delta_seconds} />
        <MetricCard label="Fit delta" value={comparison?.fit_stage_delta_seconds} />
        <MetricCard label="Accuracy delta" value={comparison?.accuracy_delta} />
        <MetricCard label="Macro F1 delta" value={comparison?.macro_f1_delta} />
        <MetricCard label="Weighted F1 delta" value={comparison?.weighted_f1_delta} />
        <MetricCard
          label="Backends"
          value={
            comparison
              ? `${comparison.left_backend ?? "N/A"} -> ${comparison.right_backend ?? "N/A"}`
              : null
          }
        />
      </div>

      <BenchmarkComparisonChart
        comparison={comparison}
        leftExperiment={selectedExperiment}
        rightExperiment={compareExperiment}
      />
    </div>
  );
}
