"use client";

import Link from "next/link";
import { HiArrowUpRight, HiBars3BottomLeft, HiChartBarSquare, HiPlayCircle } from "react-icons/hi2";

import { BenchmarkComparisonChart } from "@/components/charts/benchmark-comparison-chart";
import { TrainingHistoryChart } from "@/components/charts/training-history-chart";
import { JobLogPanel } from "@/components/dashboard/job-log-panel";
import { RunsGrid } from "@/components/dashboard/runs-grid";
import { SummaryStrip } from "@/components/dashboard/summary-strip";
import { SystemNotices } from "@/components/dashboard/system-notices";
import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { PageIntro, Panel } from "@/components/ui/primitives";
import { asObject } from "@/lib/localagent";

export function OverviewHub() {
  const { isDark } = useThemeMode();
  const {
    runs,
    selectedExperiment,
    setSelectedExperiment,
    compareExperiment,
    setCompareExperiment,
    runDetail,
    comparison,
  } = useLocalAgent();

  const dashboard = asObject(runDetail?.dashboard_summary);
  const cards = asObject(dashboard?.cards);

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Overview"
        title="Energetic local agent studio"
        description="Control dataset pipelines, monitor training runs, inspect loss curves, and compare persisted benchmarks without dropping back to long CLI commands."
        actions={
          <>
            <select
              value={selectedExperiment}
              onChange={(event) => setSelectedExperiment(event.target.value)}
              className={[
                "min-h-11 rounded-full border px-4 py-2 text-sm font-semibold outline-none transition",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-200"
                  : "border-zinc-200 bg-white text-zinc-700",
              ].join(" ")}
            >
              <option value="">Select run</option>
              {runs.map((run) => (
                <option key={run.experiment_name} value={run.experiment_name}>
                  {run.experiment_name}
                </option>
              ))}
            </select>
            <select
              value={compareExperiment}
              onChange={(event) => setCompareExperiment(event.target.value)}
              className={[
                "min-h-11 rounded-full border px-4 py-2 text-sm font-semibold outline-none transition",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-200"
                  : "border-zinc-200 bg-white text-zinc-700",
              ].join(" ")}
            >
              <option value="">Compare run</option>
              {runs
                .filter((run) => run.experiment_name !== selectedExperiment)
                .map((run) => (
                  <option key={run.experiment_name} value={run.experiment_name}>
                    {run.experiment_name}
                  </option>
                ))}
            </select>
            <Link
              href="/pipelines"
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              <HiPlayCircle />
              Launch pipeline
            </Link>
            <Link
              href="/compare"
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              ].join(" ")}
            >
              <HiChartBarSquare />
              Compare
            </Link>
          </>
        }
      />

      <SystemNotices />
      <SummaryStrip cards={cards} />

      <div className="grid gap-6 xl:grid-cols-[1.35fr_0.95fr]">
        <TrainingHistoryChart training={asObject(runDetail?.training)} />
        <JobLogPanel />
      </div>

      <BenchmarkComparisonChart
        comparison={comparison}
        leftExperiment={selectedExperiment}
        rightExperiment={compareExperiment}
      />

      <Panel
        title="Tracked experiments"
        description="Every persisted JSON artifact becomes a reusable run card here, so users can jump between training outputs instead of repeating the same commands."
        actions={
          <Link
            href="/runs"
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
              isDark
                ? "bg-zinc-950 text-zinc-200 hover:bg-zinc-800"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
            ].join(" ")}
          >
            <HiArrowUpRight />
            Full run catalog
          </Link>
        }
      >
        <RunsGrid runs={runs} />
      </Panel>

      <Panel
        title="Fast navigation"
        description="Jump straight into execution, logs, or experiment detail pages."
      >
        <div className="grid gap-4 md:grid-cols-3">
          {[
            {
              href: "/pipelines",
              label: "Pipeline studio",
              copy: "Create training, export, and dataset jobs from forms and presets.",
              icon: HiPlayCircle,
            },
            {
              href: "/jobs",
              label: "Job console",
              copy: "Inspect the local queue and live stdout or stderr tails.",
              icon: HiBars3BottomLeft,
            },
            {
              href: selectedExperiment
                ? `/runs/${encodeURIComponent(selectedExperiment)}`
                : "/runs",
              label: "Selected run",
              copy: "Open the focused experiment with charts, manifests, and class metrics.",
              icon: HiArrowUpRight,
            },
          ].map((item) => {
            const Icon = item.icon;

            return (
              <Link
                key={item.label}
                href={item.href}
                className={[
                  "rounded-[1.45rem] border p-5 transition hover:-translate-y-1",
                  isDark
                    ? "border-zinc-800 bg-zinc-950 hover:border-zinc-700"
                    : "border-zinc-200 bg-zinc-50 hover:border-zinc-300",
                ].join(" ")}
              >
                <div
                  className={[
                    "inline-flex rounded-2xl p-3 text-lg",
                    isDark
                      ? "bg-zinc-900 text-emerald-300"
                      : "bg-emerald-50 text-emerald-700",
                  ].join(" ")}
                >
                  <Icon />
                </div>
                <h3 className="mt-4 text-2xl font-black tracking-[-0.05em]">
                  {item.label}
                </h3>
                <p className="mt-3 text-sm leading-6 text-zinc-500">{item.copy}</p>
              </Link>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
