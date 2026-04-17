"use client";

import Link from "next/link";
import { HiArrowUpRight, HiChartBarSquare } from "react-icons/hi2";

import { TrainingHistoryChart } from "@/components/charts/training-history-chart";
import { RunsGrid } from "@/components/dashboard/runs-grid";
import { SummaryStrip } from "@/components/dashboard/summary-strip";
import { SystemNotices } from "@/components/dashboard/system-notices";
import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { PageIntro, Panel } from "@/components/ui/primitives";
import { asObject } from "@/lib/localagent";

export function RunsCatalogPage() {
  const { isDark } = useThemeMode();
  const { runs, selectedExperiment, setSelectedExperiment, runDetail } = useLocalAgent();
  const dashboard = asObject(runDetail?.dashboard_summary);
  const cards = asObject(dashboard?.cards);

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Runs"
        title="Experiment catalog"
        description="Browse persisted experiments, preview their high-level metrics, and jump into full detail pages with confusion matrices and manifests."
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
              <option value="">Preview run</option>
              {runs.map((run) => (
                <option key={run.experiment_name} value={run.experiment_name}>
                  {run.experiment_name}
                </option>
              ))}
            </select>
            <Link
              href={
                selectedExperiment
                  ? `/runs/${encodeURIComponent(selectedExperiment)}`
                  : "/runs"
              }
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              <HiArrowUpRight />
              Open selected
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
              Compare benchmarks
            </Link>
          </>
        }
      />

      <SystemNotices />

      <Panel
        title="Saved experiments"
        description="Run cards are indexed from persisted JSON, so this page remains useful even after the original jobs finish."
      >
        <RunsGrid runs={runs} />
      </Panel>

      {selectedExperiment ? (
        <div className="grid gap-6">
          <SummaryStrip cards={cards} />
          <TrainingHistoryChart training={asObject(runDetail?.training)} />
        </div>
      ) : null}
    </div>
  );
}
