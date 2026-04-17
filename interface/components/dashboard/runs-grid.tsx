"use client";

import Link from "next/link";
import { HiArrowUpRight, HiBeaker, HiChartBar, HiSparkles } from "react-icons/hi2";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, StatusBadge } from "@/components/ui/primitives";
import { formatMetric, type RunIndexEntry } from "@/lib/localagent";

export function RunsGrid({ runs }: { runs: RunIndexEntry[] }) {
  const { isDark } = useThemeMode();

  if (runs.length === 0) {
    return (
      <EmptyState
        title="No runs discovered"
        description="Run a dataset or training pipeline and persisted artifacts will appear here."
      />
    );
  }

  return (
    <div className="grid gap-4 xl:grid-cols-3">
      {runs.map((run) => (
        <article
          key={run.experiment_name}
          className={[
            "group rounded-[1.6rem] border p-5 transition hover:-translate-y-1",
            isDark
              ? "border-zinc-800 bg-zinc-900 hover:border-zinc-700"
              : "border-zinc-200 bg-white hover:border-zinc-300",
          ].join(" ")}
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <div
                className={[
                  "inline-flex rounded-2xl p-3 text-lg",
                  isDark
                    ? "bg-zinc-950 text-emerald-300"
                    : "bg-emerald-50 text-emerald-700",
                ].join(" ")}
              >
                <HiBeaker />
              </div>
              <h3 className="mt-4 text-2xl font-black tracking-[-0.05em]">
                {run.experiment_name}
              </h3>
              <p className={["mt-2 text-sm", isDark ? "text-zinc-400" : "text-zinc-600"].join(" ")}>
                Latest artifact: {run.latest_generated_at || "N/A"}
              </p>
            </div>
            <StatusBadge status={typeof run.cards?.benchmark_status === "string" ? run.cards?.benchmark_status : "completed"} />
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            <div
              className={[
                "rounded-2xl px-4 py-3",
                isDark ? "bg-zinc-950 text-zinc-200" : "bg-zinc-50 text-zinc-800",
              ].join(" ")}
            >
              <div className="text-xs uppercase tracking-[0.22em] text-zinc-500">
                Accuracy
              </div>
              <div className="mt-2 text-2xl font-black">
                {formatMetric(run.cards?.accuracy)}
              </div>
            </div>
            <div
              className={[
                "rounded-2xl px-4 py-3",
                isDark ? "bg-zinc-950 text-zinc-200" : "bg-zinc-50 text-zinc-800",
              ].join(" ")}
            >
              <div className="text-xs uppercase tracking-[0.22em] text-zinc-500">
                Backend
              </div>
              <div className="mt-2 text-2xl font-black">
                {formatMetric(run.training_backend)}
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href={`/runs/${encodeURIComponent(run.experiment_name)}`}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              Inspect run
              <HiArrowUpRight />
            </Link>
            <Link
              href={`/compare?left=${encodeURIComponent(run.experiment_name)}`}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              ].join(" ")}
            >
              Compare
              <HiChartBar />
            </Link>
            <Link
              href="/pipelines"
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              ].join(" ")}
            >
              New job
              <HiSparkles />
            </Link>
          </div>
        </article>
      ))}
    </div>
  );
}
