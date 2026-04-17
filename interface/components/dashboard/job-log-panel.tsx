"use client";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel, StatusBadge } from "@/components/ui/primitives";

export function JobLogPanel() {
  const { isDark } = useThemeMode();
  const { jobs, activeJobId, activeLogs, streamConnected } = useLocalAgent();
  const activeJob = jobs.find((job) => job.job_id === activeJobId) ?? null;

  return (
    <Panel
      title="Job logs"
      description="Live stdout and stderr tail for the currently selected job."
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={[
              "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]",
              streamConnected
                ? isDark
                  ? "bg-emerald-950 text-emerald-300"
                  : "bg-emerald-50 text-emerald-700"
                : isDark
                  ? "bg-zinc-950 text-zinc-400"
                  : "bg-zinc-100 text-zinc-600",
            ].join(" ")}
          >
            {streamConnected ? "websocket live" : "http fallback"}
          </span>
          {activeJob ? <StatusBadge status={activeJob.status} /> : null}
        </div>
      }
    >
      {!activeJob ? (
        <EmptyState
          title="Select a job"
          description="Open the Jobs page or click a run action to inspect live logs."
        />
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          <div>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500">
              STDOUT
            </h3>
            <div
              className={[
                "min-h-[280px] overflow-auto rounded-[1.5rem] border p-4 font-mono text-xs leading-6 whitespace-pre-wrap break-words",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-emerald-200"
                  : "border-zinc-200 bg-zinc-950 text-emerald-200",
              ].join(" ")}
            >
              {(activeLogs?.stdout ?? []).join("\n") || "No stdout yet."}
            </div>
          </div>
          <div>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500">
              STDERR
            </h3>
            <div
              className={[
                "min-h-[280px] overflow-auto rounded-[1.5rem] border p-4 font-mono text-xs leading-6 whitespace-pre-wrap break-words",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-rose-200"
                  : "border-zinc-200 bg-zinc-950 text-rose-200",
              ].join(" ")}
            >
              {(activeLogs?.stderr ?? []).join("\n") || "No stderr yet."}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
