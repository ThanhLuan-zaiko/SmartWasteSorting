"use client";

import { HiStopCircle } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel, StatusBadge } from "@/components/ui/primitives";

export function JobsPanel() {
  const { isDark } = useThemeMode();
  const { jobs, activeJobId, setActiveJobId, cancelActiveJob } = useLocalAgent();
  const activeJob = jobs.find((job) => job.job_id === activeJobId) ?? null;

  return (
    <Panel
      title="Jobs"
      description="Queued and running local jobs spawned by the Actix control plane."
      actions={
        activeJob ? (
          <button
            type="button"
            onClick={() => void cancelActiveJob()}
            disabled={activeJob.status !== "pending" && activeJob.status !== "running"}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
              isDark
                ? "bg-rose-400 text-black hover:bg-rose-300"
                : "bg-rose-500 text-white hover:bg-rose-600",
              activeJob.status !== "pending" && activeJob.status !== "running"
                ? "opacity-50"
                : "",
            ].join(" ")}
          >
            <HiStopCircle />
            Cancel active job
          </button>
        ) : null
      }
    >
      {jobs.length === 0 ? (
        <EmptyState
          title="No jobs yet"
          description="Trigger a pipeline from the Pipelines page and the job stream will show up here."
        />
      ) : (
        <div className="grid gap-3">
          {jobs.map((job) => (
            <button
              key={job.job_id}
              type="button"
              onClick={() => setActiveJobId(job.job_id)}
              className={[
                "rounded-[1.5rem] border p-4 text-left transition",
                activeJobId === job.job_id
                  ? isDark
                    ? "border-emerald-500 bg-zinc-950"
                    : "border-zinc-950 bg-zinc-50"
                  : isDark
                    ? "border-zinc-800 bg-zinc-950/70 hover:border-zinc-700"
                    : "border-zinc-200 bg-zinc-50 hover:border-zinc-300",
              ].join(" ")}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-lg font-bold">{job.job_type}</h3>
                  <p
                    className={[
                      "mt-1 text-sm",
                      isDark ? "text-zinc-400" : "text-zinc-600",
                    ].join(" ")}
                  >
                    {job.experiment_name ?? "dataset"} - {job.command}
                  </p>
                </div>
                <StatusBadge status={job.status} />
              </div>
              <p className="mt-3 text-xs uppercase tracking-[0.22em] text-zinc-500">
                {job.created_at}
              </p>
            </button>
          ))}
        </div>
      )}
    </Panel>
  );
}
