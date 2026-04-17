"use client";

import Link from "next/link";
import { HiArrowUpRight, HiBars3BottomLeft, HiChartBarSquare } from "react-icons/hi2";

import { PipelineActions } from "@/components/dashboard/pipeline-actions";
import { SystemNotices } from "@/components/dashboard/system-notices";
import { TrainingStudio } from "@/components/dashboard/training-studio";
import { useThemeMode } from "@/components/theme-provider";
import { PageIntro, Panel } from "@/components/ui/primitives";

export function PipelineStudioPage() {
  const { isDark } = useThemeMode();

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Pipelines"
        title="Build and launch runs from forms"
        description="This page replaces repetitive CLI commands with reusable controls for dataset preparation, training, evaluation, ONNX export, and benchmark jobs."
        actions={
          <>
            <Link
              href="/jobs"
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              ].join(" ")}
            >
              <HiBars3BottomLeft />
              Open jobs
            </Link>
            <Link
              href="/compare"
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              <HiChartBarSquare />
              View benchmarks
            </Link>
          </>
        }
      />

      <SystemNotices />
      <TrainingStudio />
      <PipelineActions />

      <Panel
        title="Artifact-first workflow"
        description="Every button still writes the same JSON artifacts as the CLI, so GUI and CLI stay interchangeable."
      >
        <div className="grid gap-4 md:grid-cols-3">
          {[
            "Training, evaluation, export, and benchmark steps persist structured JSON for later comparison.",
            "Job manifests and log files survive refreshes, so the interface can resume where the last session left off.",
            "Rust Actix remains the control plane while Python keeps the training pipeline logic and ONNX export.",
          ].map((copy) => (
            <article
              key={copy}
              className={[
                "rounded-[1.4rem] border p-5",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-300"
                  : "border-zinc-200 bg-zinc-50 text-zinc-700",
              ].join(" ")}
            >
              <p className="text-sm leading-7">{copy}</p>
            </article>
          ))}
        </div>
        <div className="mt-5">
          <Link
            href="/runs"
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
              isDark
                ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
            ].join(" ")}
          >
            <HiArrowUpRight />
            Browse persisted runs
          </Link>
        </div>
      </Panel>
    </div>
  );
}
