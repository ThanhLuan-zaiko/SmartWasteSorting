"use client";

import Link from "next/link";
import { HiArrowUpRight, HiPlayCircle } from "react-icons/hi2";

import { JobLogPanel } from "@/components/dashboard/job-log-panel";
import { JobsPanel } from "@/components/dashboard/jobs-panel";
import { SystemNotices } from "@/components/dashboard/system-notices";
import { useThemeMode } from "@/components/theme-provider";
import { PageIntro } from "@/components/ui/primitives";

export function JobsConsolePage() {
  const { isDark } = useThemeMode();

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Jobs"
        title="Live execution console"
        description="Inspect the local single-user queue, see active progress, and tail stdout or stderr without leaving the browser."
        actions={
          <>
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
              Start new job
            </Link>
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
              Inspect runs
            </Link>
          </>
        }
      />

      <SystemNotices />

      <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <JobsPanel />
        <JobLogPanel />
      </div>
    </div>
  );
}
