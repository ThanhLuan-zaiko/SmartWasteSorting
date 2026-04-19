"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    HiArrowPath,
    HiBars3BottomLeft,
    HiBeaker,
    HiChartBarSquare,
    HiMoon,
    HiPlayCircle,
    HiSparkles,
    HiSun,
} from "react-icons/hi2";
import { PiFlaskBold } from "react-icons/pi";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: HiSparkles },
  { href: "/pipelines", label: "Pipelines", icon: HiPlayCircle },
  { href: "/classify", label: "Classify", icon: HiBeaker },
  { href: "/jobs", label: "Jobs", icon: HiBars3BottomLeft },
  { href: "/runs", label: "Runs", icon: PiFlaskBold },
  { href: "/compare", label: "Compare", icon: HiChartBarSquare },
] as const;

export function AppFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { isDark, toggleTheme } = useThemeMode();
  const { refreshAll, jobs } = useLocalAgent();
  const activeJobs = jobs.filter(
    (job) => job.status === "pending" || job.status === "running",
  ).length;

  return (
    <div
      className={[
        "min-h-screen transition-colors duration-300",
        isDark ? "bg-black text-white" : "bg-zinc-100 text-zinc-950",
      ].join(" ")}
    >
      <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col gap-6 px-4 py-4 md:px-6 md:py-6">
        <header
          className={[
            "rounded-[2rem] border px-5 py-4 shadow-[0_24px_80px_rgba(15,23,42,0.08)]",
            isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-white",
          ].join(" ")}
        >
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div
                className={[
                  "flex h-12 w-12 items-center justify-center rounded-2xl text-xl",
                  isDark
                    ? "bg-emerald-500/10 text-emerald-300"
                    : "bg-emerald-50 text-emerald-700",
                ].join(" ")}
              >
                <PiFlaskBold />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.28em] text-emerald-500">
                  Smart Waste Local Agent
                </p>
                <h1 className="text-xl font-black tracking-[-0.06em] md:text-2xl">
                  Command center
                </h1>
              </div>
            </div>

            <nav className="flex flex-wrap gap-2">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active =
                  pathname === item.href ||
                  (item.href !== "/" && pathname.startsWith(item.href));

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={[
                      "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                      active
                        ? isDark
                          ? "bg-white text-black"
                          : "bg-zinc-950 text-white"
                        : isDark
                          ? "bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
                          : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                    ].join(" ")}
                  >
                    <Icon className="text-base" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>

            <div className="flex flex-wrap items-center gap-3">
              <div
                className={[
                  "inline-flex items-center rounded-full px-4 py-2 text-sm font-medium",
                  isDark ? "bg-zinc-900 text-zinc-300" : "bg-zinc-100 text-zinc-700",
                ].join(" ")}
              >
                {activeJobs} active job{activeJobs === 1 ? "" : "s"}
              </div>
              <button
                type="button"
                onClick={() => void refreshAll()}
                className={[
                  "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                  isDark
                    ? "bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                ].join(" ")}
              >
                <HiArrowPath />
                Refresh
              </button>
              <button
                type="button"
                onClick={toggleTheme}
                className={[
                  "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                  isDark
                    ? "bg-white text-black hover:bg-zinc-200"
                    : "bg-zinc-950 text-white hover:bg-zinc-800",
                ].join(" ")}
              >
                {isDark ? <HiSun /> : <HiMoon />}
                {isDark ? "Light" : "Dark"}
              </button>
            </div>
          </div>
        </header>

        <div className="flex-1">{children}</div>
      </div>
    </div>
  );
}
