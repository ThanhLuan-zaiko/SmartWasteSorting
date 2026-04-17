"use client";

import { useThemeMode } from "@/components/theme-provider";
import { formatMetric, type JobStatus } from "@/lib/localagent";

export function PageIntro({
  badge,
  title,
  description,
  actions,
}: {
  badge: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
}) {
  const { isDark } = useThemeMode();

  return (
    <section
      className={[
        "overflow-hidden rounded-[2rem] border px-6 py-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)] md:px-8",
        isDark
          ? "border-zinc-800 bg-zinc-950 text-white"
          : "border-zinc-200 bg-white text-zinc-950",
      ].join(" ")}
    >
      <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
        <div className="max-w-3xl">
          <span
            className={[
              "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em]",
              isDark
                ? "bg-zinc-900 text-emerald-300"
                : "bg-emerald-50 text-emerald-700",
            ].join(" ")}
          >
            {badge}
          </span>
          <h1 className="mt-5 text-4xl font-black tracking-[-0.08em] md:text-6xl">
            {title}
          </h1>
          <p
            className={[
              "mt-4 max-w-2xl text-sm leading-7 md:text-base",
              isDark ? "text-zinc-300" : "text-zinc-600",
            ].join(" ")}
          >
            {description}
          </p>
        </div>
        {actions ? <div className="flex flex-wrap gap-3">{actions}</div> : null}
      </div>
    </section>
  );
}

export function Panel({
  title,
  description,
  actions,
  children,
  className = "",
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  const { isDark } = useThemeMode();

  return (
    <section
      className={[
        "rounded-[1.75rem] border p-5 shadow-[0_18px_60px_rgba(15,23,42,0.08)] md:p-6",
        isDark
          ? "border-zinc-800 bg-zinc-900 text-white"
          : "border-zinc-200 bg-white text-zinc-950",
        className,
      ].join(" ")}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-2xl">
          <h2 className="text-2xl font-bold tracking-[-0.04em]">{title}</h2>
          {description ? (
            <p className={["mt-2 text-sm leading-6", isDark ? "text-zinc-400" : "text-zinc-600"].join(" ")}>
              {description}
            </p>
          ) : null}
        </div>
        {actions ? <div className="flex flex-wrap gap-3">{actions}</div> : null}
      </div>
      <div className="mt-6">{children}</div>
    </section>
  );
}

export function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: unknown;
  hint?: string;
}) {
  const { isDark } = useThemeMode();

  return (
    <article
      className={[
        "rounded-[1.4rem] border p-4",
        isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
      ].join(" ")}
    >
      <span
        className={[
          "text-[11px] font-semibold uppercase tracking-[0.24em]",
          isDark ? "text-zinc-500" : "text-zinc-500",
        ].join(" ")}
      >
        {label}
      </span>
      <div className="mt-3 text-3xl font-black tracking-[-0.08em]">
        {formatMetric(value)}
      </div>
      {hint ? (
        <p className={["mt-2 text-sm", isDark ? "text-zinc-400" : "text-zinc-600"].join(" ")}>
          {hint}
        </p>
      ) : null}
    </article>
  );
}

export function StatusBadge({ status }: { status: JobStatus | string | null | undefined }) {
  const { isDark } = useThemeMode();

  const tone =
    status === "completed"
      ? isDark
        ? "border-emerald-800 bg-emerald-950 text-emerald-300"
        : "border-emerald-200 bg-emerald-50 text-emerald-700"
      : status === "failed" || status === "cancelled"
        ? isDark
          ? "border-rose-900 bg-rose-950 text-rose-300"
          : "border-rose-200 bg-rose-50 text-rose-700"
        : isDark
          ? "border-amber-900 bg-amber-950 text-amber-300"
          : "border-amber-200 bg-amber-50 text-amber-700";

  return (
    <span
      className={[
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]",
        tone,
      ].join(" ")}
    >
      {status ?? "unknown"}
    </span>
  );
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  const { isDark } = useThemeMode();

  return (
    <div
      className={[
        "rounded-[1.5rem] border border-dashed p-6 text-center",
        isDark
          ? "border-zinc-800 bg-zinc-950/50 text-zinc-300"
          : "border-zinc-300 bg-zinc-50 text-zinc-600",
      ].join(" ")}
    >
      <h3 className="text-lg font-semibold text-inherit">{title}</h3>
      <p className="mt-2 text-sm leading-6">{description}</p>
    </div>
  );
}
