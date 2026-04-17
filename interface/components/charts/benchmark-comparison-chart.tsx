"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel } from "@/components/ui/primitives";
import { formatDelta, type CompareResponse } from "@/lib/localagent";

const METRICS = [
  { key: "duration_delta_seconds", label: "Duration" },
  { key: "fit_stage_delta_seconds", label: "Fit" },
  { key: "accuracy_delta", label: "Accuracy" },
  { key: "macro_f1_delta", label: "Macro F1" },
  { key: "weighted_f1_delta", label: "Weighted F1" },
] as const;

export function BenchmarkComparisonChart({
  comparison,
  leftExperiment,
  rightExperiment,
}: {
  comparison: CompareResponse | null;
  leftExperiment: string;
  rightExperiment: string;
}) {
  const { isDark } = useThemeMode();
  const data = METRICS.map((metric) => ({
    label: metric.label,
    value: comparison?.[metric.key] ?? null,
  }));

  return (
    <Panel
      title="Benchmark comparison"
      description={`Comparing ${leftExperiment || "no run"} against ${rightExperiment || "no run"}.`}
    >
      {!comparison ? (
        <EmptyState
          title="Pick two runs"
          description="Choose two experiments with benchmark artifacts to render the comparison chart."
        />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[1.4fr_0.9fr]">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <CartesianGrid
                  stroke={isDark ? "rgba(255,255,255,0.08)" : "rgba(24,24,27,0.08)"}
                  strokeDasharray="4 4"
                />
                <XAxis
                  dataKey="label"
                  stroke={isDark ? "#a1a1aa" : "#71717a"}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke={isDark ? "#a1a1aa" : "#71717a"}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  formatter={(value) =>
                    formatDelta(
                      typeof value === "number"
                        ? value
                        : Number.isFinite(Number(value))
                          ? Number(value)
                          : null,
                    )
                  }
                  contentStyle={{
                    borderRadius: 18,
                    border: "1px solid rgba(113,113,122,0.18)",
                    background: isDark ? "#09090b" : "#ffffff",
                    color: isDark ? "#fafafa" : "#09090b",
                  }}
                />
                <Bar dataKey="value" radius={[12, 12, 4, 4]} fill="#14b8a6" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid gap-3">
            {METRICS.map((metric) => (
              <article
                key={metric.key}
                className={[
                  "rounded-[1.35rem] border p-4",
                  isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
                ].join(" ")}
              >
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">
                  {metric.label}
                </div>
                <div className="mt-2 text-3xl font-black tracking-[-0.08em]">
                  {formatDelta(comparison?.[metric.key] ?? null)}
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
