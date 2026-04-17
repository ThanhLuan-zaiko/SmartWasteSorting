"use client";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel } from "@/components/ui/primitives";
import { asObject, formatMetric } from "@/lib/localagent";

export function ClassMetricsTable({
  evaluation,
}: {
  evaluation: Record<string, unknown> | null | undefined;
}) {
  const { isDark } = useThemeMode();
  const perClass = asObject(evaluation?.per_class);
  const rows = perClass ? Object.entries(perClass) : [];

  return (
    <Panel
      title="Per-class report"
      description="Precision, recall, F1, and support from the evaluation artifact."
    >
      {rows.length === 0 ? (
        <EmptyState
          title="No class metrics"
          description="The table will appear once evaluate has written the per-class report."
        />
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full border-separate border-spacing-y-2">
            <thead>
              <tr>
                {["Class", "Precision", "Recall", "F1", "Support"].map((header) => (
                  <th
                    key={header}
                    className={[
                      "px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.22em]",
                      isDark ? "text-zinc-500" : "text-zinc-500",
                    ].join(" ")}
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, payload]) => {
                const metrics = asObject(payload);
                return (
                  <tr
                    key={label}
                    className={[
                      isDark ? "bg-zinc-950/70" : "bg-zinc-50",
                    ].join(" ")}
                  >
                    <td className="rounded-l-2xl px-4 py-4 font-semibold">{label}</td>
                    <td className="px-4 py-4">{formatMetric(metrics?.precision)}</td>
                    <td className="px-4 py-4">{formatMetric(metrics?.recall)}</td>
                    <td className="px-4 py-4">{formatMetric(metrics?.f1)}</td>
                    <td className="rounded-r-2xl px-4 py-4">
                      {formatMetric(metrics?.support)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
