"use client";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel } from "@/components/ui/primitives";
import { asArray, asNumber, asString } from "@/lib/localagent";

export function ConfusionMatrix({
  evaluation,
}: {
  evaluation: Record<string, unknown> | null | undefined;
}) {
  const { isDark } = useThemeMode();
  const labels = asArray(evaluation?.labels).map((value) => asString(value) ?? "");
  const matrix = asArray(evaluation?.confusion_matrix).map((row) =>
    asArray(row).map((value) => asNumber(value) ?? 0),
  );

  return (
    <Panel
      title="Confusion matrix"
      description="Heatmap view of the selected evaluation artifact."
    >
      {labels.length === 0 || matrix.length === 0 ? (
        <EmptyState
          title="No evaluation matrix"
          description="Run evaluate and the confusion matrix will render here."
        />
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full border-separate border-spacing-2">
            <thead>
              <tr>
                <th
                  className={[
                    "rounded-2xl px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.22em]",
                    isDark ? "bg-zinc-950 text-zinc-400" : "bg-zinc-100 text-zinc-500",
                  ].join(" ")}
                >
                  Label
                </th>
                {labels.map((label) => (
                  <th
                    key={label}
                    className={[
                      "rounded-2xl px-4 py-3 text-center text-xs font-semibold uppercase tracking-[0.22em]",
                      isDark ? "bg-zinc-950 text-zinc-400" : "bg-zinc-100 text-zinc-500",
                    ].join(" ")}
                  >
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIndex) => {
                const maxCell = Math.max(...row, 1);
                return (
                  <tr key={labels[rowIndex] ?? rowIndex}>
                    <th
                      className={[
                        "rounded-2xl px-4 py-3 text-left text-sm font-semibold",
                        isDark ? "bg-zinc-950 text-zinc-200" : "bg-zinc-50 text-zinc-700",
                      ].join(" ")}
                    >
                      {labels[rowIndex] ?? `Class ${rowIndex + 1}`}
                    </th>
                    {row.map((value, columnIndex) => (
                      <td
                        key={`${rowIndex}-${columnIndex}`}
                        className="rounded-2xl px-4 py-4 text-center text-sm font-bold"
                        style={{
                          backgroundColor: isDark
                            ? `rgba(20, 184, 166, ${0.14 + value / maxCell / 1.7})`
                            : `rgba(16, 185, 129, ${0.10 + value / maxCell / 2.2})`,
                        }}
                      >
                        {value}
                      </td>
                    ))}
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
