"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, Panel } from "@/components/ui/primitives";
import { asArray, asNumber, asObject } from "@/lib/localagent";

const SERIES = [
  { key: "train_loss", name: "Train loss", stroke: "#14b8a6" },
  { key: "val_loss", name: "Val loss", stroke: "#f97316" },
  { key: "train_accuracy", name: "Train accuracy", stroke: "#3b82f6" },
  { key: "val_accuracy", name: "Val accuracy", stroke: "#ef4444" },
] as const;

export function TrainingHistoryChart({
  training,
}: {
  training: Record<string, unknown> | null | undefined;
}) {
  const { isDark } = useThemeMode();
  const history = asArray(training?.history).map((entry) => {
    const item = asObject(entry);
    return {
      epoch: asNumber(item?.epoch) ?? 0,
      train_loss: asNumber(item?.train_loss),
      val_loss: asNumber(item?.val_loss),
      train_accuracy: asNumber(item?.train_accuracy),
      val_accuracy: asNumber(item?.val_accuracy),
    };
  });

  return (
    <Panel
      title="Training history"
      description="Loss and accuracy curves from the persisted training artifact."
    >
      {history.length === 0 ? (
        <EmptyState
          title="No history yet"
          description="Run fit from the pipeline page and the chart will appear here."
        />
      ) : (
        <div className="h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid
                stroke={isDark ? "rgba(255,255,255,0.08)" : "rgba(24,24,27,0.08)"}
                strokeDasharray="4 4"
              />
              <XAxis
                dataKey="epoch"
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
                contentStyle={{
                  borderRadius: 18,
                  border: "1px solid rgba(113,113,122,0.18)",
                  background: isDark ? "#09090b" : "#ffffff",
                  color: isDark ? "#fafafa" : "#09090b",
                }}
              />
              <Legend />
              {SERIES.map((series) => (
                <Line
                  key={series.key}
                  type="monotone"
                  dataKey={series.key}
                  name={series.name}
                  stroke={series.stroke}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}
