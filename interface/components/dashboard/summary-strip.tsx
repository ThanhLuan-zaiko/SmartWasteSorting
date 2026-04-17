"use client";

import { MetricCard } from "@/components/ui/primitives";

export function SummaryStrip({
  cards,
}: {
  cards: Record<string, unknown> | null;
}) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        label="Best loss"
        value={cards?.best_loss}
        hint="Smallest tracked loss for the selected experiment."
      />
      <MetricCard
        label="Best epoch"
        value={cards?.best_epoch}
        hint="Epoch where the best checkpoint was captured."
      />
      <MetricCard
        label="Accuracy"
        value={cards?.accuracy}
        hint="Top-level classification accuracy from evaluation."
      />
      <MetricCard
        label="Macro F1"
        value={cards?.macro_f1}
        hint="Macro-balanced F1 across all waste classes."
      />
      <MetricCard label="Weighted F1" value={cards?.weighted_f1} />
      <MetricCard
        label="Backend"
        value={cards?.training_backend}
        hint="Current training backend for this run."
      />
      <MetricCard
        label="Peak RSS (MB)"
        value={cards?.peak_stage_rss_mb}
        hint="Memory peak captured during benchmark stages."
      />
      <MetricCard
        label="ONNX verified"
        value={cards?.onnx_verified}
        hint="Whether exported ONNX passed runtime verification."
      />
    </div>
  );
}
