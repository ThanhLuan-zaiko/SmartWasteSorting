"use client";

import { HiDocumentDuplicate, HiQueueList, HiSparkles } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { Panel } from "@/components/ui/primitives";
import { DATASET_ACTIONS } from "@/lib/localagent";

export function PipelineActions() {
  const { isDark } = useThemeMode();
  const {
    pipelineCatalog,
    pipelineForm,
    isSubmitting,
    setPipelineField,
    submitPipeline,
  } = useLocalAgent();

  const fieldClass = [
    "min-h-12 rounded-2xl border px-4 py-3 text-sm outline-none transition",
    isDark
      ? "border-zinc-800 bg-zinc-950 text-zinc-100 placeholder:text-zinc-500"
      : "border-zinc-200 bg-zinc-50 text-zinc-900 placeholder:text-zinc-400",
  ].join(" ");

  return (
    <Panel
      title="Dataset pipeline"
      description="Trigger scan, labeling, validation, and manifest/report steps from the UI instead of typing the CLI repeatedly."
    >
      <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label
              className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500"
              htmlFor="labels_file"
            >
              Labels file
            </label>
            <input
              id="labels_file"
              className={fieldClass}
              value={pipelineForm.labels_file}
              onChange={(event) => setPipelineField("labels_file", event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label
              className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500"
              htmlFor="template_output"
            >
              Template output
            </label>
            <input
              id="template_output"
              className={fieldClass}
              value={pipelineForm.output}
              onChange={(event) => setPipelineField("output", event.target.value)}
            />
          </div>
        </div>

        <div
          className={[
            "rounded-[1.5rem] border p-4",
            isDark ? "border-zinc-800 bg-zinc-950/70" : "border-zinc-200 bg-zinc-50",
          ].join(" ")}
        >
          <div className="flex items-center gap-3">
            <span
              className={[
                "rounded-2xl p-3 text-lg",
                isDark
                  ? "bg-zinc-900 text-emerald-300"
                  : "bg-emerald-50 text-emerald-700",
              ].join(" ")}
            >
              <HiQueueList />
            </span>
            <div>
              <p className="text-sm font-semibold">Catalog from backend</p>
              <p className="text-sm text-zinc-500">
                {pipelineCatalog.dataset_commands.join(", ")}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {DATASET_ACTIONS.map((command) => (
          <button
            key={command}
            type="button"
            disabled={isSubmitting !== null}
            onClick={() => void submitPipeline(command)}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              command === "run-all"
                ? isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800"
                : isDark
                  ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              isSubmitting !== null ? "opacity-60" : "",
            ].join(" ")}
          >
            {command === "run-all" ? <HiSparkles /> : <HiDocumentDuplicate />}
            {isSubmitting === command ? `Starting ${command}...` : command}
          </button>
        ))}
      </div>
    </Panel>
  );
}
