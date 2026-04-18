import { HiBeaker, HiOutlineSquares2X2, HiSparkles } from "react-icons/hi2";

import type { PipelineCatalogResponse, PipelineFormState } from "@/lib/localagent";
import { DISCOVERY_ACTIONS } from "@/lib/localagent";

type DiscoveryControlsProps = {
  actionBlockReason: (command: string) => string | null;
  fieldClass: string;
  isDark: boolean;
  isSubmitting: string | null;
  labelClass: string;
  pipelineCatalog: PipelineCatalogResponse;
  pipelineForm: PipelineFormState;
  setPipelineField: <K extends keyof PipelineFormState>(
    field: K,
    value: PipelineFormState[K],
  ) => void;
  submitPipeline: (command: string) => Promise<void>;
};

export function DiscoveryControls({
  actionBlockReason,
  fieldClass,
  isDark,
  isSubmitting,
  labelClass,
  pipelineCatalog,
  pipelineForm,
  setPipelineField,
  submitPipeline,
}: DiscoveryControlsProps) {
  const blockedCommands = DISCOVERY_ACTIONS.flatMap((command) => {
    const reason = actionBlockReason(command);
    return reason ? [{ command, reason }] : [];
  });

  return (
    <>
      <div className="mt-5 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="review_output">
              Review export
            </label>
            <input
              id="review_output"
              className={fieldClass}
              value={pipelineForm.review_output}
              onChange={(event) => setPipelineField("review_output", event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="review_file">
              Review draft file
            </label>
            <input
              id="review_file"
              className={fieldClass}
              value={pipelineForm.review_file}
              onChange={(event) => setPipelineField("review_file", event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className={labelClass} htmlFor="num_clusters">
              Num clusters
            </label>
            <input
              id="num_clusters"
              className={fieldClass}
              placeholder="auto"
              value={pipelineForm.num_clusters}
              onChange={(event) => setPipelineField("num_clusters", event.target.value)}
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
                isDark ? "bg-zinc-900 text-emerald-300" : "bg-emerald-50 text-emerald-700",
              ].join(" ")}
            >
              <HiOutlineSquares2X2 />
            </span>
            <div>
              <p className="text-sm font-semibold">Discovery commands</p>
              <p className="text-sm text-zinc-500">
                {pipelineCatalog.dataset_commands
                  .filter((command) =>
                    DISCOVERY_ACTIONS.includes(
                      command as (typeof DISCOVERY_ACTIONS)[number],
                    ),
                  )
                  .join(", ")}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {DISCOVERY_ACTIONS.map((command) => {
          const blockReason = actionBlockReason(command);
          const isDisabled = isSubmitting !== null || blockReason !== null;
          const isPrimary = command === "embed" || command === "cluster";

          return (
            <button
              key={command}
              type="button"
              disabled={isDisabled}
              title={blockReason ?? undefined}
              onClick={() => void submitPipeline(command)}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
                isPrimary
                  ? isDark
                    ? "bg-white text-black hover:bg-zinc-200"
                    : "bg-zinc-950 text-white hover:bg-zinc-800"
                  : isDark
                    ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                isDisabled ? "opacity-60" : "",
              ].join(" ")}
            >
              {isPrimary ? <HiSparkles /> : <HiBeaker />}
              {isSubmitting === command ? `Starting ${command}...` : command}
            </button>
          );
        })}
      </div>

      {blockedCommands.length > 0 ? (
        <div
          className={[
            "mt-4 rounded-[1.2rem] border px-4 py-3",
            isDark
              ? "border-amber-900 bg-amber-950/40 text-amber-200"
              : "border-amber-200 bg-amber-50 text-amber-800",
          ].join(" ")}
        >
          <p className="text-sm font-semibold">Why some commands are disabled</p>
          <div className="mt-2 grid gap-2">
            {blockedCommands.map(({ command, reason }) => (
              <div key={command} className="text-sm leading-6">
                <span className="font-semibold">{command}</span>: {reason}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}
