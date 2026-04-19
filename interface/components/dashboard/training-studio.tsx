"use client";

import { HiBeaker, HiBolt, HiPlayCircle, HiSparkles } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { Panel } from "@/components/ui/primitives";
import { asObject, TRAINING_ACTIONS } from "@/lib/localagent";

const MODEL_OPTIONS = [
  "mobilenet_v3_small",
  "mobilenet_v3_large",
  "resnet18",
  "efficientnet_b0",
];

const DATASET_REQUIRED_ACTIONS = new Set([
  "warm-cache",
  "pseudo-label",
  "fit",
  "evaluate",
  "benchmark",
]);

const MULTI_CLASS_REQUIRED_ACTIONS = new Set(["fit", "benchmark"]);
const PYTORCH_ONLY_ACTIONS = new Set(["fit", "evaluate", "export-onnx"]);
const CHECKPOINT_REQUIRED_ACTIONS = new Set(["pseudo-label", "evaluate", "export-onnx"]);

export function TrainingStudio() {
  const { isDark } = useThemeMode();
  const {
    runs,
    runDetail,
    workflowState,
    trainingPresets,
    trainingForm,
    isSubmitting,
    setTrainingField,
    applyPreset,
    submitTraining,
  } = useLocalAgent();

  const fieldClass = [
    "min-h-12 rounded-2xl border px-4 py-3 text-sm outline-none transition",
    isDark
      ? "border-zinc-800 bg-zinc-950 text-zinc-100 placeholder:text-zinc-500"
      : "border-zinc-200 bg-zinc-50 text-zinc-900 placeholder:text-zinc-400",
  ].join(" ");

  const labelClass = "text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500";
  const dashboardSummary = asObject(runDetail?.dashboard_summary);
  const status = asObject(dashboardSummary?.status);
  const datasetSummary = asObject(workflowState?.dataset_summary);
  const labelCounts = asObject(datasetSummary?.label_counts);
  const acceptedLabelSourceCounts = asObject(datasetSummary?.accepted_label_source_counts);
  const trainableLabelCounts = asObject(datasetSummary?.trainable_label_counts);
  const datasetReady = workflowState?.steps.dataset?.completed === true;
  const datasetReadyKnown = workflowState !== null;
  const trainingStepEnabled = workflowState?.steps.training?.enabled === true;
  const trainingStepKnown = workflowState !== null;
  const trainingStepReason = workflowState?.steps.training?.reason ?? null;
  const trainingReady = status?.training_ready === true;
  const rustPreviewSelected = trainingForm.training_backend === "rust_tch";
  const detectedLabels = Object.keys(labelCounts ?? {}).filter((label) => label !== "unknown");
  const trainableLabels = Object.keys(trainableLabelCounts ?? {}).filter(
    (label) => label !== "unknown",
  );
  const acceptedSourceSummary = Object.entries(acceptedLabelSourceCounts ?? {})
    .map(([label, count]) => `${label}: ${count}`)
    .join(", ");
  const effectiveTrainingMode =
    typeof datasetSummary?.effective_training_mode === "string"
      ? datasetSummary.effective_training_mode
      : "weak_inferred";
  const trainingStudioStatus = !trainingStepKnown
    ? "checking"
    : trainingStepEnabled
      ? "unlocked"
      : "locked";

  function actionBlockReason(command: string): string | null {
    if (!workflowState) {
      return "Checking workflow state.";
    }
    const workflowReason = workflowState.commands[command]?.reason ?? null;
    if (workflowReason) {
      return workflowReason;
    }
    if (
      datasetReadyKnown &&
      !datasetReady &&
      DATASET_REQUIRED_ACTIONS.has(command)
    ) {
      return workflowState.steps.training?.reason ?? "Complete Step 2 first.";
    }
    if (
      MULTI_CLASS_REQUIRED_ACTIONS.has(command) &&
      trainableLabels.length < 2
    ) {
      return `Training requires at least 2 trainable labels. The current manifest only contains: ${trainableLabels.join(", ") || "none"}.`;
    }
    if (CHECKPOINT_REQUIRED_ACTIONS.has(command) && !trainingReady) {
      return "Run an initial `fit` first so a checkpoint exists for this step.";
    }
    if (
      trainingForm.training_backend === "rust_tch" &&
      (PYTORCH_ONLY_ACTIONS.has(command) || command === "pseudo-label")
    ) {
      return "Choose `pytorch` for this step. `rust_tch` is preview-only for summary, export-spec, and benchmark.";
    }
    return null;
  }

  return (
    <Panel
      title="Step 3: Training studio"
      description="Once discovery has produced accepted labels, use these controls to pseudo-label the remaining pool, train the classifier, evaluate it, export ONNX, and run benchmarks."
      actions={
        <div className="flex flex-wrap gap-2">
          {Object.keys(trainingPresets).map((presetName) => (
            <button
              key={presetName}
              type="button"
              onClick={() => applyPreset(presetName)}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              ].join(" ")}
            >
              <HiSparkles />
              {presetName}
            </button>
          ))}
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="experiment_name">
            Experiment
          </label>
          <input
            id="experiment_name"
            className={fieldClass}
            value={trainingForm.experiment_name}
            onChange={(event) => setTrainingField("experiment_name", event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="training_preset">
            Preset
          </label>
          <select
            id="training_preset"
            className={fieldClass}
            value={trainingForm.training_preset}
            onChange={(event) => applyPreset(event.target.value)}
          >
            {Object.keys(trainingPresets).map((presetName) => (
              <option key={presetName} value={presetName}>
                {presetName}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="training_backend">
            Backend
          </label>
          <select
            id="training_backend"
            className={fieldClass}
            value={trainingForm.training_backend}
            onChange={(event) => setTrainingField("training_backend", event.target.value)}
          >
            <option value="pytorch">pytorch</option>
            <option value="rust_tch">rust_tch (preview)</option>
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="model_name">
            Model
          </label>
          <select
            id="model_name"
            className={fieldClass}
            value={trainingForm.model_name}
            onChange={(event) => setTrainingField("model_name", event.target.value)}
          >
            {MODEL_OPTIONS.map((modelName) => (
              <option key={modelName} value={modelName}>
                {modelName}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="device">
            Device
          </label>
          <select
            id="device"
            className={fieldClass}
            value={trainingForm.device}
            onChange={(event) => setTrainingField("device", event.target.value)}
          >
            <option value="auto">auto</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="image_size">
            Image size
          </label>
          <input
            id="image_size"
            className={fieldClass}
            value={trainingForm.image_size}
            onChange={(event) => setTrainingField("image_size", event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="batch_size">
            Batch size
          </label>
          <input
            id="batch_size"
            className={fieldClass}
            value={trainingForm.batch_size}
            onChange={(event) => setTrainingField("batch_size", event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="epochs">
            Epochs
          </label>
          <input
            id="epochs"
            className={fieldClass}
            value={trainingForm.epochs}
            onChange={(event) => setTrainingField("epochs", event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="class_bias">
            Class bias
          </label>
          <select
            id="class_bias"
            className={fieldClass}
            value={trainingForm.class_bias}
            onChange={(event) => setTrainingField("class_bias", event.target.value)}
          >
            <option value="none">none</option>
            <option value="loss">loss</option>
            <option value="sampler">sampler</option>
            <option value="both">both</option>
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="compare_experiment">
            Benchmark against
          </label>
          <select
            id="compare_experiment"
            className={fieldClass}
            value={trainingForm.compare_experiment}
            onChange={(event) => setTrainingField("compare_experiment", event.target.value)}
          >
            <option value="">No comparison</option>
            {runs
              .filter((run) => run.experiment_name !== trainingForm.experiment_name)
              .map((run) => (
                <option key={run.experiment_name} value={run.experiment_name}>
                  {run.experiment_name}
                </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="pseudo_label_threshold">
            Pseudo-label confidence
          </label>
          <input
            id="pseudo_label_threshold"
            className={fieldClass}
            value={trainingForm.pseudo_label_threshold}
            onChange={(event) => setTrainingField("pseudo_label_threshold", event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className={labelClass} htmlFor="pseudo_label_margin">
            Pseudo-label margin
          </label>
          <input
            id="pseudo_label_margin"
            className={fieldClass}
            value={trainingForm.pseudo_label_margin}
            onChange={(event) => setTrainingField("pseudo_label_margin", event.target.value)}
          />
        </div>
      </div>

      <div
        className={[
          "mt-5 rounded-[1.2rem] border px-4 py-3 text-sm leading-6",
          isDark
            ? "border-zinc-800 bg-zinc-950 text-zinc-300"
            : "border-zinc-200 bg-zinc-50 text-zinc-700",
        ].join(" ")}
      >
        <p>
          Training studio:{" "}
          <span className="font-semibold">
            {trainingStudioStatus}
          </span>
          .{" "}
          {trainingStepKnown
            ? trainingStepEnabled
              ? "Discovery labels have already been promoted into the manifest. Individual actions can still be gated by backend choice or missing checkpoints."
              : (trainingStepReason ?? "Complete Step 2 first.")
            : "Checking workflow state."}
        </p>
        <p className="mt-2">
          Dataset manifest:{" "}
          <span className="font-semibold">
            {datasetReadyKnown ? (datasetReady ? "ready" : "missing") : "checking"}
          </span>
          . `warm-cache` only prebuilds the Rust image cache; it does not unlock `fit` by itself.
        </p>
        <p className="mt-2">
          `rust_tch` is preview-only in this build. Actual `fit`, `pseudo-label`, `evaluate`, and
          `export-onnx` still require `pytorch`.
        </p>
        <p className="mt-2">
          Effective training mode: <span className="font-semibold">{effectiveTrainingMode}</span>.
        </p>
        {rustPreviewSelected ? (
          <p className="mt-2">
            Selected backend: <span className="font-semibold">rust_tch</span>. Switch to{" "}
            <span className="font-semibold">pytorch</span> to enable `fit`, `pseudo-label`,
            `evaluate`, and `export-onnx`.
          </p>
        ) : null}
        {!trainingReady ? (
          <p className="mt-2">
            `pseudo-label`, `evaluate`, and `export-onnx` stay disabled until an initial `fit`
            writes a checkpoint for this experiment.
          </p>
        ) : null}
        {detectedLabels.length > 0 ? (
          <p className="mt-2">
            Detected labels: <span className="font-semibold">{detectedLabels.join(", ")}</span>
            {detectedLabels.length < 2
              ? ". Add at least one more class or import curated labels before running fit."
              : "."}
          </p>
        ) : null}
        {trainableLabels.length > 0 ? (
          <p className="mt-2">
            Trainable labels: <span className="font-semibold">{trainableLabels.join(", ")}</span>.
          </p>
        ) : null}
        {acceptedSourceSummary ? (
          <p className="mt-2">
            Accepted label sources: <span className="font-semibold">{acceptedSourceSummary}</span>.
          </p>
        ) : null}
        <label className="mt-3 flex items-start gap-3">
          <input
            type="checkbox"
            className="mt-1 size-4 rounded border-zinc-300"
            checked={trainingForm.no_progress}
            onChange={(event) => setTrainingField("no_progress", event.target.checked)}
          />
          <span>
            <span className="font-semibold">Use --no-progress</span>
            <span
              className={[
                "block text-sm",
                isDark ? "text-zinc-400" : "text-zinc-600",
              ].join(" ")}
            >
              Turn this off to stream detailed progress lines into the Jobs websocket while
              training runs.
            </span>
          </span>
        </label>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {TRAINING_ACTIONS.map((command) => {
          const blockReason = actionBlockReason(command);
          const isDisabled = isSubmitting !== null || blockReason !== null;

          return (
            <button
              key={command}
              type="button"
              disabled={isDisabled}
              title={blockReason ?? undefined}
              onClick={() => void submitTraining(command)}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
                command === "fit" || command === "pseudo-label"
                  ? isDark
                    ? "bg-white text-black hover:bg-zinc-200"
                    : "bg-zinc-950 text-white hover:bg-zinc-800"
                  : isDark
                    ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                isDisabled ? "opacity-60" : "",
              ].join(" ")}
            >
              {command === "fit" ? <HiPlayCircle /> : <HiBeaker />}
              {isSubmitting === command ? `Starting ${command}...` : command}
            </button>
          );
        })}
        <button
          type="button"
          disabled={isSubmitting !== null || actionBlockReason("benchmark") !== null}
          title={actionBlockReason("benchmark") ?? undefined}
          onClick={() => void submitTraining("benchmark")}
          className={[
            "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
            isDark
              ? "bg-emerald-400 text-black hover:bg-emerald-300"
              : "bg-emerald-500 text-white hover:bg-emerald-600",
            isSubmitting !== null || actionBlockReason("benchmark") !== null
              ? "opacity-60"
              : "",
          ].join(" ")}
        >
          <HiBolt />
          {isSubmitting === "benchmark" ? "Starting benchmark..." : "benchmark"}
        </button>
      </div>
    </Panel>
  );
}
