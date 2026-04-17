"use client";

import { HiBeaker, HiBolt, HiPlayCircle, HiSparkles } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";
import { Panel } from "@/components/ui/primitives";
import { TRAINING_ACTIONS } from "@/lib/localagent";

const MODEL_OPTIONS = [
  "mobilenet_v3_small",
  "mobilenet_v3_large",
  "resnet18",
  "efficientnet_b0",
];

export function TrainingStudio() {
  const { isDark } = useThemeMode();
  const {
    runs,
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

  return (
    <Panel
      title="Training studio"
      description="Preset-driven run setup with manual control over model, epoch, batch, bias, backend, and benchmark comparison."
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
            <option value="rust_tch">rust_tch</option>
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
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {TRAINING_ACTIONS.map((command) => (
          <button
            key={command}
            type="button"
            disabled={isSubmitting !== null}
            onClick={() => void submitTraining(command)}
            className={[
              "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
              command === "fit"
                ? isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800"
                : isDark
                  ? "bg-zinc-950 text-zinc-300 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
              isSubmitting !== null ? "opacity-60" : "",
            ].join(" ")}
          >
            {command === "fit" ? <HiPlayCircle /> : <HiBeaker />}
            {isSubmitting === command ? `Starting ${command}...` : command}
          </button>
        ))}
        <button
          type="button"
          disabled={isSubmitting !== null}
          onClick={() => void submitTraining("benchmark")}
          className={[
            "inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold transition",
            isDark
              ? "bg-emerald-400 text-black hover:bg-emerald-300"
              : "bg-emerald-500 text-white hover:bg-emerald-600",
            isSubmitting !== null ? "opacity-60" : "",
          ].join(" ")}
        >
          <HiBolt />
          {isSubmitting === "benchmark" ? "Starting benchmark..." : "benchmark"}
        </button>
      </div>
    </Panel>
  );
}
