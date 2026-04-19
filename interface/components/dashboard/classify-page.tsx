"use client";
/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from "react";
import { HiArrowUpRight, HiBeaker, HiPhoto, HiSparkles } from "react-icons/hi2";

import { useThemeMode } from "@/components/theme-provider";
import { EmptyState, MetricCard, PageIntro, Panel } from "@/components/ui/primitives";
import {
  asArray,
  asObject,
  asString,
  fetchJson,
  formatMetric,
  type ArtifactEnvelope,
  type ImageClassificationResponse,
  type JsonObject,
} from "@/lib/localagent";

const TOP_K = 3;

export function ClassifyPage() {
  const { isDark } = useThemeMode();
  const [modelManifest, setModelManifest] = useState<JsonObject | null>(null);
  const [manifestError, setManifestError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ImageClassificationResponse | null>(null);
  const [classifyError, setClassifyError] = useState<string | null>(null);
  const [isLoadingManifest, setIsLoadingManifest] = useState(true);
  const [isClassifying, setIsClassifying] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadModelManifest() {
      setIsLoadingManifest(true);
      try {
        const envelope = await fetchJson<ArtifactEnvelope<JsonObject>>("/artifacts/model-manifest");
        if (!active) {
          return;
        }
        setModelManifest(asObject(envelope.payload));
        setManifestError(null);
      } catch (error) {
        if (!active) {
          return;
        }
        setModelManifest(null);
        setManifestError(error instanceof Error ? error.message : "Failed to load model manifest.");
      } finally {
        if (active) {
          setIsLoadingManifest(false);
        }
      }
    }

    void loadModelManifest();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const verification = asObject(asObject(modelManifest?.onnx)?.verification);
  const evaluationSummary = asObject(modelManifest?.evaluation_summary);
  const labels = asArray(modelManifest?.labels)
    .map((value) => asString(value))
    .filter((value): value is string => value !== null);
  const topPrediction = result?.predictions[0] ?? null;

  async function handleClassify() {
    if (!selectedFile) {
      setClassifyError("Choose an image first.");
      return;
    }

    setIsClassifying(true);
    setClassifyError(null);
    try {
      const imageBase64 = await readFileAsDataUrl(selectedFile);
      const payload = await fetchJson<ImageClassificationResponse>("/classify/image", {
        method: "POST",
        body: JSON.stringify({
          image_base64: imageBase64,
          file_name: selectedFile.name,
          top_k: TOP_K,
        }),
      });
      setResult(payload);
    } catch (error) {
      setResult(null);
      setClassifyError(error instanceof Error ? error.message : "Failed to classify image.");
    } finally {
      setIsClassifying(false);
    }
  }

  return (
    <div className="grid gap-6">
      <PageIntro
        badge="Classify"
        title="Use the exported ONNX model from the GUI"
        description="Upload a waste image, run the exported ONNX model through the local Actix server, and inspect the top class probabilities before you test with real trash photos."
        actions={
          <>
            <label
              htmlFor="waste-image-input"
              className={[
                "inline-flex cursor-pointer items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-950 text-white hover:bg-zinc-800",
              ].join(" ")}
            >
              <HiPhoto />
              Choose image
            </label>
            <button
              type="button"
              onClick={() => void handleClassify()}
              disabled={selectedFile === null || isClassifying || modelManifest === null}
              className={[
                "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition",
                isDark
                  ? "bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200",
                selectedFile === null || isClassifying || modelManifest === null ? "opacity-60" : "",
              ].join(" ")}
            >
              <HiSparkles />
              {isClassifying ? "Classifying..." : "Run ONNX classify"}
            </button>
          </>
        }
      />

      <input
        id="waste-image-input"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          setSelectedFile(file);
          setResult(null);
          setClassifyError(null);
        }}
      />

      <Panel
        title="Runtime status"
        description="This page uses the exported ONNX artifact and model manifest, not the PyTorch checkpoint."
      >
        {isLoadingManifest ? (
          <p className="text-sm text-zinc-500">Loading model manifest...</p>
        ) : manifestError ? (
          <EmptyState
            title="Model manifest unavailable"
            description={manifestError}
          />
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Model" value={modelManifest?.model_name} />
              <MetricCard label="Image size" value={modelManifest?.image_size} />
              <MetricCard label="Accuracy" value={evaluationSummary?.accuracy} />
              <MetricCard label="Macro F1" value={evaluationSummary?.macro_f1} />
            </div>
            <div
              className={[
                "mt-5 rounded-[1.4rem] border px-4 py-4 text-sm leading-6",
                isDark
                  ? "border-zinc-800 bg-zinc-950 text-zinc-300"
                  : "border-zinc-200 bg-zinc-50 text-zinc-700",
              ].join(" ")}
            >
              <p>
                ONNX verification: <span className="font-semibold">{formatMetric(verification?.verified)}</span>
                {" "}via{" "}
                <span className="font-semibold">{formatMetric(verification?.provider)}</span>.
              </p>
              <p className="mt-2">
                Labels: <span className="font-semibold">{labels.join(", ") || "N/A"}</span>.
              </p>
              <p className="mt-2">
                Export path: <span className="font-semibold break-all">{formatMetric(modelManifest?.onnx_path)}</span>.
              </p>
            </div>
          </>
        )}
      </Panel>

      <div className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
        <Panel
          title="Upload preview"
          description="Choose a real image from disk. The server resizes it to the exported input size and applies the same ImageNet normalization stored in the manifest."
        >
          {previewUrl ? (
            <div className="grid gap-4">
              <div
                className={[
                  "overflow-hidden rounded-[1.6rem] border",
                  isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-zinc-50",
                ].join(" ")}
              >
                <img
                  src={previewUrl}
                  alt={selectedFile?.name ?? "Selected waste image"}
                  className="h-[320px] w-full object-contain"
                />
              </div>
              <div className="text-sm leading-6 text-zinc-500">
                <p className={["font-semibold", isDark ? "text-zinc-300" : "text-zinc-700"].join(" ")}>
                  {selectedFile?.name}
                </p>
                <p>Top {TOP_K} predictions will be shown after ONNX inference completes.</p>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No image selected"
              description="Pick a waste photo from your machine, then run ONNX classify."
            />
          )}
        </Panel>

        <Panel
          title="Predictions"
          description="Scores come from the ONNX logits after softmax. Higher is more confident."
        >
          {classifyError ? (
            <EmptyState title="Classification failed" description={classifyError} />
          ) : !result ? (
            <EmptyState
              title="No prediction yet"
              description="Run the ONNX classifier to inspect the top class probabilities."
            />
          ) : (
            <div className="grid gap-5">
              <div
                className={[
                  "rounded-[1.5rem] border p-5",
                  isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-zinc-50",
                ].join(" ")}
              >
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">
                  Top prediction
                </p>
                <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
                  <div>
                    <h2 className="text-4xl font-black tracking-[-0.08em]">
                      {topPrediction?.label ?? "N/A"}
                    </h2>
                    <p className="mt-2 text-sm text-zinc-500">
                      Backend {result.backend} · {result.model_name} · {result.image_size}px
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">
                      Confidence
                    </div>
                    <div className="mt-2 text-3xl font-black tracking-[-0.08em]">
                      {topPrediction ? `${(topPrediction.score * 100).toFixed(1)}%` : "N/A"}
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid gap-4">
                {result.predictions.map((prediction) => (
                  <article
                    key={prediction.label}
                    className={[
                      "rounded-[1.35rem] border p-4",
                      isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-zinc-50",
                    ].join(" ")}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-3">
                        <span
                          className={[
                            "inline-flex rounded-2xl p-3 text-lg",
                            isDark
                              ? "bg-zinc-900 text-emerald-300"
                              : "bg-emerald-50 text-emerald-700",
                          ].join(" ")}
                        >
                          <HiBeaker />
                        </span>
                        <div>
                          <h3 className="text-lg font-bold">{prediction.label}</h3>
                          <p className="text-sm text-zinc-500">
                            Probability after ONNX softmax.
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-black tracking-[-0.06em]">
                          {(prediction.score * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div
                      className={[
                        "mt-4 h-3 overflow-hidden rounded-full",
                        isDark ? "bg-zinc-800" : "bg-zinc-200",
                      ].join(" ")}
                    >
                      <div
                        className={isDark ? "h-full bg-emerald-300" : "h-full bg-emerald-500"}
                        style={{ width: `${Math.max(4, prediction.score * 100)}%` }}
                      />
                    </div>
                  </article>
                ))}
              </div>
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="How to use it"
        description="This is the quickest way to sanity-check the exported model with photos outside the curated dataset."
      >
        <div className="grid gap-4 md:grid-cols-3">
          {[
            "Take a few real trash photos from the angles and lighting you expect in deployment.",
            "Watch for confident mistakes on `glass -> paper` or `glass -> folk`, since that is still the main confusion pattern.",
            "If the outputs look stable here, the exported ONNX model is the right artifact to wire into downstream apps.",
          ].map((copy) => (
            <article
              key={copy}
              className={[
                "rounded-[1.35rem] border p-4 text-sm leading-7",
                isDark ? "border-zinc-800 bg-zinc-950 text-zinc-300" : "border-zinc-200 bg-zinc-50 text-zinc-700",
              ].join(" ")}
            >
              <div className="inline-flex items-center gap-2 font-semibold">
                <HiArrowUpRight />
                Practical note
              </div>
              <p className="mt-3">{copy}</p>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error("Failed to read image as base64."));
    };
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read image."));
    reader.readAsDataURL(file);
  });
}
