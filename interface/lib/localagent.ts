export const API_PREFIX = "/api/localagent";
const DEFAULT_LOCALAGENT_BASE = "http://127.0.0.1:8080";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type JsonObject = Record<string, unknown>;

export type JobRecord = {
  job_id: string;
  job_type: string;
  command: string;
  experiment_name?: string | null;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress_hint?: string | null;
  stdout_log_path: string;
  stderr_log_path: string;
  exit_code?: number | null;
  error?: string | null;
};

export type JobsResponse = {
  jobs: JobRecord[];
};

export type RunIndexEntry = {
  experiment_name: string;
  latest_generated_at?: string;
  training_backend?: string | null;
  available?: JsonObject;
  cards?: JsonObject;
};

export type RunIndexResponse = {
  runs: RunIndexEntry[];
};

export type RunDetailResponse = {
  experiment_name: string;
  dashboard_summary?: JsonObject;
  overview?: JsonObject;
  training?: JsonObject | null;
  evaluation?: JsonObject | null;
  export?: JsonObject | null;
  benchmark?: JsonObject | null;
  experiment_spec?: JsonObject | null;
  model_manifest?: JsonObject | null;
  job_history?: JobRecord[];
};

export type JobLogsResponse = {
  job_id: string;
  status: JobStatus;
  stdout: string[];
  stderr: string[];
  stdout_log_path?: string;
  stderr_log_path?: string;
};

export type JobLogStream = "stdout" | "stderr";

export type JobStreamEvent =
  | {
      event: "snapshot";
      jobs: JobRecord[];
      active_logs?: JobLogsResponse | null;
    }
  | {
      event: "job_updated";
      job: JobRecord;
    }
  | {
      event: "log_line";
      job_id: string;
      stream: JobLogStream;
      line: string;
    }
  | {
      event: "resync_required";
      reason: string;
    };

export type TrainingPreset = {
  model_name?: string;
  image_size?: number;
  batch_size?: number;
  cache_format?: string;
  class_bias?: string;
};

export type TrainingPresetsResponse = {
  presets: Record<string, TrainingPreset>;
};

export type PipelineCatalogResponse = {
  dataset_commands: string[];
  training_commands: string[];
};

export type CompareResponse = {
  duration_delta_seconds?: number | null;
  fit_stage_delta_seconds?: number | null;
  accuracy_delta?: number | null;
  macro_f1_delta?: number | null;
  weighted_f1_delta?: number | null;
  left_backend?: string | null;
  right_backend?: string | null;
};

export type TrainingFormState = {
  experiment_name: string;
  training_preset: string;
  training_backend: string;
  model_name: string;
  image_size: string;
  batch_size: string;
  epochs: string;
  class_bias: string;
  device: string;
  compare_experiment: string;
};

export type PipelineFormState = {
  labels_file: string;
  output: string;
};

export const DATASET_ACTIONS = [
  "run-all",
  "export-labeling-template",
  "import-labels",
  "validate-labels",
] as const;

export const TRAINING_ACTIONS = [
  "summary",
  "export-spec",
  "warm-cache",
  "fit",
  "evaluate",
  "export-onnx",
  "report",
] as const;

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const payload = (await response.json()) as JsonObject;
      const error = payload.error;
      if (typeof error === "string" && error.length > 0) {
        message = error;
      }
    } catch {
      const text = await response.text();
      if (text.trim().length > 0) {
        message = text;
      }
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function asObject(value: unknown): JsonObject | null {
  if (!value || Array.isArray(value) || typeof value !== "object") {
    return null;
  }
  return value as JsonObject;
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

export function toNumberString(value: string): number | undefined {
  if (!value.trim()) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function formatMetric(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? `${value}` : value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  return "N/A";
}

export function formatDelta(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "N/A";
  }
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(3)}`;
}

export function isActiveJobStatus(status: JobStatus | null | undefined): boolean {
  return status === "pending" || status === "running";
}

function toWebSocketBase(base: string): string {
  if (base.startsWith("https://")) {
    return `wss://${base.slice("https://".length)}`;
  }
  if (base.startsWith("http://")) {
    return `ws://${base.slice("http://".length)}`;
  }
  return base;
}

export function buildJobsWebSocketUrl(
  jobId?: string | null,
  tailLines = 220,
): string {
  const configuredBase =
    process.env.NEXT_PUBLIC_LOCALAGENT_WS_BASE ||
    process.env.NEXT_PUBLIC_LOCALAGENT_API_BASE ||
    DEFAULT_LOCALAGENT_BASE;
  const url = new URL(`${toWebSocketBase(configuredBase).replace(/\/$/, "")}/ws/jobs`);
  if (jobId) {
    url.searchParams.set("job_id", jobId);
  }
  if (tailLines > 0) {
    url.searchParams.set("tail_lines", `${tailLines}`);
  }
  return url.toString();
}
