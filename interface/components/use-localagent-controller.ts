"use client";

import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import type { LocalAgentContextValue } from "@/components/localagent-context";
import {
  buildJobsWebSocketUrl,
  type ClusterReviewResponse,
  type ClusterReviewSaveRequest,
  type CompareResponse,
  fetchJson,
  isActiveJobStatus,
  toNumberString,
  type JobLogStream,
  type JobLogsResponse,
  type JobRecord,
  type JobsResponse,
  type JobStreamEvent,
  type PipelineCatalogResponse,
  type PipelineFormState,
  type RunDetailResponse,
  type RunIndexEntry,
  type RunIndexResponse,
  type TrainingFormState,
  type TrainingPreset,
  type TrainingPresetsResponse,
} from "@/lib/localagent";

const LIVE_LOG_TAIL_LINES = 220;

function sortJobsByRecency(jobs: JobRecord[]): JobRecord[] {
  return [...jobs].sort((left, right) => right.created_at.localeCompare(left.created_at));
}

function upsertJobRecord(current: JobRecord[], job: JobRecord): JobRecord[] {
  const withoutTarget = current.filter((entry) => entry.job_id !== job.job_id);
  return sortJobsByRecency([job, ...withoutTarget]);
}

function trimLogTail(lines: string[], nextLine: string): string[] {
  const next = [...lines, nextLine];
  return next.length > LIVE_LOG_TAIL_LINES
    ? next.slice(next.length - LIVE_LOG_TAIL_LINES)
    : next;
}

function appendLogLine(
  current: JobLogsResponse | null,
  jobId: string,
  stream: JobLogStream,
  line: string,
): JobLogsResponse {
  const base =
    current && current.job_id === jobId
      ? current
      : {
          job_id: jobId,
          status: "running" as const,
          stdout: [],
          stderr: [],
        };

  return {
    ...base,
    stdout: stream === "stdout" ? trimLogTail(base.stdout, line) : base.stdout,
    stderr: stream === "stderr" ? trimLogTail(base.stderr, line) : base.stderr,
  };
}

function parseJobStreamEvent(value: string): JobStreamEvent | null {
  try {
    return JSON.parse(value) as JobStreamEvent;
  } catch {
    return null;
  }
}

function isClusterReviewUnavailableError(message: string): boolean {
  return (
    message.includes("Run `cluster` first") ||
    message.includes("Run `run-all` first") ||
    message.includes("No cluster assignments are available")
  );
}

export function useLocalAgentController(): LocalAgentContextValue {
  const [runs, setRuns] = useState<RunIndexEntry[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const [selectedExperiment, setSelectedExperimentState] = useState("");
  const [compareExperiment, setCompareExperimentState] = useState("");
  const [runDetail, setRunDetail] = useState<RunDetailResponse | null>(null);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeLogs, setActiveLogs] = useState<JobLogsResponse | null>(null);
  const [clusterReview, setClusterReview] = useState<ClusterReviewResponse | null>(null);
  const [clusterReviewError, setClusterReviewError] = useState<string | null>(null);
  const [isClusterReviewLoading, setIsClusterReviewLoading] = useState(false);
  const [isClusterReviewSaving, setIsClusterReviewSaving] = useState(false);
  const [trainingPresets, setTrainingPresets] = useState<Record<string, TrainingPreset>>({});
  const [pipelineCatalog, setPipelineCatalog] = useState<PipelineCatalogResponse>({
    dataset_commands: [],
    training_commands: [],
  });
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<string | null>(null);
  const hadRunningJobs = useRef(false);
  const jobsRef = useRef<JobRecord[]>([]);

  const [trainingForm, setTrainingForm] = useState<TrainingFormState>({
    experiment_name: "baseline-waste-sorter",
    training_preset: "cpu_balanced",
    training_backend: "pytorch",
    model_name: "resnet18",
    image_size: "224",
    batch_size: "16",
    epochs: "25",
    class_bias: "loss",
    device: "auto",
    compare_experiment: "",
    pseudo_label_threshold: "0.85",
    pseudo_label_margin: "0.15",
    no_progress: true,
  });

  const [pipelineForm, setPipelineForm] = useState<PipelineFormState>({
    labels_file: "artifacts/manifests/labeling_template.csv",
    review_file: "artifacts/manifests/cluster_review.csv",
    template_output: "artifacts/manifests/labeling_template.csv",
    review_output: "artifacts/manifests/cluster_review.csv",
    num_clusters: "",
    no_progress: true,
  });

  async function loadCatalog() {
    const [trainingPayload, pipelinePayload] = await Promise.all([
      fetchJson<TrainingPresetsResponse>("/presets/training"),
      fetchJson<PipelineCatalogResponse>("/presets/pipeline"),
    ]);
    setTrainingPresets(trainingPayload.presets);
    setPipelineCatalog(pipelinePayload);
  }

  async function loadRuns() {
    const payload = await fetchJson<RunIndexResponse>("/runs");
    setRuns(payload.runs);

    if (!selectedExperiment && payload.runs.length > 0) {
      const initialExperiment = payload.runs[0].experiment_name;
      setSelectedExperimentState(initialExperiment);
      setTrainingForm((current) => ({
        ...current,
        experiment_name: initialExperiment,
      }));
    }

    if (payload.runs.length > 1) {
      const candidate = payload.runs.find(
        (run) => run.experiment_name !== selectedExperiment,
      );
      if (candidate && (!compareExperiment || compareExperiment === selectedExperiment)) {
        setCompareExperimentState(candidate.experiment_name);
      }
    }

    return payload;
  }

  async function loadRunDetail(experimentName: string) {
    if (!experimentName) {
      setRunDetail(null);
      return;
    }

    const payload = await fetchJson<RunDetailResponse>(
      `/runs/${encodeURIComponent(experimentName)}`,
    );
    startTransition(() => {
      setRunDetail(payload);
    });
  }

  async function loadComparison(left: string, right: string) {
    if (!left || !right || left === right) {
      setComparison(null);
      return;
    }

    try {
      const payload = await fetchJson<CompareResponse>(
        `/runs/${encodeURIComponent(left)}/compare?with=${encodeURIComponent(right)}`,
      );
      setComparison(payload);
    } catch {
      setComparison(null);
    }
  }

  async function loadClusterReview(reviewFile = pipelineForm.review_file) {
    setIsClusterReviewLoading(true);
    try {
      const query = reviewFile ? `?review_file=${encodeURIComponent(reviewFile)}` : "";
      const payload = await fetchJson<ClusterReviewResponse>(`/cluster-review${query}`);
      setClusterReview(payload);
      setClusterReviewError(null);
    } catch (error) {
      const nextMessage =
        error instanceof Error ? error.message : "Failed to load cluster review.";
      setClusterReview(null);
      setClusterReviewError(nextMessage);
      if (!isClusterReviewUnavailableError(nextMessage)) {
        throw error;
      }
    } finally {
      setIsClusterReviewLoading(false);
    }
  }

  async function loadJobs() {
    const payload = await fetchJson<JobsResponse>("/jobs");
    const nextJobs = sortJobsByRecency(payload.jobs);
    jobsRef.current = nextJobs;
    setJobs(nextJobs);

    const runningNow = payload.jobs.some((job) => isActiveJobStatus(job.status));
    const targetJobId = activeJobId ?? payload.jobs[0]?.job_id ?? null;
    if (!activeJobId && targetJobId) {
      setActiveJobId(targetJobId);
    }
    if (targetJobId) {
      try {
        const logs = await fetchJson<JobLogsResponse>(
          `/jobs/${encodeURIComponent(targetJobId)}/logs?tail_lines=${LIVE_LOG_TAIL_LINES}`,
        );
        setActiveLogs(logs);
      } catch {
        setActiveLogs(null);
      }
    }

    if (hadRunningJobs.current && !runningNow) {
      await loadRuns();
      if (selectedExperiment) {
        await loadRunDetail(selectedExperiment);
      }
      if (selectedExperiment && compareExperiment) {
        await loadComparison(selectedExperiment, compareExperiment);
      }
      await loadClusterReview();
    }

    hadRunningJobs.current = runningNow;
    return payload;
  }

  async function refreshAll() {
    try {
      await Promise.all([loadCatalog(), loadRuns(), loadJobs()]);
      if (selectedExperiment) {
        await loadRunDetail(selectedExperiment);
      }
      if (selectedExperiment && compareExperiment) {
        await loadComparison(selectedExperiment, compareExperiment);
      }
      await loadClusterReview();
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Unable to reach localagent server.",
      );
    }
  }

  async function ensureRunLoaded(experimentName: string) {
    setSelectedExperimentState(experimentName);
    setTrainingForm((current) => ({
      ...current,
      experiment_name: experimentName,
    }));
    await loadRunDetail(experimentName);
  }

  async function reloadClusterReview(reviewFile = pipelineForm.review_file) {
    try {
      await loadClusterReview(reviewFile);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Failed to load cluster review.",
      );
    }
  }

  async function saveClusterReview(payload: ClusterReviewSaveRequest) {
    setIsClusterReviewSaving(true);
    setMessage(null);
    try {
      const saved = await fetchJson<ClusterReviewResponse>("/cluster-review", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setClusterReview(saved);
      setClusterReviewError(null);
      setMessage(`Saved cluster review draft to ${saved.review_file}`);
      setConnectionError(null);
      return saved;
    } catch (error) {
      const nextMessage =
        error instanceof Error ? error.message : "Failed to save cluster review.";
      setClusterReviewError(nextMessage);
      return null;
    } finally {
      setIsClusterReviewSaving(false);
    }
  }

  async function submitPipeline(command: string) {
    setIsSubmitting(command);
    setMessage(null);
    try {
      const payload: Record<string, unknown> = {
        command,
        no_progress: pipelineForm.no_progress,
      };
      if (pipelineForm.labels_file) {
        payload.labels_file = pipelineForm.labels_file;
      }
      if (pipelineForm.review_file) {
        payload.review_file = pipelineForm.review_file;
      }
      if (command === "export-labeling-template" && pipelineForm.template_output) {
        payload.output = pipelineForm.template_output;
      }
      if (command === "export-cluster-review" && pipelineForm.review_output) {
        payload.output = pipelineForm.review_output;
      }
      if (pipelineForm.num_clusters.trim()) {
        payload.num_clusters = toNumberString(pipelineForm.num_clusters);
      }

      const created = await fetchJson<JobRecord>("/jobs/pipeline", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setActiveJobId(created.job_id);
      setMessage(`Started dataset job ${created.job_id}`);
      const jobsPayload = await fetchJson<JobsResponse>("/jobs");
      const nextJobs = sortJobsByRecency(jobsPayload.jobs);
      jobsRef.current = nextJobs;
      setJobs(nextJobs);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Failed to create pipeline job.",
      );
    } finally {
      setIsSubmitting(null);
    }
  }

  async function submitTraining(command: string) {
    setIsSubmitting(command);
    setMessage(null);
    try {
      const basePayload: Record<string, unknown> = {
        experiment_name: trainingForm.experiment_name,
        training_preset: trainingForm.training_preset || undefined,
        training_backend: trainingForm.training_backend,
        model_name: trainingForm.model_name,
        image_size: toNumberString(trainingForm.image_size),
        batch_size: toNumberString(trainingForm.batch_size),
        epochs: toNumberString(trainingForm.epochs),
        class_bias: trainingForm.class_bias,
        device: trainingForm.device,
        pseudo_label_threshold: toNumberString(trainingForm.pseudo_label_threshold),
        pseudo_label_margin: toNumberString(trainingForm.pseudo_label_margin),
        no_progress: trainingForm.no_progress,
      };

      const endpoint = command === "benchmark" ? "/jobs/benchmark" : "/jobs/training";
      const payload =
        command === "benchmark"
          ? {
              ...basePayload,
              compare_experiment: trainingForm.compare_experiment || undefined,
            }
          : {
              ...basePayload,
              command,
            };

      const created = await fetchJson<JobRecord>(endpoint, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setActiveJobId(created.job_id);
      setSelectedExperimentState(trainingForm.experiment_name);
      setMessage(`Started ${command} job ${created.job_id}`);
      const jobsPayload = await fetchJson<JobsResponse>("/jobs");
      const nextJobs = sortJobsByRecency(jobsPayload.jobs);
      jobsRef.current = nextJobs;
      setJobs(nextJobs);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Failed to create training job.",
      );
    } finally {
      setIsSubmitting(null);
    }
  }

  async function cancelActiveJob() {
    if (!activeJobId) {
      return;
    }
    try {
      const cancelled = await fetchJson<JobRecord>(
        `/jobs/${encodeURIComponent(activeJobId)}/cancel`,
        { method: "POST" },
      );
      setMessage(`Cancelled ${cancelled.job_id}`);
      const jobsPayload = await fetchJson<JobsResponse>("/jobs");
      const nextJobs = sortJobsByRecency(jobsPayload.jobs);
      jobsRef.current = nextJobs;
      setJobs(nextJobs);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Failed to cancel job.",
      );
    }
  }

  function applyPreset(presetName: string) {
    const preset = trainingPresets[presetName];
    setTrainingForm((current) => ({
      ...current,
      training_preset: presetName,
      model_name: preset?.model_name ?? current.model_name,
      image_size: preset?.image_size ? `${preset.image_size}` : current.image_size,
      batch_size: preset?.batch_size ? `${preset.batch_size}` : current.batch_size,
      class_bias: preset?.class_bias ?? current.class_bias,
    }));
  }

  function setSelectedExperiment(value: string) {
    setSelectedExperimentState(value);
    setTrainingForm((current) => ({
      ...current,
      experiment_name: value,
    }));
  }

  function setTrainingField<K extends keyof TrainingFormState>(
    field: K,
    value: TrainingFormState[K],
  ) {
    setTrainingForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function setPipelineField<K extends keyof PipelineFormState>(
    field: K,
    value: PipelineFormState[K],
  ) {
    setPipelineForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  const handleJobStreamEvent = useEffectEvent(async (event: JobStreamEvent) => {
    if (event.event === "snapshot") {
      const nextJobs = sortJobsByRecency(event.jobs);
      jobsRef.current = nextJobs;
      setJobs(nextJobs);
      setActiveLogs(event.active_logs ?? null);
      hadRunningJobs.current = nextJobs.some((job) => isActiveJobStatus(job.status));
      return;
    }

    if (event.event === "job_updated") {
      const nextJobs = upsertJobRecord(jobsRef.current, event.job);
      const runningNow = nextJobs.some((job) => isActiveJobStatus(job.status));
      jobsRef.current = nextJobs;
      setJobs(nextJobs);

      if (event.job.job_id === activeJobId && !isActiveJobStatus(event.job.status)) {
        try {
          const logs = await fetchJson<JobLogsResponse>(
            `/jobs/${encodeURIComponent(event.job.job_id)}/logs?tail_lines=${LIVE_LOG_TAIL_LINES}`,
          );
          setActiveLogs(logs);
        } catch {
          setActiveLogs((current) =>
            current && current.job_id === event.job.job_id
              ? { ...current, status: event.job.status }
              : current,
          );
        }
      } else if (event.job.job_id === activeJobId) {
        setActiveLogs((current) =>
          current && current.job_id === event.job.job_id
            ? { ...current, status: event.job.status }
            : current,
        );
      }

      if (hadRunningJobs.current && !runningNow) {
        await loadRuns();
        if (selectedExperiment) {
          await loadRunDetail(selectedExperiment);
        }
        if (selectedExperiment && compareExperiment) {
          await loadComparison(selectedExperiment, compareExperiment);
        }
      }
      hadRunningJobs.current = runningNow;
      return;
    }

    if (event.event === "log_line") {
      if (event.job_id !== activeJobId) {
        return;
      }
      setActiveLogs((current) =>
        appendLogLine(current, event.job_id, event.stream, event.line),
      );
      return;
    }

    if (event.event === "resync_required") {
      try {
        const jobsPayload = await fetchJson<JobsResponse>("/jobs");
        const nextJobs = sortJobsByRecency(jobsPayload.jobs);
        jobsRef.current = nextJobs;
        setJobs(nextJobs);
        if (activeJobId) {
          const logs = await fetchJson<JobLogsResponse>(
            `/jobs/${encodeURIComponent(activeJobId)}/logs?tail_lines=${LIVE_LOG_TAIL_LINES}`,
          );
          setActiveLogs(logs);
        }
      } catch (error) {
        setConnectionError(error instanceof Error ? error.message : event.reason);
      }
    }
  });

  const refreshAllEffect = useEffectEvent(async () => {
    await refreshAll();
  });

  const loadJobsEffect = useEffectEvent(async () => {
    await loadJobs();
  });

  useEffect(() => {
    void refreshAllEffect();
  }, []);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  useEffect(() => {
    if (!selectedExperiment) {
      return;
    }
    void loadRunDetail(selectedExperiment);
  }, [selectedExperiment]);

  useEffect(() => {
    if (!selectedExperiment || !compareExperiment) {
      setComparison(null);
      return;
    }
    void loadComparison(selectedExperiment, compareExperiment);
  }, [compareExperiment, selectedExperiment]);

  useEffect(() => {
    setActiveLogs((current) => (current && current.job_id === activeJobId ? current : null));

    const socket = new WebSocket(buildJobsWebSocketUrl(activeJobId, LIVE_LOG_TAIL_LINES));

    socket.onopen = () => {
      setStreamConnected(true);
      setConnectionError(null);
    };

    socket.onmessage = (message) => {
      if (typeof message.data !== "string") {
        return;
      }
      const event = parseJobStreamEvent(message.data);
      if (!event) {
        return;
      }
      void handleJobStreamEvent(event);
    };

    socket.onerror = () => {
      setStreamConnected(false);
    };

    socket.onclose = () => {
      setStreamConnected(false);
    };

    return () => {
      setStreamConnected(false);
      socket.close();
    };
  }, [activeJobId]);

  useEffect(() => {
    if (streamConnected) {
      return;
    }
    const interval = window.setInterval(() => {
      void loadJobsEffect();
    }, jobs.some((job) => isActiveJobStatus(job.status)) ? 1500 : 5000);
    return () => window.clearInterval(interval);
  }, [streamConnected, activeJobId, jobs, selectedExperiment, compareExperiment]);

  return {
    runs,
    jobs,
    streamConnected,
    selectedExperiment,
    compareExperiment,
    runDetail,
    comparison,
    activeJobId,
    activeLogs,
    clusterReview,
    clusterReviewError,
    isClusterReviewLoading,
    isClusterReviewSaving,
    trainingPresets,
    pipelineCatalog,
    connectionError,
    message,
    isSubmitting,
    trainingForm,
    pipelineForm,
    setSelectedExperiment,
    setCompareExperiment: setCompareExperimentState,
    setActiveJobId,
    setTrainingField,
    setPipelineField,
    applyPreset,
    refreshAll,
    ensureRunLoaded,
    reloadClusterReview,
    saveClusterReview,
    submitPipeline,
    submitTraining,
    cancelActiveJob,
  };
}
