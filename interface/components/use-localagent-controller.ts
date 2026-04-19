"use client";

import { useEffect, useEffectEvent, useRef, useState } from "react";

import {
  appendLogLine,
  LIVE_LOG_TAIL_LINES,
  parseJobStreamEvent,
  sortJobsByRecency,
  upsertJobRecord,
} from "@/components/localagent/controller-helpers";
import { useLocalAgentActions } from "@/components/localagent/controller-actions";
import type { LocalAgentContextValue } from "@/components/localagent-context";
import {
  buildJobsWebSocketUrl,
  fetchJson,
  isActiveJobStatus,
  type ClusterReviewResponse,
  type CompareResponse,
  type JobLogsResponse,
  type JobRecord,
  type JobsResponse,
  type JobStreamEvent,
  type PipelineCatalogResponse,
  type PipelineFormState,
  type RunDetailResponse,
  type RunIndexEntry,
  type TrainingFormState,
  type TrainingPreset,
  type WorkflowStateResponse,
} from "@/lib/localagent";

export function useLocalAgentController(): LocalAgentContextValue {
  const [runs, setRuns] = useState<RunIndexEntry[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const [selectedExperiment, setSelectedExperimentState] = useState("");
  const [compareExperiment, setCompareExperimentState] = useState("");
  const [runDetail, setRunDetail] = useState<RunDetailResponse | null>(null);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [workflowState, setWorkflowState] = useState<WorkflowStateResponse | null>(null);
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

  const actions = useLocalAgentActions({
    activeJobId,
    compareExperiment,
    hadRunningJobs,
    jobsRef,
    pipelineForm,
    selectedExperiment,
    setActiveJobId,
    setActiveLogs,
    setClusterReview,
    setClusterReviewError,
    setComparison,
    setConnectionError,
    setIsClusterReviewLoading,
    setIsClusterReviewSaving,
    setIsSubmitting,
    setJobs,
    setMessage,
    setPipelineCatalog,
    setPipelineForm,
    setRunDetail,
    setRuns,
    setSelectedExperimentState,
    setCompareExperimentState,
    setTrainingForm,
    setTrainingPresets,
    setWorkflowState,
    trainingForm,
    trainingPresets,
  });

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
        await actions.loadRuns();
        if (selectedExperiment) {
          await actions.loadRunDetail(selectedExperiment);
        }
        if (selectedExperiment && compareExperiment) {
          await actions.loadComparison(selectedExperiment, compareExperiment);
        }
        await actions.loadClusterReview();
        await actions.loadWorkflowState();
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
    await actions.refreshAll();
  });

  const loadJobsEffect = useEffectEvent(async () => {
    await actions.loadJobs();
  });

  const loadWorkflowStateEffect = useEffectEvent(async (reviewFile: string) => {
    try {
      await actions.loadWorkflowState(reviewFile);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Failed to load workflow state.",
      );
    }
  });

  const loadRunDetailEffect = useEffectEvent(async (experimentName: string) => {
    await actions.loadRunDetail(experimentName);
  });

  const loadComparisonEffect = useEffectEvent(async (left: string, right: string) => {
    await actions.loadComparison(left, right);
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
    void loadRunDetailEffect(selectedExperiment);
  }, [selectedExperiment]);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void loadWorkflowStateEffect(pipelineForm.review_file);
    }, 250);
    return () => window.clearTimeout(timeoutId);
  }, [pipelineForm.review_file]);

  useEffect(() => {
    void loadComparisonEffect(selectedExperiment, compareExperiment);
  }, [compareExperiment, selectedExperiment]);

  useEffect(() => {
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

  const visibleActiveLogs =
    activeLogs && activeLogs.job_id === activeJobId ? activeLogs : null;

  return {
    runs,
    jobs,
    streamConnected,
    selectedExperiment,
    compareExperiment,
    runDetail,
    comparison,
    workflowState,
    activeJobId,
    activeLogs: visibleActiveLogs,
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
    setSelectedExperiment: actions.setSelectedExperiment,
    setCompareExperiment: setCompareExperimentState,
    setActiveJobId,
    setTrainingField: actions.setTrainingField,
    setPipelineField: actions.setPipelineField,
    applyPreset: actions.applyPreset,
    refreshAll: actions.refreshAll,
    ensureRunLoaded: actions.ensureRunLoaded,
    reloadClusterReview: actions.reloadClusterReview,
    saveClusterReview: actions.saveClusterReview,
    submitPipeline: actions.submitPipeline,
    submitTraining: actions.submitTraining,
    cancelActiveJob: actions.cancelActiveJob,
  };
}
