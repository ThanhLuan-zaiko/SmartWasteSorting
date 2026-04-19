"use client";

import { startTransition, type Dispatch, type MutableRefObject, type SetStateAction } from "react";

import type {
  ClusterReviewResponse,
  ClusterReviewSaveRequest,
  CompareResponse,
  JobLogsResponse,
  JobRecord,
  JobsResponse,
  PipelineCatalogResponse,
  PipelineFormState,
  RunDetailResponse,
  RunIndexEntry,
  RunIndexResponse,
  TrainingFormState,
  TrainingPreset,
  TrainingPresetsResponse,
  WorkflowStateResponse,
} from "@/lib/localagent";
import { fetchJson, isActiveJobStatus, toNumberString } from "@/lib/localagent";

import {
  isClusterReviewUnavailableError,
  LIVE_LOG_TAIL_LINES,
  sortJobsByRecency,
} from "@/components/localagent/controller-helpers";

type SetState<T> = Dispatch<SetStateAction<T>>;

export type LocalAgentControllerActionDeps = {
  activeJobId: string | null;
  compareExperiment: string;
  hadRunningJobs: MutableRefObject<boolean>;
  jobsRef: MutableRefObject<JobRecord[]>;
  pipelineForm: PipelineFormState;
  selectedExperiment: string;
  setActiveJobId: SetState<string | null>;
  setActiveLogs: SetState<JobLogsResponse | null>;
  setClusterReview: SetState<ClusterReviewResponse | null>;
  setClusterReviewError: SetState<string | null>;
  setComparison: SetState<CompareResponse | null>;
  setConnectionError: SetState<string | null>;
  setIsClusterReviewLoading: SetState<boolean>;
  setIsClusterReviewSaving: SetState<boolean>;
  setIsSubmitting: SetState<string | null>;
  setJobs: SetState<JobRecord[]>;
  setMessage: SetState<string | null>;
  setPipelineCatalog: SetState<PipelineCatalogResponse>;
  setPipelineForm: SetState<PipelineFormState>;
  setRunDetail: SetState<RunDetailResponse | null>;
  setRuns: SetState<RunIndexEntry[]>;
  setSelectedExperimentState: SetState<string>;
  setCompareExperimentState: SetState<string>;
  setTrainingForm: SetState<TrainingFormState>;
  setTrainingPresets: SetState<Record<string, TrainingPreset>>;
  setWorkflowState: SetState<WorkflowStateResponse | null>;
  trainingForm: TrainingFormState;
  trainingPresets: Record<string, TrainingPreset>;
};

function applyJobsSnapshot(
  nextJobs: JobRecord[],
  jobsRef: MutableRefObject<JobRecord[]>,
  setJobs: SetState<JobRecord[]>,
) {
  jobsRef.current = nextJobs;
  setJobs(nextJobs);
}

export function useLocalAgentActions({
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
}: LocalAgentControllerActionDeps) {
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

  async function loadWorkflowState(reviewFile = pipelineForm.review_file) {
    const query = reviewFile ? `?review_file=${encodeURIComponent(reviewFile)}` : "";
    const payload = await fetchJson<WorkflowStateResponse>(`/workflow/state${query}`);
    setWorkflowState(payload);
  }

  async function loadJobs() {
    const payload = await fetchJson<JobsResponse>("/jobs");
    const nextJobs = sortJobsByRecency(payload.jobs);
    applyJobsSnapshot(nextJobs, jobsRef, setJobs);

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
      await loadWorkflowState();
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
      await Promise.all([loadClusterReview(), loadWorkflowState()]);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(
        error instanceof Error ? error.message : "Unable to reach localagent server.",
      );
    }
  }

  async function ensureRunLoaded(experimentName: string) {
    setSelectedExperimentState((current) =>
      current === experimentName ? current : experimentName,
    );
    setTrainingForm((current) =>
      current.experiment_name === experimentName
        ? current
        : {
            ...current,
            experiment_name: experimentName,
          },
    );
    await loadRunDetail(experimentName);
  }

  async function reloadClusterReview(reviewFile = pipelineForm.review_file) {
    try {
      await Promise.all([loadClusterReview(reviewFile), loadWorkflowState(reviewFile)]);
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
      await loadWorkflowState(saved.review_file);
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
      applyJobsSnapshot(nextJobs, jobsRef, setJobs);
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
      applyJobsSnapshot(nextJobs, jobsRef, setJobs);
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
      applyJobsSnapshot(nextJobs, jobsRef, setJobs);
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
    setSelectedExperimentState((current) => (current === value ? current : value));
    setTrainingForm((current) =>
      current.experiment_name === value
        ? current
        : {
            ...current,
            experiment_name: value,
          },
    );
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

  return {
    applyPreset,
    cancelActiveJob,
    ensureRunLoaded,
    loadClusterReview,
    loadComparison,
    loadJobs,
    loadRunDetail,
    loadRuns,
    loadWorkflowState,
    refreshAll,
    reloadClusterReview,
    saveClusterReview,
    setPipelineField,
    setSelectedExperiment,
    setTrainingField,
    submitPipeline,
    submitTraining,
  };
}
