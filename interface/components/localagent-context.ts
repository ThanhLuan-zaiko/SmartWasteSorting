"use client";

import { createContext, useContext } from "react";

import {
  type ClusterReviewResponse,
  type ClusterReviewSaveRequest,
  type CompareResponse,
  type JobLogsResponse,
  type JobRecord,
  type PipelineCatalogResponse,
  type PipelineFormState,
  type RunDetailResponse,
  type RunIndexEntry,
  type TrainingFormState,
  type TrainingPreset,
  type WorkflowStateResponse,
} from "@/lib/localagent";

export type LocalAgentContextValue = {
  runs: RunIndexEntry[];
  jobs: JobRecord[];
  streamConnected: boolean;
  selectedExperiment: string;
  compareExperiment: string;
  runDetail: RunDetailResponse | null;
  comparison: CompareResponse | null;
  workflowState: WorkflowStateResponse | null;
  activeJobId: string | null;
  activeLogs: JobLogsResponse | null;
  clusterReview: ClusterReviewResponse | null;
  clusterReviewError: string | null;
  isClusterReviewLoading: boolean;
  isClusterReviewSaving: boolean;
  trainingPresets: Record<string, TrainingPreset>;
  pipelineCatalog: PipelineCatalogResponse;
  connectionError: string | null;
  message: string | null;
  isSubmitting: string | null;
  trainingForm: TrainingFormState;
  pipelineForm: PipelineFormState;
  setSelectedExperiment: (value: string) => void;
  setCompareExperiment: (value: string) => void;
  setActiveJobId: (jobId: string | null) => void;
  setTrainingField: <K extends keyof TrainingFormState>(
    field: K,
    value: TrainingFormState[K],
  ) => void;
  setPipelineField: <K extends keyof PipelineFormState>(
    field: K,
    value: PipelineFormState[K],
  ) => void;
  applyPreset: (presetName: string) => void;
  refreshAll: () => Promise<void>;
  ensureRunLoaded: (experimentName: string) => Promise<void>;
  reloadClusterReview: (reviewFile?: string) => Promise<void>;
  saveClusterReview: (payload: ClusterReviewSaveRequest) => Promise<ClusterReviewResponse | null>;
  submitPipeline: (command: string) => Promise<void>;
  submitTraining: (command: string) => Promise<void>;
  cancelActiveJob: () => Promise<void>;
};

export const LocalAgentContext = createContext<LocalAgentContextValue | null>(null);

export function useLocalAgent() {
  const context = useContext(LocalAgentContext);
  if (!context) {
    throw new Error("useLocalAgent must be used inside LocalAgentProvider");
  }
  return context;
}
