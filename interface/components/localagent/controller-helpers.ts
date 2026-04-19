"use client";

import type {
  JobLogStream,
  JobLogsResponse,
  JobRecord,
  JobStreamEvent,
} from "@/lib/localagent";

export const LIVE_LOG_TAIL_LINES = 220;

export function sortJobsByRecency(jobs: JobRecord[]): JobRecord[] {
  return [...jobs].sort((left, right) => right.created_at.localeCompare(left.created_at));
}

export function upsertJobRecord(current: JobRecord[], job: JobRecord): JobRecord[] {
  const withoutTarget = current.filter((entry) => entry.job_id !== job.job_id);
  return sortJobsByRecency([job, ...withoutTarget]);
}

function trimLogTail(lines: string[], nextLine: string): string[] {
  const next = [...lines, nextLine];
  return next.length > LIVE_LOG_TAIL_LINES
    ? next.slice(next.length - LIVE_LOG_TAIL_LINES)
    : next;
}

export function appendLogLine(
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

export function parseJobStreamEvent(value: string): JobStreamEvent | null {
  try {
    return JSON.parse(value) as JobStreamEvent;
  } catch {
    return null;
  }
}

export function isClusterReviewUnavailableError(message: string): boolean {
  return (
    message.includes("Run `cluster` first") ||
    message.includes("Run `run-all` first") ||
    message.includes("No cluster assignments are available")
  );
}
