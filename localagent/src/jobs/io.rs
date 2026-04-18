use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use tokio::{
    fs::OpenOptions,
    io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader},
};

use super::{JobLogStream, JobManager, JobStreamEvent};

pub(super) async fn stream_child_output<R>(
    manager: JobManager,
    job_id: String,
    reader: R,
    log_path: PathBuf,
    stream: JobLogStream,
) -> Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut lines = BufReader::new(reader).lines();
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .await
        .with_context(|| format!("failed to open log file for append: {}", log_path.display()))?;

    while let Some(line) = lines.next_line().await? {
        log_file.write_all(line.as_bytes()).await?;
        log_file.write_all(b"\n").await?;
        manager.emit_event(JobStreamEvent::LogLine {
            job_id: job_id.clone(),
            stream,
            line,
        });
    }

    log_file.flush().await?;
    Ok(())
}

pub(super) fn read_tail_lines(path: &Path, max_lines: usize) -> Result<Vec<String>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read log file: {}", path.display()))?;
    let mut lines = content.lines().map(str::to_string).collect::<Vec<_>>();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    Ok(lines)
}

pub(super) fn timestamp_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| OffsetDateTime::now_utc().unix_timestamp().to_string())
}

pub(super) fn sequence_from_job_id(job_id: &str) -> u64 {
    job_id
        .rsplit('-')
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}
