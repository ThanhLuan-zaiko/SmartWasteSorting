use std::{
    collections::{BTreeMap, HashMap},
    sync::{atomic::AtomicU64, Arc, Mutex},
};

use tokio::{
    process::Child,
    sync::{broadcast, Mutex as AsyncMutex},
};

use crate::RuntimeConfig;

mod commands;
mod io;
mod runtime;
#[cfg(test)]
mod tests;
mod types;

pub use types::{
    BenchmarkJobRequest, JobLogStream, JobLogsSnapshot, JobRecord, JobStatus, JobStreamEvent,
    PipelineCommand, PipelineJobRequest, TrainingCommand, TrainingJobRequest,
};

const JOB_EVENT_BUFFER: usize = 2_048;

#[derive(Clone)]
pub struct JobManager {
    config: RuntimeConfig,
    jobs: Arc<Mutex<BTreeMap<String, types::JobRecord>>>,
    children: Arc<Mutex<HashMap<String, Arc<AsyncMutex<Child>>>>>,
    events: broadcast::Sender<types::JobStreamEvent>,
    sequence: Arc<AtomicU64>,
}
