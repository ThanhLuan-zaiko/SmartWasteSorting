use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::RuntimeConfig;

const CLUSTER_REPRESENTATIVE_LIMIT: usize = 5;
const REVIEW_HEADER: [&str; 9] = [
    "cluster_id",
    "cluster_size",
    "outlier_count",
    "representative_sample_ids",
    "representative_paths",
    "current_majority_label",
    "label",
    "status",
    "notes",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterReviewRepresentative {
    pub sample_id: String,
    pub relative_path: String,
    pub label: String,
    pub label_source: String,
    pub annotation_status: String,
    pub review_status: String,
    pub cluster_distance: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterReviewCluster {
    pub cluster_id: i64,
    pub cluster_size: usize,
    pub outlier_count: usize,
    pub representative_sample_ids: String,
    pub representative_paths: String,
    pub current_majority_label: Option<String>,
    pub label: String,
    pub status: String,
    pub notes: String,
    pub representatives: Vec<ClusterReviewRepresentative>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterReviewState {
    pub review_file: String,
    pub cluster_count: usize,
    pub reviewed_count: usize,
    pub stale_reset_count: usize,
    pub clusters: Vec<ClusterReviewCluster>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ClusterReviewSaveRequest {
    pub review_file: Option<String>,
    pub clusters: Vec<ClusterReviewSaveCluster>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ClusterReviewSaveCluster {
    pub cluster_id: i64,
    pub cluster_size: usize,
    pub outlier_count: usize,
    pub representative_sample_ids: String,
    pub representative_paths: String,
    pub label: Option<String>,
    pub status: String,
    pub notes: Option<String>,
}

#[derive(Debug)]
pub enum ClusterReviewError {
    InvalidRequest(String),
    Conflict {
        message: String,
        stale_cluster_ids: Vec<i64>,
    },
    Internal(String),
}

impl ClusterReviewError {
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidRequest(message) | Self::Internal(message) => message,
            Self::Conflict { message, .. } => message,
        }
    }

    pub fn stale_cluster_ids(&self) -> &[i64] {
        match self {
            Self::Conflict {
                stale_cluster_ids, ..
            } => stale_cluster_ids,
            _ => &[],
        }
    }
}

impl fmt::Display for ClusterReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for ClusterReviewError {}

#[derive(Clone, Debug)]
pub struct ClusterReviewStore {
    config: RuntimeConfig,
}

impl ClusterReviewStore {
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config }
    }

    pub fn load_review_state(
        &self,
        review_file: Option<&str>,
    ) -> Result<ClusterReviewState, ClusterReviewError> {
        let review_path = self.resolve_review_path(review_file);
        load_review_state_from_paths(&self.manifest_csv_path(), &review_path)
    }

    pub fn save_review(
        &self,
        request: ClusterReviewSaveRequest,
    ) -> Result<ClusterReviewState, ClusterReviewError> {
        let mut state = self.load_review_state(request.review_file.as_deref())?;
        let review_path = PathBuf::from(&state.review_file);
        let by_cluster_id = state
            .clusters
            .iter()
            .enumerate()
            .map(|(index, cluster)| (cluster.cluster_id, index))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        let mut stale_cluster_ids = Vec::new();

        for cluster in request.clusters {
            if !seen.insert(cluster.cluster_id) {
                return Err(ClusterReviewError::InvalidRequest(format!(
                    "Duplicate cluster_id {} in save payload.",
                    cluster.cluster_id
                )));
            }

            let Some(current_index) = by_cluster_id.get(&cluster.cluster_id).copied() else {
                stale_cluster_ids.push(cluster.cluster_id);
                continue;
            };
            let current = &mut state.clusters[current_index];

            if !fingerprints_match(
                current,
                cluster.cluster_size,
                cluster.outlier_count,
                &cluster.representative_sample_ids,
            ) {
                stale_cluster_ids.push(cluster.cluster_id);
                continue;
            }

            let raw_label = cluster.label.unwrap_or_default();
            let status = normalize_status(&cluster.status, &raw_label)?;
            let normalized_label = normalize_label(&raw_label);
            if status == "labeled" && normalized_label.is_empty() {
                return Err(ClusterReviewError::InvalidRequest(format!(
                    "Cluster {} is marked labeled but has no usable label.",
                    cluster.cluster_id
                )));
            }

            current.status = status.to_string();
            current.label = if status == "labeled" {
                normalized_label
            } else {
                String::new()
            };
            current.notes = cluster.notes.unwrap_or_default().trim().to_string();
        }

        if !stale_cluster_ids.is_empty() {
            stale_cluster_ids.sort_unstable();
            stale_cluster_ids.dedup();
            return Err(ClusterReviewError::Conflict {
                message: format!(
                    "Cluster review payload is stale for clusters: {}. Reload review state and try again.",
                    stale_cluster_ids
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                stale_cluster_ids,
            });
        }

        self.write_review_rows(&review_path, &state.clusters)?;
        self.load_review_state(request.review_file.as_deref())
    }

    fn artifact_root(&self) -> PathBuf {
        let path = PathBuf::from(&self.config.artifact_dir);
        if path.is_absolute() {
            path
        } else {
            self.project_root().join(path)
        }
    }

    fn manifest_csv_path(&self) -> PathBuf {
        self.artifact_root()
            .join("manifests")
            .join("dataset_manifest.csv")
    }

    fn resolve_review_path(&self, review_file: Option<&str>) -> PathBuf {
        let Some(trimmed) = review_file.map(str::trim).filter(|value| !value.is_empty()) else {
            return self
                .artifact_root()
                .join("manifests")
                .join("cluster_review.csv");
        };
        let path = PathBuf::from(trimmed);
        if path.is_absolute() {
            path
        } else {
            self.project_root().join(path)
        }
    }

    fn project_root(&self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn write_review_rows(
        &self,
        review_path: &Path,
        clusters: &[ClusterReviewCluster],
    ) -> Result<(), ClusterReviewError> {
        if let Some(parent) = review_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                ClusterReviewError::Internal(format!(
                    "Failed to create review directory `{}`: {}",
                    parent.display(),
                    error
                ))
            })?;
        }

        let mut output = REVIEW_HEADER.join(",");
        output.push('\n');
        for cluster in clusters {
            let row = [
                cluster.cluster_id.to_string(),
                cluster.cluster_size.to_string(),
                cluster.outlier_count.to_string(),
                cluster.representative_sample_ids.clone(),
                cluster.representative_paths.clone(),
                cluster.current_majority_label.clone().unwrap_or_default(),
                if cluster.status == "labeled" {
                    cluster.label.clone()
                } else {
                    String::new()
                },
                cluster.status.clone(),
                cluster.notes.clone(),
            ];
            output.push_str(
                &row.iter()
                    .map(|value| escape_csv_field(value))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            output.push('\n');
        }

        std::fs::write(review_path, output).map_err(|error| {
            ClusterReviewError::Internal(format!(
                "Failed to write review file `{}`: {}",
                review_path.display(),
                error
            ))
        })
    }
}

pub fn load_review_state_from_paths(
    manifest_path: &Path,
    review_path: &Path,
) -> Result<ClusterReviewState, ClusterReviewError> {
    let mut clusters = load_current_clusters_from_manifest(manifest_path)?;
    let stale_reset_count = merge_saved_rows(review_path, &mut clusters)?;
    let reviewed_count = clusters
        .iter()
        .filter(|cluster| cluster.status != "unlabeled")
        .count();
    Ok(ClusterReviewState {
        review_file: review_path.display().to_string(),
        cluster_count: clusters.len(),
        reviewed_count,
        stale_reset_count,
        clusters,
    })
}

#[derive(Clone, Debug)]
struct ManifestClusterRow {
    sample_id: String,
    relative_path: String,
    label: String,
    label_source: String,
    annotation_status: String,
    review_status: String,
    cluster_distance: Option<f64>,
    is_cluster_outlier: bool,
}

#[derive(Clone, Debug)]
struct CsvRow {
    fields: BTreeMap<String, String>,
}

impl CsvRow {
    fn field(&self, name: &str) -> &str {
        self.fields.get(name).map(String::as_str).unwrap_or("")
    }
}

fn build_cluster_state(cluster_id: i64, members: &[ManifestClusterRow]) -> ClusterReviewCluster {
    let representatives = members
        .iter()
        .take(CLUSTER_REPRESENTATIVE_LIMIT)
        .map(|member| ClusterReviewRepresentative {
            sample_id: member.sample_id.clone(),
            relative_path: member.relative_path.clone(),
            label: if member.label.trim().is_empty() {
                "unknown".to_string()
            } else {
                member.label.clone()
            },
            label_source: if member.label_source.trim().is_empty() {
                "unknown".to_string()
            } else {
                member.label_source.clone()
            },
            annotation_status: if member.annotation_status.trim().is_empty() {
                "unlabeled".to_string()
            } else {
                member.annotation_status.clone()
            },
            review_status: if member.review_status.trim().is_empty() {
                "unreviewed".to_string()
            } else {
                member.review_status.clone()
            },
            cluster_distance: member.cluster_distance,
        })
        .collect::<Vec<_>>();

    let mut label_counts = BTreeMap::<String, usize>::new();
    for member in members {
        let label = member.label.trim();
        if label.is_empty() || label == "unknown" {
            continue;
        }
        *label_counts.entry(label.to_string()).or_insert(0) += 1;
    }

    let current_majority_label = label_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| left.0.cmp(right.0)))
        .map(|(label, _)| label.clone());

    ClusterReviewCluster {
        cluster_id,
        cluster_size: members.len(),
        outlier_count: members
            .iter()
            .filter(|member| member.is_cluster_outlier)
            .count(),
        representative_sample_ids: representatives
            .iter()
            .map(|member| member.sample_id.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        representative_paths: representatives
            .iter()
            .map(|member| member.relative_path.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        current_majority_label,
        label: String::new(),
        status: "unlabeled".to_string(),
        notes: String::new(),
        representatives,
    }
}

fn load_current_clusters_from_manifest(
    manifest_path: &Path,
) -> Result<Vec<ClusterReviewCluster>, ClusterReviewError> {
    if !manifest_path.is_file() {
        return Err(ClusterReviewError::InvalidRequest(format!(
            "Cluster review requires a dataset manifest at `{}`. Run `run-all` first.",
            manifest_path.display()
        )));
    }

    let rows = read_csv_rows(manifest_path)?;
    let mut grouped = BTreeMap::<i64, Vec<ManifestClusterRow>>::new();
    for row in rows {
        let Some(cluster_id) = parse_optional_i64(row.field("cluster_id"), "cluster_id")? else {
            continue;
        };
        grouped
            .entry(cluster_id)
            .or_default()
            .push(ManifestClusterRow {
                sample_id: row.field("sample_id").to_string(),
                relative_path: row.field("relative_path").to_string(),
                label: row.field("label").to_string(),
                label_source: row.field("label_source").to_string(),
                annotation_status: row.field("annotation_status").to_string(),
                review_status: row.field("review_status").to_string(),
                cluster_distance: parse_optional_f64(
                    row.field("cluster_distance"),
                    "cluster_distance",
                )?,
                is_cluster_outlier: parse_bool(row.field("is_cluster_outlier")),
            });
    }

    if grouped.is_empty() {
        return Err(ClusterReviewError::InvalidRequest(
            "No cluster assignments are available. Run `cluster` first.".to_string(),
        ));
    }

    let mut clusters = grouped
        .into_iter()
        .map(|(cluster_id, mut members)| {
            members.sort_by(|left, right| {
                left.cluster_distance
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right.cluster_distance.unwrap_or(f64::INFINITY))
                    .then_with(|| left.relative_path.cmp(&right.relative_path))
            });
            build_cluster_state(cluster_id, &members)
        })
        .collect::<Vec<_>>();
    clusters.sort_by(|left, right| {
        right
            .cluster_size
            .cmp(&left.cluster_size)
            .then(left.cluster_id.cmp(&right.cluster_id))
    });
    Ok(clusters)
}

fn merge_saved_rows(
    review_path: &Path,
    clusters: &mut [ClusterReviewCluster],
) -> Result<usize, ClusterReviewError> {
    if !review_path.is_file() {
        return Ok(0);
    }
    let saved_rows = read_csv_rows(review_path)?
        .into_iter()
        .filter_map(
            |row| match parse_optional_i64(row.field("cluster_id"), "cluster_id") {
                Ok(Some(cluster_id)) => Some(Ok((cluster_id, row))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    let mut stale_reset_count = 0;
    for cluster in clusters {
        let Some(row) = saved_rows.get(&cluster.cluster_id) else {
            continue;
        };
        let label = row.field("label");
        let status = normalize_status(row.field("status"), label)?;
        let normalized_label = normalize_label(label);
        if !fingerprints_match(
            cluster,
            parse_usize(row.field("cluster_size"), "cluster_size")?,
            parse_usize(row.field("outlier_count"), "outlier_count")?,
            row.field("representative_sample_ids"),
        ) {
            stale_reset_count += 1;
            continue;
        }
        if status == "labeled" && normalized_label.is_empty() {
            return Err(ClusterReviewError::InvalidRequest(format!(
                "Cluster review row for cluster {} is labeled but has no usable label.",
                cluster.cluster_id
            )));
        }
        cluster.status = status.to_string();
        cluster.label = if status == "labeled" {
            normalized_label
        } else {
            String::new()
        };
        cluster.notes = row.field("notes").trim().to_string();
    }
    Ok(stale_reset_count)
}

fn fingerprints_match(
    cluster: &ClusterReviewCluster,
    cluster_size: usize,
    outlier_count: usize,
    representative_sample_ids: &str,
) -> bool {
    cluster.cluster_size == cluster_size
        && cluster.outlier_count == outlier_count
        && cluster.representative_sample_ids == representative_sample_ids.trim()
}

fn normalize_status(raw_status: &str, raw_label: &str) -> Result<&'static str, ClusterReviewError> {
    if raw_status.trim().is_empty() {
        return Ok(if normalize_label(raw_label).is_empty() {
            "unlabeled"
        } else {
            "labeled"
        });
    }

    let normalized = raw_status.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "labeled" | "labelled" | "curated" => Ok("labeled"),
        "unlabeled" | "unlabelled" | "pending" | "unknown" => Ok("unlabeled"),
        "excluded" | "skip" | "ignored" => Ok("excluded"),
        _ => Err(ClusterReviewError::InvalidRequest(format!(
            "Unsupported cluster review status `{}`.",
            raw_status
        ))),
    }
}

fn normalize_label(raw_label: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_separator = false;
    for character in raw_label.trim().chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() {
            normalized.push(character);
            previous_was_separator = false;
            continue;
        }
        if !previous_was_separator && !normalized.is_empty() {
            normalized.push('_');
        }
        previous_was_separator = true;
    }
    normalized.trim_matches('_').to_string()
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn parse_optional_i64(value: &str, field_name: &str) -> Result<Option<i64>, ClusterReviewError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed.parse::<i64>().map(Some).map_err(|error| {
        ClusterReviewError::InvalidRequest(format!(
            "Failed to parse `{}` value `{}` as integer: {}",
            field_name, value, error
        ))
    })
}

fn parse_optional_f64(value: &str, field_name: &str) -> Result<Option<f64>, ClusterReviewError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed.parse::<f64>().map(Some).map_err(|error| {
        ClusterReviewError::InvalidRequest(format!(
            "Failed to parse `{}` value `{}` as float: {}",
            field_name, value, error
        ))
    })
}

fn parse_usize(value: &str, field_name: &str) -> Result<usize, ClusterReviewError> {
    value.trim().parse::<usize>().map_err(|error| {
        ClusterReviewError::InvalidRequest(format!(
            "Failed to parse `{}` value `{}` as integer: {}",
            field_name, value, error
        ))
    })
}

fn read_csv_rows(path: &Path) -> Result<Vec<CsvRow>, ClusterReviewError> {
    let content = std::fs::read_to_string(path).map_err(|error| {
        ClusterReviewError::Internal(format!("Failed to read `{}`: {}", path.display(), error))
    })?;
    let records = parse_csv_records(&content)?;
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let headers = records[0].clone();
    let mut rows = Vec::new();
    for record in records.into_iter().skip(1) {
        if record.iter().all(|value| value.is_empty()) {
            continue;
        }
        let mut fields = BTreeMap::new();
        for (index, header) in headers.iter().enumerate() {
            fields.insert(
                header.clone(),
                record.get(index).cloned().unwrap_or_default(),
            );
        }
        rows.push(CsvRow { fields });
    }
    Ok(rows)
}

fn parse_csv_records(content: &str) -> Result<Vec<Vec<String>>, ClusterReviewError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = normalized.chars().peekable();
    let mut in_quotes = false;

    while let Some(character) = chars.next() {
        match character {
            '"' => {
                if in_quotes {
                    if matches!(chars.peek(), Some('"')) {
                        field.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else if field.is_empty() {
                    in_quotes = true;
                } else {
                    field.push(character);
                }
            }
            ',' if !in_quotes => {
                record.push(std::mem::take(&mut field));
            }
            '\n' if !in_quotes => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            _ => field.push(character),
        }
    }

    if in_quotes {
        return Err(ClusterReviewError::InvalidRequest(
            "CSV contains an unterminated quoted field.".to_string(),
        ));
    }

    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }

    Ok(records)
}

fn escape_csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClusterReviewCluster, ClusterReviewError, ClusterReviewSaveCluster,
        ClusterReviewSaveRequest, ClusterReviewStore,
    };
    use crate::RuntimeConfig;

    fn temp_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "localagent_cluster_review_{name}_{}",
            std::process::id()
        ))
    }

    fn write_manifest(root: &std::path::Path, rows: &[&str]) {
        let manifest_dir = root.join("artifacts").join("manifests");
        std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
        let path = manifest_dir.join("dataset_manifest.csv");
        let mut content = [
            "sample_id,relative_path,label,label_source,annotation_status,review_status,cluster_id,cluster_distance,is_cluster_outlier",
            &rows.join("\n"),
        ]
        .join("\n");
        content.push('\n');
        std::fs::write(path, content).expect("failed to write manifest");
    }

    fn write_review(root: &std::path::Path, rows: &[&str]) {
        let manifest_dir = root.join("artifacts").join("manifests");
        std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
        let path = manifest_dir.join("cluster_review.csv");
        let mut content = [
            "cluster_id,cluster_size,outlier_count,representative_sample_ids,representative_paths,current_majority_label,label,status,notes",
            &rows.join("\n"),
        ]
        .join("\n");
        content.push('\n');
        std::fs::write(path, content).expect("failed to write review");
    }

    fn build_store(root: &std::path::Path) -> ClusterReviewStore {
        ClusterReviewStore::new(RuntimeConfig {
            artifact_dir: root.join("artifacts").display().to_string(),
            ..RuntimeConfig::default()
        })
    }

    fn cluster_by_id(clusters: &[ClusterReviewCluster], cluster_id: i64) -> &ClusterReviewCluster {
        clusters
            .iter()
            .find(|cluster| cluster.cluster_id == cluster_id)
            .expect("missing cluster")
    }

    #[test]
    fn loads_and_merges_saved_review_rows() {
        let root = temp_root("load_merge");
        let _ = std::fs::remove_dir_all(&root);
        write_manifest(
            &root,
            &[
                "sample-a,a.jpg,r,filename,inferred,unreviewed,1,0.10,false",
                "sample-b,b.jpg,r,filename,inferred,unreviewed,1,0.15,true",
                "sample-c,c.jpg,glass,cluster_review,labeled,cluster_accepted,2,0.20,false",
                "sample-d,d.jpg,glass,cluster_review,labeled,cluster_accepted,2,0.25,false",
            ],
        );
        write_review(
            &root,
            &[
                "1,2,1,sample-a|sample-b,a.jpg|b.jpg,r,metal,labeled,sorted by ui",
                "2,2,0,sample-c|sample-d,c.jpg|d.jpg,glass,,unlabeled,",
            ],
        );

        let store = build_store(&root);
        let state = store
            .load_review_state(None)
            .expect("failed to load review state");

        assert_eq!(state.cluster_count, 2);
        assert_eq!(state.reviewed_count, 1);
        assert_eq!(state.stale_reset_count, 0);
        assert_eq!(cluster_by_id(&state.clusters, 1).label, "metal");
        assert_eq!(cluster_by_id(&state.clusters, 1).status, "labeled");
        assert_eq!(
            cluster_by_id(&state.clusters, 2)
                .current_majority_label
                .as_deref(),
            Some("glass")
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resets_stale_saved_rows_on_load() {
        let root = temp_root("stale_load");
        let _ = std::fs::remove_dir_all(&root);
        write_manifest(
            &root,
            &[
                "sample-a,a.jpg,r,filename,inferred,unreviewed,7,0.10,false",
                "sample-b,b.jpg,r,filename,inferred,unreviewed,7,0.15,false",
            ],
        );
        write_review(
            &root,
            &["7,3,0,sample-a|sample-b,a.jpg|b.jpg,r,glass,labeled,stale size"],
        );

        let store = build_store(&root);
        let state = store
            .load_review_state(None)
            .expect("failed to load review state");

        assert_eq!(state.stale_reset_count, 1);
        assert_eq!(state.reviewed_count, 0);
        assert_eq!(cluster_by_id(&state.clusters, 7).status, "unlabeled");
        assert!(cluster_by_id(&state.clusters, 7).label.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_review_rejects_stale_payload() {
        let root = temp_root("stale_save");
        let _ = std::fs::remove_dir_all(&root);
        write_manifest(
            &root,
            &[
                "sample-a,a.jpg,r,filename,inferred,unreviewed,4,0.10,false",
                "sample-b,b.jpg,r,filename,inferred,unreviewed,4,0.15,false",
            ],
        );
        let store = build_store(&root);
        let error = store
            .save_review(ClusterReviewSaveRequest {
                review_file: None,
                clusters: vec![ClusterReviewSaveCluster {
                    cluster_id: 4,
                    cluster_size: 3,
                    outlier_count: 0,
                    representative_sample_ids: "sample-a|sample-b".to_string(),
                    representative_paths: "a.jpg|b.jpg".to_string(),
                    label: Some("glass".to_string()),
                    status: "labeled".to_string(),
                    notes: Some("draft".to_string()),
                }],
            })
            .expect_err("expected stale save to fail");

        match error {
            ClusterReviewError::Conflict {
                stale_cluster_ids, ..
            } => assert_eq!(stale_cluster_ids, vec![4]),
            other => panic!("unexpected error: {other}"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_review_writes_normalized_labels() {
        let root = temp_root("save_ok");
        let _ = std::fs::remove_dir_all(&root);
        write_manifest(
            &root,
            &[
                "sample-a,a.jpg,r,filename,inferred,unreviewed,9,0.10,false",
                "sample-b,b.jpg,r,filename,inferred,unreviewed,9,0.15,true",
            ],
        );
        let store = build_store(&root);
        let state = store
            .save_review(ClusterReviewSaveRequest {
                review_file: None,
                clusters: vec![ClusterReviewSaveCluster {
                    cluster_id: 9,
                    cluster_size: 2,
                    outlier_count: 1,
                    representative_sample_ids: "sample-a|sample-b".to_string(),
                    representative_paths: "a.jpg|b.jpg".to_string(),
                    label: Some("Metal Can".to_string()),
                    status: "labeled".to_string(),
                    notes: Some("approved".to_string()),
                }],
            })
            .expect("expected save to succeed");

        assert_eq!(state.reviewed_count, 1);
        assert_eq!(cluster_by_id(&state.clusters, 9).label, "metal_can");
        let review_path = root
            .join("artifacts")
            .join("manifests")
            .join("cluster_review.csv");
        let content = std::fs::read_to_string(review_path).expect("failed to read saved review");
        assert!(content.contains("metal_can"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
