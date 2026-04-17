use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, ImageFormat};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use tracing::warn;

#[derive(Debug, Serialize)]
pub struct CacheBuildSummary {
    pub total: usize,
    pub processed: usize,
    pub skipped: usize,
    pub errors: usize,
    pub cache_dir: String,
    pub cache_format: String,
    pub failure_report_path: Option<String>,
    pub image_size: u32,
}

#[derive(Debug, Serialize)]
pub struct CacheFailureRecord {
    pub sample_id: String,
    pub image_path: String,
    pub error: String,
}

#[derive(Clone, Copy, Debug)]
pub enum CacheFormat {
    Png,
    Raw,
}

impl CacheFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "png" => Ok(Self::Png),
            "raw" => Ok(Self::Raw),
            other => Err(anyhow!("unsupported cache format: {other}")),
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Raw => "raw",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Raw => "raw",
        }
    }
}

pub fn build_training_cache(
    entries: &[(String, String)],
    cache_dir: &Path,
    failure_report_path: Option<&Path>,
    image_size: u32,
    cache_format: CacheFormat,
    force: bool,
    show_progress: bool,
) -> Result<CacheBuildSummary> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create cache directory: {}", cache_dir.display()))?;

    let progress = if show_progress {
        let bar = ProgressBar::new(entries.len() as u64);
        let style = ProgressStyle::with_template(
            "{spinner:.green} {msg:<18} [{bar:40.cyan/blue}] {pos:>6}/{len:<6} {percent:>3}% {elapsed_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("##-");
        bar.set_style(style);
        bar.set_message("Rust image cache");
        Some(bar)
    } else {
        None
    };

    let results = entries
        .par_iter()
        .map(|(sample_id, image_path)| {
            let output_path = cache_output_path(cache_dir, sample_id, cache_format);
            let result = prepare_single_image(
                sample_id,
                Path::new(image_path),
                &output_path,
                image_size,
                cache_format,
                force,
            );
            if let Some(bar) = &progress {
                bar.inc(1);
            }
            result.map_err(|error| (sample_id.clone(), image_path.clone(), error))
        })
        .collect::<Vec<_>>();

    if let Some(bar) = &progress {
        bar.finish_with_message(format!("Rust image cache ready: {}", cache_dir.display()));
    }

    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;
    let mut failures: Vec<CacheFailureRecord> = Vec::new();

    for result in results {
        match result {
            Ok(CacheOutcome::Written) => processed += 1,
            Ok(CacheOutcome::SkippedExisting) => skipped += 1,
            Err((sample_id, image_path, error)) => {
                errors += 1;
                failures.push(CacheFailureRecord {
                    sample_id,
                    image_path,
                    error: error.to_string(),
                });
                warn!(error = %error, "failed to cache training image");
            }
        }
    }

    if let Some(report_path) = failure_report_path {
        write_failure_report(report_path, &failures)?;
    }

    Ok(CacheBuildSummary {
        total: entries.len(),
        processed,
        skipped,
        errors,
        cache_dir: cache_dir.display().to_string(),
        cache_format: cache_format.as_str().to_string(),
        failure_report_path: failure_report_path.map(|path| path.display().to_string()),
        image_size,
    })
}

enum CacheOutcome {
    Written,
    SkippedExisting,
}

fn prepare_single_image(
    sample_id: &str,
    image_path: &Path,
    output_path: &Path,
    image_size: u32,
    cache_format: CacheFormat,
    force: bool,
) -> Result<CacheOutcome> {
    if output_path.exists() && !force {
        return Ok(CacheOutcome::SkippedExisting);
    }

    let image = image::open(image_path).with_context(|| {
        format!(
            "failed to decode image for sample {sample_id}: {}",
            image_path.display()
        )
    })?;
    let resized = image::imageops::resize(
        &image.to_rgb8(),
        image_size,
        image_size,
        FilterType::Triangle,
    );
    match cache_format {
        CacheFormat::Png => {
            resized
                .save_with_format(output_path, ImageFormat::Png)
                .with_context(|| {
                    format!(
                        "failed to write cached image for sample {sample_id}: {}",
                        output_path.display()
                    )
                })?;
        }
        CacheFormat::Raw => {
            std::fs::write(output_path, resized.as_raw()).with_context(|| {
                format!(
                    "failed to write raw cached image for sample {sample_id}: {}",
                    output_path.display()
                )
            })?;
        }
    }
    Ok(CacheOutcome::Written)
}

fn cache_output_path(cache_dir: &Path, sample_id: &str, cache_format: CacheFormat) -> PathBuf {
    cache_dir.join(format!("{sample_id}.{}", cache_format.extension()))
}

fn write_failure_report(report_path: &Path, failures: &[CacheFailureRecord]) -> Result<()> {
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create cache failure report directory: {}",
                parent.display()
            )
        })?;
    }

    let payload = serde_json::to_string_pretty(failures)
        .context("failed to serialize cache failure report")?;
    std::fs::write(report_path, payload).with_context(|| {
        format!(
            "failed to write cache failure report: {}",
            report_path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{build_training_cache, CacheFormat};

    #[test]
    fn writes_failure_report_for_unreadable_images() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_cache_test_{}", std::process::id()));
        let cache_dir = test_root.join("cache");
        let report_path = test_root.join("reports").join("cache_failures.json");
        let valid_path = test_root.join("valid.png");
        let invalid_path = test_root.join("broken.png");

        std::fs::create_dir_all(&test_root).expect("failed to create test root");
        let image = image::RgbImage::from_pixel(8, 8, image::Rgb([12, 34, 56]));
        image.save(&valid_path).expect("failed to save valid image");
        std::fs::write(&invalid_path, b"not-an-image").expect("failed to write broken image");

        let entries = vec![
            ("valid".to_string(), valid_path.display().to_string()),
            ("broken".to_string(), invalid_path.display().to_string()),
        ];

        let summary = build_training_cache(
            &entries,
            &cache_dir,
            Some(&report_path),
            32,
            CacheFormat::Png,
            false,
            false,
        )
        .expect("cache build failed");

        assert_eq!(summary.total, 2);
        assert_eq!(summary.processed, 1);
        assert_eq!(summary.errors, 1);
        assert!(report_path.is_file());

        let report =
            std::fs::read_to_string(&report_path).expect("failed to read cache failure report");
        assert!(report.contains("\"sample_id\": \"broken\""));

        let _ = std::fs::remove_dir_all(&test_root);
    }

    #[test]
    fn writes_raw_cache_without_png_encoding() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_raw_cache_test_{}", std::process::id()));
        let cache_dir = test_root.join("cache");
        let image_path = test_root.join("valid.png");

        std::fs::create_dir_all(&test_root).expect("failed to create test root");
        let image = image::RgbImage::from_pixel(4, 4, image::Rgb([12, 34, 56]));
        image.save(&image_path).expect("failed to save valid image");

        let entries = vec![("valid".to_string(), image_path.display().to_string())];

        let summary = build_training_cache(
            &entries,
            &cache_dir,
            None,
            8,
            CacheFormat::Raw,
            false,
            false,
        )
        .expect("cache build failed");

        assert_eq!(summary.processed, 1);
        let output_path = cache_dir.join("valid.raw");
        assert!(output_path.is_file());
        assert_eq!(
            std::fs::metadata(output_path)
                .expect("missing raw cache")
                .len(),
            8 * 8 * 3
        );

        let _ = std::fs::remove_dir_all(&test_root);
    }
}
