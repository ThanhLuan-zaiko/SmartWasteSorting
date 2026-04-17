use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ClassificationReport {
    pub labels: Vec<String>,
    pub num_samples: usize,
    pub accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub per_class: BTreeMap<String, PerClassMetrics>,
    pub confusion_matrix: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize)]
pub struct PerClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

pub fn compute_class_weight_map(
    train_labels: &[String],
    class_names: &[String],
) -> BTreeMap<String, f64> {
    let mut counts: BTreeMap<String, usize> = class_names
        .iter()
        .cloned()
        .map(|label| (label, 0usize))
        .collect();

    for label in train_labels {
        if let Some(count) = counts.get_mut(label) {
            *count += 1;
        }
    }

    let present_labels: Vec<_> = counts
        .iter()
        .filter_map(|(label, count)| {
            if *count > 0 {
                Some(label.clone())
            } else {
                None
            }
        })
        .collect();
    if present_labels.is_empty() {
        return counts
            .keys()
            .cloned()
            .map(|label| (label, 1.0_f64))
            .collect();
    }

    let total_samples: usize = present_labels
        .iter()
        .filter_map(|label| counts.get(label).copied())
        .sum();
    let num_present_labels = present_labels.len() as f64;

    counts
        .into_iter()
        .map(|(label, count)| {
            let weight = if count == 0 {
                0.0
            } else {
                total_samples as f64 / (num_present_labels * count as f64)
            };
            (label, weight)
        })
        .collect()
}

pub fn build_classification_report(
    predictions: &[usize],
    targets: &[usize],
    class_names: &[String],
) -> Result<ClassificationReport> {
    if predictions.len() != targets.len() {
        return Err(anyhow!(
            "predictions and targets must have the same length: {} != {}",
            predictions.len(),
            targets.len()
        ));
    }

    let num_classes = class_names.len();
    let mut confusion_matrix = vec![vec![0usize; num_classes]; num_classes];
    for (target, prediction) in targets.iter().zip(predictions.iter()) {
        if *target < num_classes && *prediction < num_classes {
            confusion_matrix[*target][*prediction] += 1;
        }
    }

    let mut per_class = BTreeMap::new();
    let mut macro_precision = 0.0_f64;
    let mut macro_recall = 0.0_f64;
    let mut macro_f1 = 0.0_f64;
    let mut weighted_f1 = 0.0_f64;
    let mut total_support = 0usize;
    let mut total_correct = 0usize;

    for (class_index, label) in class_names.iter().enumerate() {
        let true_positives = confusion_matrix[class_index][class_index];
        let support: usize = confusion_matrix[class_index].iter().sum();
        let predicted_count: usize = confusion_matrix.iter().map(|row| row[class_index]).sum();
        let false_positives = predicted_count.saturating_sub(true_positives);
        let false_negatives = support.saturating_sub(true_positives);
        let precision = if predicted_count > 0 {
            true_positives as f64 / predicted_count as f64
        } else {
            0.0
        };
        let recall = if support > 0 {
            true_positives as f64 / support as f64
        } else {
            0.0
        };
        let f1 = if (precision + recall) > 0.0 {
            (2.0 * precision * recall) / (precision + recall)
        } else {
            0.0
        };

        per_class.insert(
            label.clone(),
            PerClassMetrics {
                precision,
                recall,
                f1,
                support,
                true_positives,
                false_positives,
                false_negatives,
            },
        );
        macro_precision += precision;
        macro_recall += recall;
        macro_f1 += f1;
        weighted_f1 += f1 * support as f64;
        total_support += support;
        total_correct += true_positives;
    }

    let divisor = num_classes.max(1) as f64;
    Ok(ClassificationReport {
        labels: class_names.to_vec(),
        num_samples: total_support,
        accuracy: if total_support > 0 {
            total_correct as f64 / total_support as f64
        } else {
            0.0
        },
        macro_precision: macro_precision / divisor,
        macro_recall: macro_recall / divisor,
        macro_f1: macro_f1 / divisor,
        weighted_f1: if total_support > 0 {
            weighted_f1 / total_support as f64
        } else {
            0.0
        },
        per_class,
        confusion_matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_classification_report, compute_class_weight_map};

    #[test]
    fn computes_balanced_class_weights() {
        let class_names = vec!["glass".to_string(), "plastic".to_string()];
        let labels = vec![
            "glass".to_string(),
            "glass".to_string(),
            "plastic".to_string(),
        ];

        let weight_map = compute_class_weight_map(&labels, &class_names);

        assert!(weight_map["plastic"] > weight_map["glass"]);
    }

    #[test]
    fn builds_confusion_matrix_metrics() {
        let class_names = vec!["glass".to_string(), "plastic".to_string()];
        let report = build_classification_report(&[0, 1, 1, 1], &[0, 1, 0, 1], &class_names)
            .expect("failed to build report");

        assert_eq!(report.num_samples, 4);
        assert_eq!(report.confusion_matrix[0][1], 1);
        assert_eq!(report.per_class["glass"].support, 2);
        assert_eq!(report.accuracy, 0.75);
    }
}
