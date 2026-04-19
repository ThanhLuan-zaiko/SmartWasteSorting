# Tài liệu 04: Pseudo-label, ONNX, artifact và lưu trữ cục bộ

## 1. File này dùng để làm gì

- File này giải thích các file đầu ra cục bộ của dự án.
- File này giải thích pseudo-label chạy ra sao.
- File này giải thích checkpoint, report, benchmark, ONNX, labels, model manifest, cache và embedding artifact.
- Nếu bạn muốn biết “file nào sinh ở bước nào”, file này là nơi nên đọc.

## 2. Các loại file cục bộ chính của dự án

- `.parquet`
- `.csv`
- `.json`
- `.jsonl`
- `.npz`
- `.pt`
- `.onnx`
- `.raw`
- `.png`

## 3. Cấu trúc lưu trữ quan trọng nhất

- `localagent/artifacts/manifests/`
- `localagent/artifacts/reports/`
- `localagent/artifacts/checkpoints/`
- `localagent/artifacts/cache/`
- `localagent/artifacts/jobs/`
- `localagent/models/`

## 4. Tư duy đúng về artifact

- Artifact không chỉ là “file log”.
- Artifact là hợp đồng giữa các bước pipeline.
- Artifact của bước trước thường là đầu vào logic của bước sau.
- UI cũng phụ thuộc trực tiếp hoặc gián tiếp vào artifact.

## 5. File `.parquet` quan trọng nhất

- `localagent/artifacts/manifests/dataset_manifest.parquet`

## 6. File `.csv` quan trọng song song với manifest

- `localagent/artifacts/manifests/dataset_manifest.csv`

## 7. Vì sao vừa có parquet vừa có csv

- Parquet tốt cho thao tác bảng và lưu kiểu dữ liệu.
- CSV dễ mở bằng công cụ phổ thông.
- Rust cluster review logic hiện đọc manifest CSV cho một số workflow UI.

## 8. File `.npz` quan trọng nhất của discovery

- `localagent/artifacts/manifests/dataset_embeddings.npz`

## 9. File `.csv` quan trọng nhất của review cụm

- `localagent/artifacts/manifests/cluster_review.csv`

## 10. File `.csv` quan trọng nhất của gán nhãn thủ công

- `localagent/artifacts/manifests/labeling_template.csv`

## 11. File `.json` tổng hợp dữ liệu quan trọng nhất

- `localagent/artifacts/reports/summary.json`

## 12. File `.json` mô tả embedding

- `localagent/artifacts/reports/embedding_summary.json`

## 13. File `.json` mô tả clustering

- `localagent/artifacts/reports/cluster_summary.json`

## 14. File `.pt` checkpoint tốt nhất

- `localagent/artifacts/checkpoints/<experiment_name>.pt`

## 15. File `.pt` checkpoint gần nhất để resume

- `localagent/artifacts/checkpoints/<experiment_name>.last.pt`

## 16. File `.json` training report

- `localagent/artifacts/reports/<experiment_name>_training.json`

## 17. File `.json` evaluation report

- `localagent/artifacts/reports/<experiment_name>_evaluation.json`

## 18. File `.csv` confusion matrix

- `localagent/artifacts/reports/<experiment_name>_confusion_matrix.csv`

## 19. File `.json` pseudo-label report

- `localagent/artifacts/reports/<experiment_name>_pseudo_label.json`

## 20. File `.json` export report

- `localagent/artifacts/reports/<experiment_name>_export.json`

## 21. File `.json` benchmark report

- `localagent/artifacts/reports/<experiment_name>_benchmark.json`

## 22. File `.json` experiment spec

- `localagent/artifacts/reports/<experiment_name>_experiment_spec.json`

## 23. File `.json` bundle report

- `localagent/artifacts/reports/<experiment_name>_report.json`

## 24. File `.json` catalog run

- `localagent/artifacts/reports/run_index.json`

## 25. File `.onnx` model export

- `localagent/models/waste_classifier.onnx`

## 26. File `.json` labels export

- `localagent/models/labels.json`

## 27. File `.json` model manifest

- `localagent/models/model_manifest.json`

## 28. Cache ảnh dạng `.raw` hoặc `.png` nằm ở đâu

- `localagent/artifacts/cache/training/<image_size>px/`
- `localagent/artifacts/cache/training/<image_size>px-raw/`

## 29. Vị trí code sinh artifact path training

- `_checkpoint_path` ở `localagent/python/localagent/training/trainer.py:1891`
- `_latest_checkpoint_path` ở `:1894`
- `_evaluation_report_path` ở `:1897`
- `_confusion_matrix_path` ở `:1904`
- `_training_report_path` ở `:1911`
- `_experiment_spec_path` ở `:1914`
- `_benchmark_report_path` ở `:1921`
- `_pseudo_label_report_path` ở `:1928`
- `_export_report_path` ở `:1935`
- `_artifact_bundle_path` ở `:1938`
- `_model_manifest_path` ở `:1941`

## 30. Artifact của Step 1 sinh ra khi nào

- Ngay sau `run-all`, `scan`, `split`, `report` hoặc các bước cập nhật manifest.

## 31. Artifact của discovery sinh ra khi nào

- `dataset_embeddings.npz` sinh khi chạy `embed`.
- `embedding_summary.json` sinh khi chạy `embed`.
- `cluster_summary.json` sinh khi chạy `cluster`.
- `cluster_review.csv` sinh khi chạy `export-cluster-review`.

## 32. Artifact của training sinh ra khi nào

- Checkpoint sinh trong `fit`.
- Training report sinh cuối `fit`.
- Evaluation report sinh trong `fit` nếu có split test, hoặc khi chạy `evaluate`.
- Export report sinh trong `export-onnx`.
- Benchmark report sinh trong `benchmark`.

## 33. Pseudo-label có vai trò gì

- Nó là bước bán giám sát sau khi đã có một model ban đầu.
- Model đang có được dùng để dự đoán các sample chưa được accepted.
- Chỉ những sample có confidence đủ cao và margin đủ lớn mới được nhận.
- Sample bị reject không biến mất.
- Sample bị reject chỉ được cập nhật `suggested_label` và các score để người dùng biết.

## 34. Vị trí code của `pseudo_label`

- `localagent/python/localagent/training/trainer.py:1126`

## 35. Điều kiện để pseudo-label chạy

- Backend phải là `pytorch`.
- Manifest phải tồn tại.
- Checkpoint phải tồn tại.
- Dataset summary phải chứng minh Step 2 đã hoàn tất.

## 36. `pseudo_label` lấy candidate như thế nào

- Chỉ lấy sample `is_valid`.
- Chỉ lấy sample có `label_source` là `unknown` hoặc `filename`.
- Bỏ qua sample đã `excluded`, `labeled` hoặc `pseudo_labeled`.

## 37. Vị trí code chọn candidate pseudo-label

- `_pseudo_label_candidates` ở `localagent/python/localagent/training/trainer.py:1709`

## 38. Một vòng pseudo-label làm gì

- Load manifest.
- Load checkpoint.
- Lấy class names từ checkpoint hoặc manifest.
- Lọc candidate.
- Build model và load state dict.
- Chạy suy luận từng sample candidate.
- Tính softmax.
- Lấy top 1 score và top 2 score.
- Tính margin.
- So với ngưỡng confidence và margin.
- Update manifest tương ứng.
- Re-assign split.
- Ghi report pseudo-label.

## 39. Công thức margin đang dùng

- `margin = top1_score - top2_score`

## 40. Công thức điều kiện accept pseudo-label

- `score >= pseudo_label_confidence_threshold`
- Và `margin >= pseudo_label_margin_threshold`

## 41. Ý nghĩa của confidence threshold

- Yêu cầu model phải đủ tự tin về lớp đứng đầu.

## 42. Ý nghĩa của margin threshold

- Yêu cầu model phải phân biệt lớp đứng đầu và lớp đứng thứ hai đủ rõ.

## 43. Vì sao phải dùng cả hai

- Chỉ dùng confidence có thể nhận các sample mơ hồ khi phân phối xác suất bị lệch.
- Thêm margin giúp giảm trường hợp top 1 và top 2 quá sát nhau.

## 44. Report pseudo-label hiện tại của repo cho biết gì

- `candidate_count = 790`
- `accepted_count = 570`
- `rejected_count = 220`
- `confidence_threshold = 0.95`
- `margin_threshold = 0.25`
- `average_accepted_score ≈ 0.9944`

## 45. Điều đó cho bạn biết gì

- Pseudo-label chỉ được chấp nhận khá chặt.
- Dù chặt như vậy vẫn thêm được 570 sample accepted.
- 220 sample vẫn bị từ chối để giữ độ an toàn.

## 46. Pseudo-label update manifest thế nào khi accept

- `annotation_status = pseudo_labeled`
- `label_source = model_pseudo`
- `review_status = pseudo_accepted`
- `suggested_label = predicted_label`
- `suggested_label_source = model_pseudo`
- `pseudo_label_score` được ghi
- `pseudo_label_margin` được ghi

## 47. Pseudo-label update manifest thế nào khi reject

- Không đổi `annotation_status` sang pseudo_labeled.
- Không biến sample thành accepted label.
- Chỉ ghi `suggested_label`.
- Chỉ ghi `suggested_label_source = model_pseudo`.
- Ghi `pseudo_label_score`.
- Ghi `pseudo_label_margin`.
- Ghi `review_status = pseudo_rejected`.

## 48. Sau pseudo-label vì sao phải re-assign split

- Vì có thêm sample trainable mới.
- Những sample được accept phải được đưa vào `train`, `val` hoặc `test`.

## 49. Vị trí code re-assign split sau pseudo-label

- `updated_frame = pipeline.assign_splits(...)` trong `WasteTrainer.pseudo_label`.

## 50. `labels.json` là gì

- Một mảng JSON chứa danh sách lớp theo thứ tự index của model.
- Đây không phải map phức tạp.
- Ví dụ hiện tại repo có:
- `["folk", "glass", "paper"]`

## 51. Vị trí code ghi `labels.json`

- `_write_labels_payload` ở `localagent/python/localagent/training/trainer.py:1759`

## 52. Khi nào `labels.json` được ghi

- Trong `export_label_index`
- Trong `fit`
- Trong `export_onnx`

## 53. `model_manifest.json` là gì

- Đây là metadata cần cho inference.
- Nó nói model nằm ở đâu.
- Nó nói labels nằm ở đâu.
- Nó nói image size là gì.
- Nó nói normalization mean/std là gì.
- Nó nói input spec và output spec là gì.
- Nó nói trạng thái verify ONNX là gì.
- Nó có thể chứa evaluation summary.

## 54. Vị trí code build `model_manifest.json`

- `_build_model_manifest` ở `localagent/python/localagent/training/trainer.py:1831`

## 55. `model_manifest.json` hiện tại của repo cho biết gì

- `model_name = mobilenet_v3_small`
- `onnx_path = localagent/models/waste_classifier.onnx`
- `labels = ["folk", "glass", "paper"]`
- `image_size = 160`
- `normalization.preset = imagenet`
- `onnx.opset = 17`
- `verification.verified = true`
- `evaluation_summary.accuracy ≈ 0.9296`

## 56. Vì sao `model_manifest.json` quan trọng hơn nhiều người tưởng

- Rust inference ảnh upload dựa vào file này để biết cách preprocess.
- Nếu file này sai, ONNX vẫn có thể chạy nhưng kết quả sẽ sai.

## 57. `export_report.json` là gì

- Là báo cáo riêng cho bước export ONNX.
- Nó không chứa mọi artifact khác.
- Nó tập trung vào đường dẫn ONNX, labels, spec, verification.

## 58. `training_report.json` là gì

- Là báo cáo riêng cho bước fit.
- Nó chứa history theo epoch.
- Nó chứa checkpoint path.
- Nó chứa class weight map.
- Nó chứa summary evaluation nếu fit đã tự test best model.

## 59. `evaluation_report.json` là gì

- Là báo cáo metrics chi tiết cho một checkpoint trên split đánh giá.
- Nó chứa `accuracy`, `macro_f1`, `weighted_f1`.
- Nó chứa `per_class`.
- Nó chứa `confusion_matrix`.

## 60. `benchmark_report.json` là gì

- Là báo cáo end to end cho chuỗi `fit -> evaluate -> export_onnx -> report`.
- Nó chứa duration từng stage.
- Nó chứa metrics tóm tắt.
- Nó chứa backend capability.

## 61. `experiment_spec.json` là gì

- Là snapshot cấu hình thí nghiệm.
- Nó giúp bạn biết run đó dùng preset nào, batch size nào, image size nào, early stopping ra sao.

## 62. `bundle report` là gì

- Là file hợp nhất nhiều artifact lại một chỗ.
- Nó giúp API `overview` và dashboard đọc nhanh hơn.

## 63. Cache `.raw` là gì

- Là mảng byte RGB của ảnh đã resize.
- Không có metadata kèm theo trong file.
- Kích thước phải suy ra từ `image_size`.
- Ví dụ ảnh `160x160x3` sẽ có `76800` byte.

## 64. Cache `.png` là gì

- Là ảnh PNG đã resize sẵn.
- Dễ inspect thủ công hơn.
- Nhưng khi training phải decode PNG lại.

## 65. Khi nào nên xem trực tiếp file artifact

- Khi muốn debug pipeline.
- Khi API trả số liệu lạ.
- Khi UI hiển thị sai.
- Khi muốn kiểm tra model gần nhất là model nào.

## 66. Vị trí code Rust đọc artifact

- `ArtifactStore.read_artifact` ở `localagent/src/artifacts.rs`

## 67. `ArtifactStore` map loại artifact sang path thế nào

- `DatasetSummary`
- `Training`
- `Evaluation`
- `Export`
- `Benchmark`
- `ExperimentSpec`
- `Bundle`
- `ModelManifest`

## 68. Vì sao `run_index.json` tồn tại

- UI cần một catalog run để hiển thị danh sách experiment.
- Job manager khi hoàn tất job sẽ sync lại run index.

## 69. Vị trí code sync `run_index`

- `ArtifactStore.sync_run_index` ở `localagent/src/artifacts.rs`
- `JobManager.finish_job` ở `localagent/src/jobs/runtime.rs` gọi sync lại.

## 70. Job JSON là artifact kiểu gì

- Đây là artifact vận hành hơn là artifact học máy.
- Nó lưu `job_id`, `job_type`, `command`, `status`, `stdout_log_path`, `stderr_log_path`, `artifacts`.

## 71. Vị trí code schema job

- `localagent/src/jobs/types.rs`

## 72. Pseudo-label report có phải artifact duy nhất cập nhật sample chưa nhãn không

- Không.
- `cluster_review` và `import_labels` cũng cập nhật sample chưa nhãn.

## 73. Khi nào nên chạy pseudo-label

- Sau khi đã có model baseline đủ ổn.
- Sau khi accepted labels ban đầu đã đủ đại diện.
- Sau khi bạn hiểu rõ ngưỡng accept đang dùng.

## 74. Khi nào không nên chạy pseudo-label

- Khi model baseline còn quá yếu.
- Khi số lớp còn chưa ổn định.
- Khi dữ liệu đang bị lệch nghiêm trọng mà model thiên nặng một lớp.

## 75. Command pseudo-label ví dụ

```powershell
cd localagent
uv run python -m localagent.training.train pseudo-label --checkpoint artifacts/checkpoints/baseline-waste-sorter.pt --pseudo-label-threshold 0.95 --pseudo-label-margin 0.25
```

## 76. Command evaluate ví dụ

```powershell
cd localagent
uv run python -m localagent.training.train evaluate --checkpoint artifacts/checkpoints/baseline-waste-sorter.pt
```

## 77. Command export ONNX ví dụ

```powershell
cd localagent
uv run python -m localagent.training.train export-onnx --checkpoint artifacts/checkpoints/baseline-waste-sorter.pt
```

## 78. Command report ví dụ

```powershell
cd localagent
uv run python -m localagent.training.train report
```

## 79. Command benchmark ví dụ

```powershell
cd localagent
uv run python -m localagent.training.train benchmark --training-preset cpu_fast --experiment-name baseline-waste-sorter --no-progress
```

## 80. Accuracy hiện tại của repo nằm trong file nào

- `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json`
- `localagent/models/model_manifest.json`
- `localagent/artifacts/reports/baseline-waste-sorter_benchmark.json`

## 81. Confusion matrix hiện tại nằm trong file nào

- `localagent/artifacts/reports/baseline-waste-sorter_confusion_matrix.csv`
- `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json`

## 82. Duration fit hiện tại nằm trong file nào

- `localagent/artifacts/reports/baseline-waste-sorter_benchmark.json`

## 83. Nếu muốn biết run nào mới nhất

- Xem `localagent/artifacts/reports/run_index.json`
- Hoặc gọi API `GET /runs`.

## 84. Nếu muốn biết export ONNX hiện tại dùng image size bao nhiêu

- Xem `localagent/models/model_manifest.json`
- Hiện tại là `160`.

## 85. Nếu muốn biết labels index của ONNX hiện tại

- Xem `localagent/models/labels.json`
- Hoặc `localagent/models/model_manifest.json`

## 86. Nếu muốn biết checkpoint nào đã được export

- Xem `checkpoint_path` trong `localagent/artifacts/reports/baseline-waste-sorter_export.json`

## 87. Nếu muốn biết benchmark có verify ONNX không

- Xem `onnx_verified` trong `baseline-waste-sorter_benchmark.json`
- Hiện tại là `true`.

## 88. Nếu muốn biết pseudo-label đã thêm bao nhiêu sample

- Xem `accepted_count` trong `baseline-waste-sorter_pseudo_label.json`
- Hiện tại là `570`.

## 89. Nếu muốn biết pseudo-label còn bỏ lại bao nhiêu sample

- Xem `rejected_count` trong `baseline-waste-sorter_pseudo_label.json`
- Hiện tại là `220`.

## 90. Nếu muốn biết artifact bundle gồm gì

- `training_plan`
- `dataset_summary`
- `pseudo_label`
- `training`
- `evaluation`
- `export`
- `benchmark`
- `experiment_spec`
- `model_manifest`

## 91. Vị trí code build bundle report

- `build_artifact_report` ở `localagent/python/localagent/training/trainer.py:1099`

## 92. Vì sao bundle report hữu ích

- Bạn chỉ cần đọc một file là thấy gần như toàn cảnh của experiment.

## 93. Test nào đảm bảo pseudo-label hoạt động đúng

- `test_pseudo_label_updates_manifest_with_confidence_gate`

## 94. Test nào đảm bảo export ONNX hoạt động đúng

- `test_export_onnx_writes_manifest_and_export_report`

## 95. Test nào đảm bảo bundle report hoạt động đúng

- `test_build_artifact_report_bundles_existing_training_outputs`

## 96. Test nào đảm bảo benchmark hoạt động đúng

- `test_benchmark_writes_report_for_pytorch_backend`
- `test_compare_benchmark_reports_returns_stage_and_metric_deltas`

## 97. Điều cần nhớ nhất của file này

- Artifact là nền tảng của mọi bước sau.
- Pseudo-label không phải auto-label vô điều kiện.
- ONNX export chỉ đáng tin khi verify đã pass.
- `model_manifest.json` là hợp đồng suy luận cực quan trọng.
- Mọi thứ đều được lưu cục bộ, không cần dịch vụ ngoài để theo dõi experiment.

## 98. Phụ lục A: ma trận file cục bộ quan trọng nhất trong repo

- `localagent/artifacts/manifests/dataset_manifest.parquet` là manifest chuẩn để pipeline và trainer đọc.
- `localagent/artifacts/manifests/dataset_manifest.csv` là bản dễ mở bằng mắt thường.
- `localagent/artifacts/manifests/dataset_embeddings.npz` là embedding artifact cho discovery.
- `localagent/artifacts/manifests/cluster_review.csv` là file review cluster cho con người.
- `localagent/artifacts/manifests/labeling_template.csv` là template gán nhãn thủ công.
- `localagent/artifacts/reports/summary.json` là ảnh chụp Step 1 tổng quát nhất.
- `localagent/artifacts/reports/split_summary.csv` là thống kê train/val/test/excluded.
- `localagent/artifacts/reports/quality_summary.csv` là thống kê ảnh lỗi, ảnh nhỏ, duplicate.
- `localagent/artifacts/reports/extension_summary.csv` là thống kê theo phần mở rộng file.
- `localagent/artifacts/reports/label_summary.csv` là thống kê nhãn theo nhiều góc nhìn.
- `localagent/artifacts/reports/embedding_summary.json` mô tả extractor và kích thước vector.
- `localagent/artifacts/reports/cluster_summary.json` mô tả số cluster và số outlier.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` là report chi tiết sau `fit`.
- `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json` là report đánh giá.
- `localagent/artifacts/reports/baseline-waste-sorter_confusion_matrix.csv` là ma trận nhầm lẫn.
- `localagent/artifacts/reports/baseline-waste-sorter_pseudo_label.json` là report pseudo-label.
- `localagent/artifacts/reports/baseline-waste-sorter_export.json` là report export ONNX.
- `localagent/artifacts/reports/baseline-waste-sorter_experiment_spec.json` là cấu hình run ở dạng artifact.
- `localagent/artifacts/reports/baseline-waste-sorter_benchmark.json` là report benchmark.
- `localagent/artifacts/reports/baseline-waste-sorter_report.json` là bundle report hợp nhất.
- `localagent/artifacts/reports/run_index.json` là chỉ mục để UI liệt kê các run.
- `localagent/artifacts/cache/training/160px-raw/` là cache ảnh cho run ảnh 160px.
- `localagent/artifacts/cache/training/224px-raw/` là cache ảnh cho run ảnh 224px.
- `localagent/artifacts/jobs/*.json` là nhật ký metadata job do Rust server tạo.
- `localagent/models/waste_classifier.onnx` là mô hình ONNX export.
- `localagent/models/labels.json` là mapping nhãn cho inference.
- `localagent/models/model_manifest.json` là mô tả hợp đồng inference.
- `localagent/checkpoints/` là nơi trainer lưu checkpoint trong khi train.
- Các đường dẫn mặc định chịu ảnh hưởng bởi `AgentPaths` ở `localagent/python/localagent/config/settings.py:13`.
- Nơi thật sự trainer dựng path artifact nằm ở nhiều helper cuối `trainer.py` từ khoảng `1891`.
- Nơi Rust server đọc artifact để phục vụ UI là `localagent/src/artifacts.rs`.
- Nơi Python ghi bundle report là `WasteTrainer.build_artifact_report()` tại `trainer.py:1099`.

## 99. Phụ lục B: anatomy của `summary.json`

- `summary.json` được sinh bởi `generate_reports()` ở `localagent/python/localagent/data/pipeline.py:158`.
- Field `dataset_root` cho biết thư mục dữ liệu đã được quét.
- Field này đặc biệt quan trọng khi repo có cả `dataset/` lẫn `datasets/`.
- Field `total_files` là tổng file ảnh được thấy trong raw dataset dir.
- Field `valid_files` là số file vượt qua bước decode và quality gate cơ bản.
- Field `invalid_files` là số file lỗi hoặc bị loại vì không decode được.
- Field `duplicate_files` là số file bị coi là trùng nội dung.
- Field `training_ready_files` là số ảnh cuối cùng đủ điều kiện đi vào training mode hiệu lực.
- Field `effective_training_mode` nói pipeline đang train trên accepted labels hay inferred labels.
- Field `split_counts` cho biết phân bố `train`, `val`, `test`, `excluded`.
- Field `label_counts` cho biết tổng nhãn hiệu lực, kể cả `unknown`.
- Field `trainable_label_counts` loại bỏ phần không train được.
- Field `label_source_counts` cho biết nhãn đến từ đâu.
- Field `accepted_label_source_counts` lọc riêng những nguồn được xem là hợp lệ cho training.
- Field `annotation_status_counts` cho biết `unlabeled`, `labeled`, `pseudo_labeled`.
- Field `review_status_counts` cho biết cluster accepted, pseudo accepted, pending review.
- Field `clustered_files` cho biết số ảnh đã có cluster id.
- Field `cluster_outlier_files` cho biết số outlier sau discovery.
- Field `embedding_artifact_exists` cho biết `.npz` đã có chưa.
- Field `cluster_summary_exists` cho biết report cluster đã có chưa.
- Field `width_stats` mô tả chiều rộng min, median, max.
- Field `height_stats` mô tả chiều cao min, median, max.
- Field `cluster_preview_total` cho biết số cluster có preview.
- Snapshot hiện tại cho thấy `total_files = 9999`.
- Snapshot hiện tại cho thấy `valid_files = 9989`.
- Snapshot hiện tại cho thấy `invalid_files = 10`.
- Snapshot hiện tại cho thấy `duplicate_files = 10`.
- Snapshot hiện tại cho thấy `training_ready_files = 9769`.
- Snapshot hiện tại cho thấy `effective_training_mode = accepted_labels_only`.
- Snapshot hiện tại cho thấy `split_counts.train = 7814`.
- Snapshot hiện tại cho thấy `split_counts.val = 975`.
- Snapshot hiện tại cho thấy `split_counts.test = 980`.
- Snapshot hiện tại cho thấy `split_counts.excluded = 230`.
- Snapshot hiện tại cho thấy `label_counts.glass = 8776`.
- Snapshot hiện tại cho thấy `label_counts.paper = 529`.
- Snapshot hiện tại cho thấy `label_counts.folk = 464`.
- Snapshot hiện tại cho thấy `label_counts.unknown = 230`.
- Snapshot hiện tại cho thấy `accepted_label_source_counts.cluster_review = 9199`.
- Snapshot hiện tại cho thấy `accepted_label_source_counts.model_pseudo = 570`.
- Snapshot hiện tại cho thấy `annotation_status_counts.labeled = 9199`.
- Snapshot hiện tại cho thấy `annotation_status_counts.pseudo_labeled = 570`.
- Snapshot hiện tại cho thấy `annotation_status_counts.unlabeled = 230`.
- Snapshot hiện tại cho thấy `review_status_counts.cluster_accepted = 9199`.
- Snapshot hiện tại cho thấy `review_status_counts.pseudo_accepted = 570`.
- Snapshot hiện tại cho thấy `review_status_counts.pseudo_rejected = 220`.
- Snapshot hiện tại cho thấy `review_status_counts.pending_review = 10`.
- Snapshot hiện tại cho thấy `cluster_outlier_files = 790`.
- Snapshot hiện tại cho thấy median width xấp xỉ `225`.
- Snapshot hiện tại cho thấy median height xấp xỉ `224`.
- Nếu bạn chỉ được mở một file report duy nhất để biết dataset đang ở đâu trong workflow, hãy mở `summary.json`.

## 100. Phụ lục C: anatomy của `baseline-waste-sorter_training.json`

- Report training được ghi bởi `fit()` ở `localagent/python/localagent/training/trainer.py:1270`.
- File này không chỉ chứa metric cuối.
- Nó còn chứa cấu hình run, lịch sử epoch và artifact liên quan.
- Field `experiment_name` là khóa logic liên kết nhiều artifact khác.
- Field `training_preset` cho biết preset khởi tạo config.
- Field `training_backend` cho biết backend fit đã dùng.
- Field `epochs_completed` cho biết trainer thực sự chạy bao nhiêu epoch.
- Field `best_epoch` cho biết epoch tốt nhất theo validation criterion.
- Field `best_loss` là validation loss tốt nhất.
- Field `stopped_early` cho biết early stopping có kích hoạt hay không.
- Field `stop_reason` giải thích tại sao run dừng.
- Field `device` cho biết run dùng CPU hay accelerator.
- Field `model.model_name` cho biết backbone CNN nào đã được huấn luyện.
- Field `image_size` cho biết kích thước đầu vào đã dùng.
- Field `batch_size` cho biết kích thước batch.
- Field `cache_summary` cho biết cache có được warm, reuse hay fail hay không.
- Field `class_weight_map` cho biết trọng số từng lớp trong loss.
- Field `history` là danh sách metric theo epoch.
- Field `train_accuracy` trong từng epoch giúp nhìn tốc độ khớp train.
- Field `val_accuracy` trong từng epoch giúp theo dõi tổng quát nhưng không thay thế loss.
- Field `evaluation_summary` là bản tóm tắt của report evaluation.
- Field `test_accuracy` là accuracy tập test.
- Snapshot hiện tại dùng preset `cpu_fast`.
- Snapshot hiện tại dùng backend `pytorch`.
- Snapshot hiện tại dùng model `mobilenet_v3_small`.
- Snapshot hiện tại dùng `image_size = 160`.
- Snapshot hiện tại dùng `batch_size = 32`.
- Snapshot hiện tại dừng sau `epochs_completed = 6`.
- Snapshot hiện tại có `best_epoch = 3`.
- Snapshot hiện tại có `best_loss ≈ 0.1913`.
- Snapshot hiện tại có `stopped_early = true`.
- Snapshot hiện tại có lý do dừng là không cải thiện quá `0.001` trong `3` epoch.
- Snapshot hiện tại có `device = cpu`.
- Snapshot hiện tại có `test_accuracy ≈ 0.9296`.
- Snapshot hiện tại có `macro_f1 ≈ 0.8074`.
- Snapshot hiện tại có `weighted_f1 ≈ 0.9366`.
- Snapshot hiện tại có class weight lớn cho `folk` và `paper`.
- Điều này phản ánh dataset lệch mạnh về `glass`.
- Nếu bạn thấy `train_accuracy` tăng nhưng `val_loss` xấu, hãy nghi overfitting.
- Nếu bạn thấy `cache_summary` bỏ qua toàn bộ vì cache đã có, đó là tín hiệu tốt về tốc độ.
- Nếu bạn thấy `class_weight_map` rỗng trong dataset lệch, hãy kiểm tra `class_bias`.

## 101. Phụ lục D: anatomy của `baseline-waste-sorter_evaluation.json`

- Report evaluation được ghi bởi `evaluate()` ở `trainer.py:869`.
- Field `labels` là thứ tự nhãn dùng cho confusion matrix.
- Field `num_samples` là số mẫu tập đánh giá.
- Field `accuracy` là tỷ lệ dự đoán đúng toàn cục.
- Field `macro_precision` là precision trung bình không trọng số.
- Field `macro_recall` là recall trung bình không trọng số.
- Field `macro_f1` là F1 trung bình không trọng số.
- Field `weighted_precision` là precision trung bình có trọng số support.
- Field `weighted_recall` là recall trung bình có trọng số support.
- Field `weighted_f1` là F1 trung bình có trọng số.
- Field `per_class` chứa metric chi tiết từng lớp.
- Mỗi mục `per_class` có `precision`.
- Mỗi mục `per_class` có `recall`.
- Mỗi mục `per_class` có `f1`.
- Mỗi mục `per_class` có `support`.
- Field `confusion_matrix` là ma trận đếm theo thứ tự `labels`.
- Snapshot hiện tại có `num_samples = 980`.
- Snapshot hiện tại có accuracy khoảng `0.9296`.
- Snapshot hiện tại có macro F1 khoảng `0.8074`.
- Snapshot hiện tại cho lớp `glass` precision rất cao.
- Snapshot hiện tại cho lớp `paper` precision thấp hơn đáng kể.
- Snapshot hiện tại cho lớp `folk` recall khá cao nhưng precision chưa lý tưởng.
- Điều đó nói rằng model nhận diện lớp hiếm khá bắt được nhưng còn false positive.
- Muốn cải thiện cần xem thêm confusion matrix CSV.
- Confusion matrix CSV được ghi song song ở `baseline-waste-sorter_confusion_matrix.csv`.
- Nếu bạn trình bày kết quả cho người không chuyên, đừng chỉ đưa accuracy.
- Hãy giải thích thêm macro F1 vì dataset lệch lớp.

## 102. Phụ lục E: anatomy của `baseline-waste-sorter_pseudo_label.json`

- Report pseudo-label được ghi bởi `pseudo_label()` ở `trainer.py:1126`.
- Field `candidate_count` là số mẫu được xem xét pseudo-label.
- Field `accepted_count` là số mẫu vượt cả confidence lẫn margin threshold.
- Field `rejected_count` là số mẫu bị từ chối.
- Field `confidence_threshold` là ngưỡng top-1 probability tối thiểu.
- Field `margin_threshold` là ngưỡng chênh top1-top2 tối thiểu.
- Field `average_accepted_score` cho biết độ tự tin trung bình của các mẫu được nhận.
- Snapshot hiện tại có `candidate_count = 790`.
- Snapshot hiện tại có `accepted_count = 570`.
- Snapshot hiện tại có `rejected_count = 220`.
- Snapshot hiện tại có `confidence_threshold = 0.95`.
- Snapshot hiện tại có `margin_threshold = 0.25`.
- Snapshot hiện tại có `average_accepted_score ≈ 0.9944`.
- Đây là mức threshold chặt.
- Mức threshold chặt phù hợp khi pseudo-label chỉ nên bổ sung phần model rất chắc chắn.
- Mẫu bị từ chối không phải vô dụng.
- Chúng nên đi tiếp sang human review hoặc vòng train sau.
- Logic accept nằm ở `trainer.py:1197`.
- Ở đó điều kiện là `score >= threshold` và `margin >= margin_threshold`.
- Vì vậy một mẫu confidence cao nhưng top2 quá sát vẫn bị reject.

## 103. Phụ lục F: anatomy của `baseline-waste-sorter_export.json` và `model_manifest.json`

- Report export được ghi bởi `export_onnx()` ở `trainer.py:959`.
- `baseline-waste-sorter_export.json` mô tả quá trình tạo file ONNX.
- Field `onnx_path` chỉ file `.onnx` vừa xuất.
- Field `onnx_opset` ghi phiên bản opset.
- Field `input_spec` cho biết batch, channels, height, width khi export.
- Field `verification` cho biết verify có pass không.
- Nếu `verify_onnx = true`, code dùng `onnxruntime` để so logits ONNX và PyTorch.
- Logic verify nằm ở `trainer.py:1029` đến `trainer.py:1042`.
- Field `max_abs_diff` là chênh lệch tuyệt đối lớn nhất giữa hai nhánh.
- Snapshot hiện tại có `verified = true`.
- Snapshot hiện tại dùng `CPUExecutionProvider`.
- Snapshot hiện tại có `max_abs_diff ≈ 1.55e-06`.
- Điều này là đủ tốt để tin file ONNX gần như tương đương PyTorch cho case kiểm tra đó.
- `model_manifest.json` được dựng bởi `_build_model_manifest()` ở `trainer.py:1831`.
- File này cực kỳ quan trọng cho inference.
- Nó chứa `labels`.
- Nó chứa `image_size`.
- Nó chứa normalization preset.
- Nó chứa thông tin ONNX verify.
- Nó chứa tóm tắt metric evaluation.
- Rust inference side dùng file này để biết phải preprocess ảnh thế nào.
- Logic inference nằm ở `localagent/src/inference.rs`.
- `classify_uploaded_image()` trong `inference.rs` dựa vào `model_manifest.json` và `waste_classifier.onnx`.
- Nếu `labels.json` và `model_manifest.json` lệch nhau, hệ thống inference sẽ rất rủi ro.
- Vì vậy export ONNX nên được xem như một gói gồm ít nhất ba file:
- `waste_classifier.onnx`.
- `labels.json`.
- `model_manifest.json`.

## 104. Phụ lục G: lifecycle của cache, checkpoint và job log

- Warm cache được kích hoạt bởi `warm_image_cache()` ở `trainer.py:1569`.
- Cache nằm dưới `localagent/artifacts/cache/training/`.
- Tên thư mục cache thường phản ánh kích thước ảnh và format.
- Ví dụ `160px-raw`.
- Ví dụ `224px-raw`.
- Nếu đổi `image_size`, cache cũ không tái sử dụng trực tiếp cho run mới.
- Nếu đổi `cache_format`, cache cũ cũng không tái sử dụng trực tiếp.
- Nếu `force_cache = true`, trainer sẽ tạo lại cache dù cache cũ đã có.
- Checkpoint được sinh trong quá trình `fit`.
- Checkpoint latest hữu ích cho resume sau lỗi hoặc dừng giữa chừng.
- Resume logic nằm ở `_load_resume_state()` tại `trainer.py:505`.
- Nếu bạn hủy job từ UI, Rust server sẽ đánh `cancel_requested` trong `JobRecord`.
- Metadata job nằm ở `localagent/artifacts/jobs/*.json`.
- Log stdout và stderr của job được lưu thành file riêng.
- `JobRecord.stdout_log_path` và `stderr_log_path` chỉ tới hai file đó.
- UI lấy snapshot log qua `GET /jobs/{job_id}/logs`.
- WebSocket chỉ stream log mới.
- Job log không thay thế report JSON.
- Report JSON là dữ liệu cấu trúc để UI tổng hợp.
- Job log là dữ liệu chuỗi để debug.

## 105. Phụ lục H: dọn dẹp, backup và cách không tự bắn vào chân

- Không xóa `dataset_manifest.parquet` khi chưa chắc mình còn raw dataset ổn định.
- Không xóa `model_manifest.json` rồi giữ mỗi `.onnx`.
- Không xóa `run_index.json` nếu UI đang cần liệt kê run.
- Không xóa cache ngay trước benchmark nếu bạn muốn số đo phản ánh run warm-cache.
- Nếu cần backup tối thiểu cho một model deploy được, giữ:
- `waste_classifier.onnx`.
- `labels.json`.
- `model_manifest.json`.
- Nếu cần backup một run để tái hiện, giữ thêm:
- `*_experiment_spec.json`.
- `*_training.json`.
- `*_evaluation.json`.
- `*_export.json`.
- `*_report.json`.
- Nếu cần backup cả vòng dữ liệu, giữ thêm manifest và cluster review.
- Trước khi dọn cache, kiểm tra có run nào sắp resume không.
- Trước khi ghi đè experiment name cũ, quyết định rõ bạn muốn tiếp tục hay tạo run mới.
- Nếu nghi report bị lỗi, tái tạo report bằng command `report` thay vì sửa tay JSON.

## 106. Phụ lục I: map artifact với code và test

- `generate_reports()` ở `pipeline.py:158` ghi hầu hết report Step 1.
- `extract_embeddings()` ở `discovery.py:72` tạo dữ liệu cho `dataset_embeddings.npz`.
- `cluster_embeddings()` ở `discovery.py:112` tạo dữ liệu cho `cluster_summary.json`.
- `fit()` ở `trainer.py:1270` ghi training report.
- `evaluate()` ở `trainer.py:869` ghi evaluation JSON và confusion matrix CSV.
- `export_onnx()` ở `trainer.py:959` ghi export report, labels và model manifest.
- `pseudo_label()` ở `trainer.py:1126` ghi pseudo-label report.
- `build_artifact_report()` ở `trainer.py:1099` tạo report bundle hợp nhất.
- `ArtifactStore` ở `localagent/src/artifacts.rs` là lớp Rust đọc toàn bộ artifact này.
- `localagent/tests/test_training_artifacts.py` là test đầu tiên nên đọc nếu bạn sửa format report.

## 107. Phụ lục J: checklist giữ artifact cho từng mục tiêu

- Nếu mục tiêu là debug Step 1, giữ manifest và `summary.json`.
- Nếu mục tiêu là debug discovery, giữ thêm `dataset_embeddings.npz`, `embedding_summary.json`, `cluster_summary.json`, `cluster_review.csv`.
- Nếu mục tiêu là debug training, giữ thêm `*_training.json`, `*_evaluation.json`, `*_confusion_matrix.csv`, checkpoint.
- Nếu mục tiêu là debug pseudo-label, giữ thêm `*_pseudo_label.json` và manifest sau cập nhật.
- Nếu mục tiêu là deploy inference, giữ `waste_classifier.onnx`, `labels.json`, `model_manifest.json`.
- Nếu mục tiêu là so sánh run, giữ `run_index.json`, `*_benchmark.json`, `*_report.json`.
- Nếu mục tiêu là sửa UI dashboard, giữ các report JSON mà `ArtifactStore` đang đọc.
- Nếu mục tiêu là sửa server inference, giữ bộ ba ONNX, labels, model manifest.
- Nếu mục tiêu là forensic sau job fail, giữ `artifacts/jobs/*.json`, stdout log, stderr log.
- Nếu mục tiêu là tái lập cấu hình huấn luyện, giữ `*_experiment_spec.json`.
- Nếu mục tiêu là bàn giao cho người khác, đừng chỉ đưa mỗi `.onnx`.
- Nếu mục tiêu là backup tối thiểu, hãy chép cả thư mục `models/` và report run tương ứng.
- Nếu mục tiêu là dọn đĩa, xóa cache cẩn thận hơn xóa manifest.
- Nếu mục tiêu là rerun training, xóa cache không sai nhưng làm tốn thời gian hơn.
- Nếu mục tiêu là chứng minh một metric, artifact chính phải là evaluation JSON và confusion matrix CSV.

## 108. Phụ lục K: câu hỏi tự kiểm tra khi một artifact trông “lạ”

- Artifact này do lệnh nào sinh ra.
- Artifact này được sinh ở hàm nào.
- Artifact này được server đọc ở đâu.
- Artifact này được UI hiển thị ở đâu.
- Artifact này có tên experiment đúng chưa.
- Artifact này có timestamp hay nội dung mới chưa.
- Artifact này có thể là file cũ còn sót lại không.
- Artifact này có phụ thuộc manifest mới nhất không.
- Artifact này có đang bị ghi đè bởi run khác không.
- Artifact này có cần file đi kèm nào nữa không.
- Nếu artifact là ONNX, verify report có pass không.
- Nếu artifact là report training, class weight map có hợp lý không.
- Nếu artifact là cluster review, fingerprint có còn current không.
- Nếu artifact là job record, stdout và stderr path có tồn tại không.
- Nếu trả lời được hết các câu trên, bạn sẽ ít khi debug sai hướng.

## 109. Phụ lục L: danh sách file nên mở theo thứ tự khi điều tra artifact

- Mở `summary.json` trước.
- Mở `dataset_manifest.csv` sau.
- Mở `embedding_summary.json` nếu đang ở Step 2.
- Mở `cluster_summary.json` nếu đang ở Step 2.
- Mở `cluster_review.csv` nếu đang ở khâu review.
- Mở `*_training.json` nếu đang ở Step 3.
- Mở `*_evaluation.json` và `*_confusion_matrix.csv` nếu đang xem chất lượng.
- Mở `*_pseudo_label.json` nếu đang xem mở rộng nhãn.
- Mở `*_export.json` nếu đang xem ONNX.
- Mở `model_manifest.json` nếu đang xem inference.
- Mở `run_index.json` nếu UI không hiển thị run mới.

## 110. Phụ lục M: bộ file tối thiểu cho từng loại bàn giao

- Bàn giao để infer cục bộ: `waste_classifier.onnx`, `labels.json`, `model_manifest.json`.
- Bàn giao để chứng minh chất lượng: thêm `*_evaluation.json` và `*_confusion_matrix.csv`.
- Bàn giao để tái hiện training: thêm `*_experiment_spec.json`, `*_training.json`, checkpoint.
- Bàn giao để tiếp tục pipeline dữ liệu: thêm manifest và `cluster_review.csv`.
- Bàn giao để debug job: thêm `artifacts/jobs/*.json`, stdout log, stderr log.
- Bàn giao để so sánh benchmark: thêm `*_benchmark.json` và `run_index.json`.
- Nếu thiếu một trong các file này, người nhận thường phải đoán hoặc tái chạy không cần thiết.

## 111. Phụ lục N: chốt nhanh file artifact nào đọc trước trong từng tình huống

- Nghi Step 1 sai: mở `summary.json`.
- Nghi cluster sai: mở `cluster_summary.json`.
- Nghi pseudo-label sai: mở `*_pseudo_label.json`.
- Nghi metric sai: mở `*_evaluation.json`.
- Nghi export sai: mở `*_export.json`.
- Nghi inference sai: mở `model_manifest.json`.
- Nghi UI không thấy run: mở `run_index.json`.

## 112. Phụ lục O: câu hỏi chốt trước khi xóa một artifact

- Artifact này có còn được route hay UI đọc không.
- Artifact này có thể tái tạo dễ dàng không.
- Artifact này có đang là bằng chứng của một benchmark hoặc metric không.
- Artifact này có gắn với experiment duy nhất nào không.
- Artifact này có file phụ thuộc đi kèm không.
- Artifact này có đang là đầu vào cho bước kế tiếp không.
- Nếu chưa chắc, đừng xóa.

## 113. Dòng chốt

- Artifact tốt là artifact biết rõ nguồn sinh, nơi đọc và mục đích tồn tại.
- Artifact thiếu ngữ cảnh là nguồn gây debug sai hướng.
- Artifact đúng cặp với run mới là nền tảng của reproducibility.

## 114. Chốt phụ cuối

- Luôn giữ cặp `model_manifest.json` và `waste_classifier.onnx`.
- Luôn kiểm tra report đi kèm trước khi chia sẻ artifact.
- Luôn ưu tiên artifact mới nhất nhưng phải đúng experiment name.
