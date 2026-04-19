# Tài liệu 03: Huấn luyện CNN và cách chọn cấu hình theo máy

## 1. File này nói về cái gì

- File này nói riêng về Step 3 là training.
- File này không nói về discovery chi tiết.
- File này tập trung vào cấu hình training, chọn backbone, chọn preset, chọn cache format, chọn batch size, chọn image size, chọn device và hiểu các flag CLI.
- File này cũng giải thích vì sao một số cấu hình hợp với CPU yếu còn cấu hình khác chỉ hợp với máy mạnh hơn.

## 2. Vị trí code chính của training

- `localagent/python/localagent/training/train.py`
- `localagent/python/localagent/training/trainer.py`
- `localagent/python/localagent/training/manifest_dataset.py`
- `localagent/python/localagent/vision/transforms.py`
- `localagent/python/localagent/training/benchmarking.py`

## 3. Entry point của training CLI

- `build_parser` ở `localagent/python/localagent/training/train.py:56`
- `build_config` ở `localagent/python/localagent/training/train.py:113`
- `main` ở `localagent/python/localagent/training/train.py:219`

## 4. Điều kiện để training được mở khóa

- Step 2 phải hoàn tất.
- Tức là dataset summary phải cho thấy có accepted labels từ `cluster_review`.
- Logic khóa này nằm ở `localagent/python/localagent/training/train.py:213`.
- Logic khóa tương tự cũng nằm ở `localagent/src/jobs/commands.rs`.

## 5. Vì sao training bị khóa khi chưa có accepted labels

- Vì dự án muốn tránh train nghiêm túc chỉ từ weak label filename khi đã bước sang workflow chuẩn.
- Điều này giúp pipeline nhất quán với mục tiêu bán giám sát có người xác nhận.

## 6. Class trung tâm của training là gì

- Là `WasteTrainer`.
- Nó nằm ở `localagent/python/localagent/training/trainer.py:40`.

## 7. `WasteTrainer` làm những gì

- Tóm tắt kế hoạch training.
- Dựng model.
- Dựng dataset và dataloader.
- Tính class weight.
- Warm cache ảnh bằng Rust.
- Train model.
- Resume checkpoint.
- Evaluate.
- Export ONNX.
- Build artifact bundle.
- Benchmark.
- Pseudo-label.

## 8. Backbone CNN đang được hỗ trợ

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `resnet18`
- `efficientnet_b0`

## 9. Vị trí code định nghĩa backbone hỗ trợ

- `SUPPORTED_CNN_MODELS` ở `localagent/python/localagent/training/trainer.py`.

## 10. `build_model_stub` làm gì

- Chọn factory model tương ứng với `model_name`.
- Nếu `pretrained_backbone=True`, cố load weight pretrained.
- Nếu load pretrained thất bại, fallback sang random init.
- Thay classifier head bằng số lớp mới.
- Freeze backbone nếu `freeze_backbone=True`.

## 11. Vị trí code của `build_model_stub`

- `localagent/python/localagent/training/trainer.py:128`

## 12. Head được thay thế thế nào

- Với `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`, code thay `model.classifier[-1]`.
- Với `resnet18`, code thay `model.fc`.
- Vị trí code là `_replace_classifier_head` tại `localagent/python/localagent/training/trainer.py:192`.

## 13. Freeze backbone được thực hiện thế nào

- Với MobileNet và EfficientNet, code set `requires_grad=False` cho `model.features.parameters()`.
- Với ResNet18, code freeze mọi parameter không bắt đầu bằng `fc.`.
- Vị trí code là `_freeze_model_backbone` tại `localagent/python/localagent/training/trainer.py:208`.

## 14. Điều này có ý nghĩa thực tế gì

- Freeze backbone giảm số tham số cần tối ưu.
- Freeze backbone giảm chi phí tính gradient.
- Freeze backbone hợp với CPU.
- Freeze backbone đặc biệt hợp khi dữ liệu đã đủ lớn và backbone pretrained đã đủ mạnh.

## 15. Khi nào nên bật `--train-backbone`

- Khi bạn có GPU.
- Khi dữ liệu khác nhiều so với miền ảnh tự nhiên của ImageNet.
- Khi bạn chấp nhận thời gian train lâu hơn.
- Khi accuracy đang chững vì head-only fine-tuning không đủ.

## 16. Khi nào không nên bật `--train-backbone`

- Khi chỉ có CPU.
- Khi RAM ít.
- Khi muốn benchmark nhanh.
- Khi mới cần baseline trước.

## 17. Dữ liệu training được lấy từ đâu

- Từ manifest parquet.
- Đường dẫn mặc định là `localagent/artifacts/manifests/dataset_manifest.parquet`.
- Cấu hình nằm trong `TrainingConfig.manifest_path`.

## 18. `load_manifest` của trainer làm gì

- Đọc parquet.
- Chuẩn hóa cột nếu manifest cũ.
- Trả về `polars.DataFrame`.

## 19. `class_names` của trainer lấy gì

- Chỉ lấy các label đang thực sự train được.
- Không lấy `unknown`.
- Nếu đã có accepted labels, chỉ lấy sample từ nguồn accepted.

## 20. Vì sao `class_names` rất quan trọng

- Thứ tự label quyết định thứ tự output của classifier.
- Thứ tự này cũng được ghi vào `labels.json`.
- Nếu mismatch giữa checkpoint và manifest hiện tại, trainer sẽ báo lỗi khi resume/evaluate/export.

## 21. `build_datasets` làm gì

- Dựng `ManifestImageDataset` cho `train`, `val`, `test`.
- Truyền `label_to_index`.
- Truyền `image_size`, `cache_dir`, `cache_format`, `normalization_preset`.

## 22. `ManifestImageDataset` lọc sample thế nào

- `is_valid` phải đúng.
- `label` không được là `unknown`.
- `annotation_status` không được là `unlabeled` hoặc `excluded`.
- `split` phải đúng với dataset đang dựng.

## 23. Vị trí code của `ManifestImageDataset`

- `localagent/python/localagent/training/manifest_dataset.py:11`

## 24. `ManifestImageDataset` đọc ảnh thế nào

- Nếu có cache tương ứng trong `cache_dir`, dùng file cache.
- Nếu không, đọc ảnh gốc từ `image_path`.
- Nếu file cache có đuôi `.raw`, phải biết `raw_image_size`.
- Sau đó áp transform.

## 25. Vì sao có cả cache `.png` và `.raw`

- `.png` tiện đọc hơn, dễ inspect.
- `.raw` tránh chi phí encode/decode PNG.
- `.raw` thường tốt hơn cho run dài và dataset lớn.
- Đổi lại `.raw` cần biết trước kích thước resize chuẩn.

## 26. Vị trí code quyết định đuôi cache

- `ManifestImageDataset.__init__` ở `localagent/python/localagent/training/manifest_dataset.py`
- `WasteTrainer._cache_file_suffix` ở `localagent/python/localagent/training/trainer.py:2045`

## 27. `build_dataloaders` làm gì

- Dựng `torch.utils.data.DataLoader`.
- Tạo sampler cho train nếu dùng class bias kiểu sampler.
- Bật `pin_memory` nếu device là CUDA.
- Bật `persistent_workers` nếu `num_workers > 0`.
- Thêm `prefetch_factor` nếu có worker.

## 28. Vị trí code của `build_dataloaders`

- `localagent/python/localagent/training/trainer.py:312`

## 29. Class imbalance được xử lý thế nào

- Có thể không xử lý.
- Có thể xử lý bằng weight ở loss.
- Có thể xử lý bằng weighted sampler.
- Có thể dùng cả hai.

## 30. Giá trị hợp lệ của `class_bias_strategy`

- `none`
- `loss`
- `sampler`
- `both`

## 31. Vị trí code xây sampler

- `_build_train_sampler` ở `localagent/python/localagent/training/trainer.py:2048`

## 32. Vị trí code xây class weights cho loss

- `_loss_class_weights` ở `localagent/python/localagent/training/trainer.py:2069`

## 33. Công thức class weight trong code fallback Python

- Nếu có `K` lớp hiện diện.
- Nếu tổng số sample train là `N`.
- Nếu lớp `c` có `n_c` sample.
- Weight của lớp `c` là `N / (K * n_c)`.

## 34. Ý nghĩa của công thức trên

- Lớp hiếm hơn sẽ có trọng số lớn hơn.
- Lớp phổ biến hơn sẽ có trọng số nhỏ hơn.
- Đây là cách cân bằng đơn giản nhưng hiệu quả cho mất cân bằng vừa và nặng.

## 35. Repo hiện tại mất cân bằng lớp đến mức nào

- `train_imbalance_ratio` gần nhất khoảng `18.92`.
- Điều này rất nặng.
- Vì vậy preset hiện tại dùng `class_bias = loss`.

## 36. Các preset training hiện có

- `cpu_fast`
- `cpu_balanced`
- `cpu_stronger`

## 37. Vị trí code định nghĩa preset

- `TRAINING_PRESETS` ở `localagent/python/localagent/training/train.py`
- Server Rust cũng trả preset tương ứng qua `GET /presets/training`.

## 38. Preset `cpu_fast`

- `model_name = mobilenet_v3_small`
- `image_size = 160`
- `batch_size = 32`
- `cache_format = raw`
- `class_bias_strategy = loss`

## 39. Khi nào nên dùng `cpu_fast`

- Máy chỉ có CPU.
- RAM vừa phải.
- Muốn chạy baseline nhanh.
- Muốn train nhanh để kiểm workflow.

## 40. Preset `cpu_balanced`

- `model_name = resnet18`
- `image_size = 224`
- `batch_size = 16`
- `cache_format = raw`
- `class_bias_strategy = loss`

## 41. Khi nào nên dùng `cpu_balanced`

- Máy CPU đủ khỏe.
- Muốn cân bằng giữa tốc độ và chất lượng.
- Muốn backbone dễ hiểu và dễ debug.

## 42. Preset `cpu_stronger`

- `model_name = efficientnet_b0`
- `image_size = 224`
- `batch_size = 8`
- `cache_format = raw`
- `class_bias_strategy = loss`

## 43. Khi nào nên dùng `cpu_stronger`

- Máy CPU khỏe hoặc có nhiều thời gian.
- Muốn chất lượng tốt hơn baseline nhanh.
- Chấp nhận batch nhỏ và train lâu hơn.

## 44. Tại sao `cache_format = raw` được ưu tiên trong preset CPU

- Vì ảnh đã resize trước giúp giảm chi phí I/O.
- Vì `.raw` tránh decode PNG ở mỗi epoch.
- Vì bài toán hiện tại có gần 10 nghìn ảnh training-ready.

## 45. Chọn `image_size` thế nào theo máy

- Máy yếu: `160`.
- Máy trung bình: `224`.
- Máy mạnh hơn và có GPU: có thể thử lớn hơn, nhưng code preset hiện tại dừng ở `224`.
- Nên giữ nhất quán giữa train và export.

## 46. Chọn `batch_size` thế nào theo máy

- CPU yếu: `8` tới `16`.
- CPU trung bình: `16` tới `32`.
- CPU mạnh hoặc GPU: tăng dần cho tới khi hệ thống bắt đầu swap RAM hoặc OOM.
- Luôn benchmark bằng run ngắn trước.

## 47. Chọn `num_workers` thế nào theo máy

- Trên Windows mặc định là `0`.
- Đây là lựa chọn an toàn.
- Nếu bạn hiểu rõ multiprocessing của PyTorch trên môi trường mình, có thể tăng.
- Nhưng với Windows và CLI đơn giản, `0` thường đỡ lỗi nhất.

## 48. Vị trí code của default `num_workers`

- `TrainingConfig.num_workers` trong `localagent/python/localagent/config/settings.py`

## 49. Chọn `epochs` thế nào

- Chạy thử workflow: `3` tới `10`.
- Baseline thực dụng: `10` tới `25`.
- Fine-tune lâu: `25+`.
- Nếu bật early stopping, có thể đặt cao hơn rồi để trainer tự dừng.

## 50. Early stopping trong repo hoạt động thế nào

- Theo dõi `val_loss` nếu có validation.
- Nếu không có validation, theo dõi `train_loss`.
- Nếu loss không cải thiện quá `min_delta` trong `patience` epoch, trainer dừng.

## 51. Vị trí code của early stopping

- `_should_stop_early` ở `localagent/python/localagent/training/trainer.py:2243`
- Logic update `epochs_without_improvement` nằm trong `fit`.

## 52. Khi nào nên giảm `early_stopping_patience`

- Khi dataset lớn.
- Khi validation loss ổn định.
- Khi bạn cần baseline nhanh.

## 53. Khi nào nên tăng `early_stopping_patience`

- Khi loss dao động nhiều.
- Khi data nhỏ.
- Khi model cần nhiều epoch để ổn định.

## 54. `warm_image_cache` làm gì

- Chuẩn bị ảnh resize sẵn vào `artifacts/cache/training`.
- Gọi Rust bridge nếu khả dụng.
- Nếu Rust cache lỗi ở một số ảnh, trainer thử cứu bằng OpenCV.

## 55. Vị trí code của `warm_image_cache`

- `localagent/python/localagent/training/trainer.py:1569`

## 56. Vị trí code của bridge Rust cache

- `RustAccelerationBridge.prepare_image_cache` ở `localagent/python/localagent/bridge/rust_acceleration.py`
- `prepare_image_cache` binding ở `localagent/src/python_api.rs:84`

## 57. Vì sao cache ảnh rất quan trọng ở repo này

- Dữ liệu gần 10 nghìn ảnh.
- Nhiều epoch sẽ phải decode và resize lặp lại.
- Cache giảm đáng kể chi phí này.

## 58. Report cache lỗi nằm ở đâu

- `artifacts/reports/training_cache_failures_<image_size>px.json`
- Nếu cache format là raw, tên có thêm `_raw`.

## 59. Repo hiện tại có cache nào sẵn

- `localagent/artifacts/cache/training/160px-raw`
- `localagent/artifacts/cache/training/224px-raw`

## 60. Điều này gợi ý gì

- Repo đã chạy ít nhất hai cấu hình image size.
- Cụ thể là `160` và `224`.
- Đó là dấu hiệu cho thấy preset `cpu_fast` và `cpu_balanced` hoặc `cpu_stronger` đã được thử.

## 61. `fit` làm gì theo thứ tự

- Đảm bảo backend training được hỗ trợ.
- Load manifest.
- Warm cache.
- Build dataloader.
- Kiểm tra split train có dữ liệu.
- Kiểm tra số lớp hợp lệ.
- Build model.
- Build criterion và optimizer.
- Nếu có checkpoint resume thì load state.
- Chạy epoch train.
- Chạy epoch val nếu có.
- Cập nhật best model.
- Ghi checkpoint latest mỗi epoch.
- Ghi checkpoint best khi có best mới.
- In summary theo epoch.
- Dừng sớm nếu cần.
- Sau cùng export label index.
- Nếu có split test thì đánh giá best model lên test.
- Ghi training report JSON.

## 62. Vị trí code của `fit`

- `localagent/python/localagent/training/trainer.py:1270`

## 63. Criterion đang dùng là gì

- `torch.nn.CrossEntropyLoss`

## 64. Optimizer đang dùng là gì

- `torch.optim.Adam`

## 65. Learning rate mặc định là bao nhiêu

- `1e-3`

## 66. Weight decay mặc định là bao nhiêu

- `1e-4`

## 67. Vị trí code của default learning rate và weight decay

- `localagent/python/localagent/config/settings.py`
- Trong `TrainingConfig.learning_rate`
- Trong `TrainingConfig.weight_decay`

## 68. `resume_from_checkpoint` hoạt động thế nào

- Trainer load model state.
- Trainer load optimizer state nếu có.
- Trainer load history.
- Trainer tiếp tục từ epoch kế tiếp.
- Nếu `epochs` yêu cầu không lớn hơn epoch đã xong, trainer báo lỗi.

## 69. Vị trí code resume

- `_load_resume_state` ở `localagent/python/localagent/training/trainer.py:505`

## 70. Hai loại checkpoint của repo

- Best checkpoint.
- Latest checkpoint.

## 71. Vì sao cần best checkpoint

- Để export và evaluate từ model tốt nhất, không phải model cuối cùng.

## 72. Vì sao cần latest checkpoint

- Để resume training nếu bị dừng giữa chừng.

## 73. `show_progress=False` có nghĩa gì

- Tắt progress bar dạng thanh.
- Nhưng trainer vẫn in snapshot text theo batch và epoch.
- Điều này hữu ích cho log file và CI.

## 74. Vị trí code in snapshot batch

- `_print_batch_snapshot` ở `localagent/python/localagent/training/trainer.py:2226`

## 75. Vị trí code in summary epoch

- `_print_epoch_summary` ở `localagent/python/localagent/training/trainer.py:2180`

## 76. `evaluate` làm gì

- Load checkpoint.
- Build model theo config lưu trong checkpoint nếu cần override.
- Warm cache.
- Build dataloaders.
- Ưu tiên split `test`, nếu không có thì dùng `val`.
- Chạy một epoch đánh giá.
- Tính classification report và confusion matrix.
- Ghi report JSON và confusion matrix CSV.

## 77. Vị trí code của `evaluate`

- `localagent/python/localagent/training/trainer.py:869`

## 78. `export_onnx` làm gì

- Load checkpoint.
- Build model theo config lưu trong checkpoint.
- Tạo dummy input.
- Export ONNX.
- Chạy `onnx.checker`.
- Nếu bật verify, so sánh output ONNX và PyTorch.
- Ghi export report JSON.
- Ghi `model_manifest.json`.

## 79. Vị trí code của `export_onnx`

- `localagent/python/localagent/training/trainer.py:959`

## 80. `benchmark` làm gì

- Chạy `fit`.
- Chạy `evaluate`.
- Chạy `export_onnx`.
- Chạy `build_artifact_report`.
- Đo thời gian từng stage.
- Ghi benchmark report.

## 81. Vị trí code của `benchmark`

- `localagent/python/localagent/training/trainer.py:385`

## 82. Vì sao benchmark rất hữu ích

- Nó ép toàn bộ workflow training chạy end to end.
- Nó cho bạn artifact đồng nhất.
- Nó cho bạn số đo thời gian rõ ràng.

## 83. `experiment_spec` là gì

- Là snapshot cấu hình run.
- Nó không chứa trọng số model.
- Nó chứa thông số để tái hiện run ở mức cấu hình.

## 84. Vị trí code của `ExperimentSpec`

- `localagent/python/localagent/training/benchmarking.py:18`

## 85. `experiment_spec` hiện tại của repo cho biết gì

- Backend hiện tại là `pytorch`.
- Preset gần nhất là `cpu_fast`.
- Model gần nhất là `mobilenet_v3_small`.
- `image_size = 160`.
- `batch_size = 32`.
- `epochs = 10`.
- `cache_format = raw`.
- `class_bias_strategy = loss`.
- Early stopping đang bật.

## 86. `training_report.json` hiện tại của repo cho biết gì

- Best epoch là `3`.
- Best loss khoảng `0.1913`.
- Epoch hoàn tất là `6`.
- Training dừng sớm.
- Device là `cpu`.
- Class weight map rất lệch giữa `glass` và hai lớp còn lại.

## 87. Ý nghĩa của best epoch bằng `3`

- Sau epoch 3, validation loss không cải thiện đáng kể.
- Early stopping ngăn model train quá lâu trên CPU.

## 88. Mẫu chọn cấu hình theo máy yếu

- `training_preset = cpu_fast`
- `epochs = 10`
- `device = auto`
- `class_bias = loss`
- `no_progress = true`

## 89. Mẫu chọn cấu hình theo máy CPU khá

- `training_preset = cpu_balanced`
- `epochs = 15` hoặc `25`
- `class_bias = loss`
- `cache_format = raw`

## 90. Mẫu chọn cấu hình theo máy mạnh hơn

- `training_preset = cpu_stronger`
- `epochs = 25`
- Có thể thử `train_backbone = true` nếu tài nguyên cho phép.

## 91. Mẫu chọn cấu hình để debug nhanh

- `model_name = mobilenet_v3_small`
- `image_size = 160`
- `batch_size = 8`
- `epochs = 3`
- `no_progress = true`

## 92. Mẫu chọn cấu hình để kiểm tra imbalance

- `class_bias = loss`
- Hoặc `class_bias = both` nếu muốn thử sampler nữa.

## 93. CLI training commands chính

- `summary`
- `export-spec`
- `export-labels`
- `warm-cache`
- `pseudo-label`
- `fit`
- `evaluate`
- `export-onnx`
- `report`
- `benchmark`

## 94. Flag chung quan trọng của training

- `--manifest`
- `--training-preset`
- `--experiment-name`
- `--training-backend`
- `--model-name`
- `--no-pretrained`
- `--train-backbone`
- `--image-size`
- `--batch-size`
- `--epochs`
- `--num-workers`
- `--device`
- `--cache-dir`
- `--resume-from`
- `--checkpoint`
- `--onnx-output`
- `--spec-output`
- `--cache-format`
- `--no-rust-cache`
- `--force-cache`
- `--class-bias`
- `--early-stopping-patience`
- `--early-stopping-min-delta`
- `--disable-early-stopping`
- `--onnx-opset`
- `--export-batch-size`
- `--skip-onnx-verify`
- `--pseudo-label-threshold`
- `--pseudo-label-margin`
- `--no-progress`

## 95. Flag `--training-backend`

- Hợp lệ hiện tại là `pytorch` hoặc `rust_tch`.
- Nhưng `fit`, `evaluate`, `pseudo-label`, `export-onnx` chỉ chạy với `pytorch`.
- `rust_tch` hiện mang tính preview cho `summary`, `export-spec`, `benchmark`.

## 96. Flag `--model-name`

- Chọn backbone CNN.
- Nếu truyền model không hỗ trợ, code sẽ báo lỗi.
- Test xác nhận điều này nằm ở `test_build_model_stub_rejects_unknown_cnn_backbone`.

## 97. Flag `--no-pretrained`

- Tắt load pretrained weights.
- Chỉ nên dùng khi bạn có lý do rõ ràng.
- Trên dataset vừa và nhỏ, pretrained thường là lựa chọn tốt hơn.

## 98. Flag `--train-backbone`

- Bật fine-tune cả backbone.
- Mặc định bị tắt, tức là backbone sẽ freeze.

## 99. Flag `--cache-format`

- Chọn `png` hoặc `raw`.
- `raw` thường nhanh hơn cho training dài.
- `png` dễ inspect hơn.

## 100. Flag `--class-bias`

- Chọn `none`, `loss`, `sampler`, `both`.
- Đây là flag cực quan trọng khi data lệch lớp.

## 101. Flag `--resume-from`

- Cho trainer resume từ checkpoint latest hoặc checkpoint tùy chọn.

## 102. Flag `--checkpoint`

- Dùng cho `evaluate`, `pseudo-label`, `export-onnx`.
- Chỉ định checkpoint cụ thể để đọc.

## 103. Flag `--onnx-output`

- Cho phép đổi nơi ghi file ONNX.

## 104. Flag `--spec-output`

- Cho phép đổi nơi ghi `experiment_spec`.

## 105. Flag `--force-cache`

- Ép rebuild cache dù file cache đã tồn tại.

## 106. Flag `--no-rust-cache`

- Tắt hoàn toàn bước cache Rust.
- Trainer khi đó đọc ảnh gốc trực tiếp.

## 107. Flag `--onnx-opset`

- Chọn opset của ONNX export.
- Mặc định hiện tại là `17`.

## 108. Flag `--export-batch-size`

- Chọn batch size của dummy input khi export ONNX.
- Mặc định là `1`.

## 109. Flag `--skip-onnx-verify`

- Bỏ bước verify ONNX.
- Chỉ nên dùng khi bạn biết rõ mình đang làm gì.

## 110. Các test quan trọng cho training

- `test_trainer_builds_datasets_and_exports_labels`
- `test_trainer_warm_cache_prefers_cached_images`
- `test_build_model_stub_supports_multiple_cnn_backbones`
- `test_fit_stops_early_when_validation_loss_stalls`
- `test_fit_can_resume_from_latest_checkpoint`
- `test_fit_handles_keyboard_interrupt_and_saves_latest_checkpoint`
- `test_export_onnx_writes_manifest_and_export_report`
- `test_benchmark_writes_report_for_pytorch_backend`

## 111. Những test đó giúp bạn hiểu gì

- Backbone nào được hỗ trợ.
- Early stopping phải hoạt động ra sao.
- Resume checkpoint phải hoạt động ra sao.
- Interrupt phải vẫn lưu latest checkpoint.
- ONNX export phải ghi đúng report.
- Benchmark phải chạy trọn pipeline.

## 112. Command gợi ý cho baseline CPU nhanh

```powershell
cd localagent
uv run python -m localagent.training.train fit --experiment-name waste-fast --training-preset cpu_fast --epochs 10 --no-progress
```

## 113. Command gợi ý cho baseline cân bằng

```powershell
cd localagent
uv run python -m localagent.training.train fit --experiment-name waste-balanced --training-preset cpu_balanced --epochs 15 --no-progress
```

## 114. Command gợi ý cho cấu hình mạnh hơn

```powershell
cd localagent
uv run python -m localagent.training.train fit --experiment-name waste-stronger --training-preset cpu_stronger --epochs 25 --no-progress
```

## 115. Command gợi ý để inspect trước khi train

```powershell
cd localagent
uv run python -m localagent.training.train summary --training-preset cpu_fast
```

## 116. Command gợi ý để chỉ warm cache

```powershell
cd localagent
uv run python -m localagent.training.train warm-cache --training-preset cpu_fast
```

## 117. Command gợi ý để benchmark end to end

```powershell
cd localagent
uv run python -m localagent.training.train benchmark --training-preset cpu_fast --experiment-name baseline-waste-sorter --no-progress
```

## 118. Điều cần nhớ nhất của file này

- Chọn cấu hình phải dựa trên máy thật của bạn.
- Đừng bật `train-backbone` trên CPU yếu rồi mong đợi runtime ngắn.
- Đừng bỏ class weighting khi dataset lệch nặng.
- Đừng bỏ cache khi dataset lớn.
- Đừng export ONNX trước khi có checkpoint tốt.

## 119. Phụ lục A: cách chọn cấu hình theo phần cứng thật

- Nếu máy chỉ có CPU phổ thông, bắt đầu bằng preset `cpu_fast`.
- Preset `cpu_fast` được khai báo ở `localagent/python/localagent/training/train.py:18`.
- Preset này dùng `mobilenet_v3_small`.
- Preset này dùng `image_size = 160`.
- Preset này dùng `batch_size = 32`.
- Preset này phù hợp khi mục tiêu là có baseline sớm.
- Nếu CPU ổn hơn và bạn muốn cân bằng tốc độ với độ chính xác, dùng `cpu_balanced`.
- Preset `cpu_balanced` nằm ở `train.py:25`.
- Preset này dùng `resnet18`.
- Preset này dùng `image_size = 224`.
- Preset này dùng `batch_size = 16`.
- Nếu máy mạnh hơn và chấp nhận chậm, dùng `cpu_stronger`.
- Preset `cpu_stronger` nằm ở `train.py:32`.
- Preset này dùng `efficientnet_b0`.
- Preset này dùng `batch_size = 8`.
- Nếu bạn có GPU nhưng repo đang vận hành chủ yếu trên CPU, hãy xác minh `device` trước bằng `auto` hoặc `cuda`.
- Hàm quyết định device nằm ở `WasteTrainer._resolve_device()` tại `localagent/python/localagent/training/trainer.py:1884`.
- Nếu `device = auto`, trainer sẽ tự chọn theo khả năng môi trường.
- Nếu hết RAM khi warm cache, giảm `image_size` hoặc đổi `cache_format`.
- Nếu hết thời gian train, giảm `epochs` trước khi giảm chất lượng dữ liệu.
- Nếu metric dao động mạnh, kiểm tra `batch_size` có quá nhỏ không.
- Nếu machine rất yếu, giữ `train_backbone = false`.
- Logic freeze backbone nằm ở `_freeze_model_backbone()` tại `trainer.py:208`.
- Nếu muốn fine-tune toàn bộ backbone, bật `--train-backbone` nhưng chỉ khi runtime chấp nhận được.
- Nếu dữ liệu lệch lớp nặng, giữ `class_bias = loss` hoặc `both`.
- Logic class weights nằm ở `_loss_class_weights()` tại `trainer.py:2069`.
- Logic sampler bias nằm ở `_build_train_sampler()` tại `trainer.py:2048`.

## 120. Phụ lục B: catalog flag training và ý nghĩa vận hành

- `--training-preset` chọn bộ cấu hình mặc định.
- `--experiment-name` quyết định prefix của mọi artifact report, checkpoint và benchmark.
- `--training-backend` hiện nên để `pytorch`.
- Backend `rust_tch` có scaffold nhưng chưa hỗ trợ đầy đủ cho `fit/evaluate/export-onnx/pseudo-label`.
- Ràng buộc đó được enforce ở `localagent/src/jobs/commands.rs`.
- `--model-name` chọn backbone CNN.
- Repo hiện hỗ trợ `mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18`, `efficientnet_b0`.
- Việc thay classifier head nằm ở `_replace_classifier_head()` tại `trainer.py:192`.
- `--no-pretrained` tắt pretrained backbone.
- Nếu tắt pretrained trên dataset nhỏ, bạn thường sẽ mất chất lượng.
- `--train-backbone` cho phép fine-tune backbone thay vì chỉ train head.
- `--image-size` tác động trực tiếp tới chi phí tính toán.
- `--batch-size` tác động tới RAM, throughput và độ ổn định gradient.
- `--epochs` đặt số epoch tối đa chứ không chắc trainer sẽ chạy hết.
- Early stopping có thể dừng sớm.
- `--num-workers` điều khiển số worker của DataLoader.
- Trên Windows, `num_workers = 0` thường ít rắc rối hơn.
- `--device` chọn `auto`, `cpu` hoặc thiết bị tăng tốc khác nếu môi trường hỗ trợ.
- `--cache-dir` đổi chỗ lưu cache ảnh.
- `--resume-from` tiếp tục từ checkpoint cũ.
- `--checkpoint` chỉ đường dẫn checkpoint cụ thể cho evaluate hoặc export.
- `--onnx-output` đổi đường dẫn file ONNX xuất ra.
- `--spec-output` đổi đường dẫn experiment spec.
- `--cache-format` quyết định kiểu cache, ví dụ `raw`.
- `--no-rust-cache` tắt nhánh bridge Rust khi warm cache.
- `--force-cache` buộc tạo lại cache.
- `--class-bias` chọn `none`, `loss`, `sampler` hoặc `both`.
- `--early-stopping-patience` đặt số epoch chờ cải thiện.
- `--early-stopping-min-delta` đặt ngưỡng cải thiện tối thiểu.
- `--disable-early-stopping` tắt cơ chế dừng sớm.
- `--onnx-opset` chọn phiên bản opset ONNX.
- `--export-batch-size` chọn batch dùng khi export.
- `--skip-onnx-verify` bỏ qua bước so ONNX với PyTorch.
- `--pseudo-label-threshold` đặt ngưỡng confidence.
- `--pseudo-label-margin` đặt ngưỡng chênh lệch top1-top2.
- `--no-progress` in log text thay vì progress bar dày.
- Tất cả flag này được khai báo ở `localagent/python/localagent/training/train.py:56`.
- Phần build config từ flag vào dataclass nằm ở `train.py:113`.

## 121. Phụ lục C: vài cấu hình gợi ý theo tình huống

- Khi mục tiêu là dựng baseline nhanh trên CPU, dùng `cpu_fast`, `epochs = 10`, `class_bias = loss`.
- Khi mục tiêu là báo cáo demo có chất lượng khá, dùng `cpu_balanced`, giữ pretrained, giữ early stopping.
- Khi mục tiêu là thử model mạnh hơn mà chưa có GPU, dùng `cpu_stronger` nhưng chấp nhận runtime dài.
- Khi dữ liệu tăng gấp đôi, ưu tiên warm cache trước khi tăng model size.
- Khi lớp hiếm bị bỏ sót, thử `class_bias = both`.
- Khi validation loss lên xuống mạnh, giảm learning rate hoặc giữ backbone frozen lâu hơn.
- Khi muốn export model phục vụ inference ngay, giữ `image_size` và normalization ổn định giữa train và export.
- Khi muốn pseudo-label tập outlier, giữ threshold cao như `0.95` và margin cao như `0.25`.
- Snapshot hiện tại của repo đã từng chạy pseudo-label với ngưỡng đó trong `baseline-waste-sorter_pseudo_label.json`.
- Khi muốn benchmark một run mới với run cũ, dùng command benchmark qua server hoặc CLI.
- Benchmark artifact được tổng hợp trong `baseline-waste-sorter_benchmark.json`.
- Nếu bạn chỉ muốn xem config hiện tại mà chưa train, dùng command `summary` hoặc `export-spec`.

## 122. Phụ lục D: lỗi phổ biến khi train và cách dò

- Nếu `fit` bị chặn từ UI, kiểm tra `workflow_state` xem Step 3 đã mở chưa.
- Nếu `fit` chạy nhưng accuracy thấp bất thường, kiểm tra lại manifest accepted labels.
- Nếu `pseudo-label` không chấp nhận mẫu nào, hạ threshold hoặc xem model có thật sự đủ tốt chưa.
- Nếu `export-onnx` fail, đọc log tại `GET /jobs/{job_id}/logs`.
- Nếu ONNX verify fail, đừng dùng file `.onnx` đó cho production.
- Nếu resume không hoạt động, kiểm tra đường dẫn checkpoint và logic `_load_resume_state()` tại `trainer.py:505`.
- Nếu DataLoader lỗi trên Windows, thử `num_workers = 0`.
- Nếu out-of-memory, giảm `batch_size`.
- Nếu train quá chậm, giảm `image_size` hoặc đổi model nhỏ hơn.
- Nếu lớp hiếm có recall thấp, kiểm tra `class_weight_map` trong training report.
- Nếu kết quả mỗi lần chạy lệch đáng kể, xem lại seed và nguồn ngẫu nhiên.
- Nếu UI vẫn hiển thị run cũ, refresh `run_index.json` hoặc nhìn lại `ArtifactStore` ở `localagent/src/artifacts.rs`.

## 123. Phụ lục E: map giữa training và test

- `localagent/tests/test_train_cli.py` bao phủ parser, preset và nhiều flag build config.
- `localagent/tests/test_training_fit.py:12` kiểm tra early stopping hoạt động.
- `localagent/tests/test_training_fit.py:73` kiểm tra resume từ checkpoint.
- `localagent/tests/test_training_fit.py:154` kiểm tra `KeyboardInterrupt` vẫn lưu checkpoint latest.
- `localagent/tests/test_training_fit.py:201` kiểm tra chế độ `--no-progress`.
- `localagent/tests/test_training_artifacts.py:13` kiểm tra fit ghi evaluation report và confusion matrix.
- `localagent/tests/test_training_artifacts.py:62` kiểm tra evaluate dùng checkpoint đã lưu.
- `localagent/tests/test_training_artifacts.py:136` kiểm tra export ONNX và model manifest.
- `localagent/tests/test_training_artifacts.py:260` kiểm tra benchmark cho backend PyTorch.
- `localagent/tests/test_training_artifacts.py:309` xác nhận backend Rust hiện chưa được hỗ trợ cho benchmark đầy đủ.
- `localagent/tests/test_training_pseudo_label.py:14` kiểm tra confidence gate của pseudo-label.

## 124. Phụ lục F: checklist trước khi bấm `fit`

- Manifest đã được tạo lại sau thay đổi dataset gần nhất.
- `validate-labels` đã pass.
- `summary.json` cho thấy Step 1 đã ổn.
- Cluster review đã được promote nếu workflow yêu cầu accepted labels.
- `training_preset` đã chọn đúng theo phần cứng.
- `model_name` khớp mục tiêu chạy thử hay chạy nghiêm túc.
- `image_size` không vượt quá RAM máy.
- `batch_size` không quá tham so với máy.
- `class_bias` không bị để `none` khi dữ liệu lệch nặng.
- `train_backbone` chỉ bật khi bạn chấp nhận runtime tăng mạnh.
- `device` không bị ghi sai.
- `experiment_name` không đè lên run bạn muốn giữ riêng.
- `cache_dir` còn đủ dung lượng đĩa.
- `resume_from` chỉ dùng khi bạn thật sự muốn nối tiếp run cũ.
- `onnx_output` không trỏ tới chỗ bạn không muốn ghi đè.
- `early_stopping` đang bật nếu bạn cần tiết kiệm thời gian CPU.
- `no_progress` nên bật khi chạy qua server để log sạch hơn.
- Bạn đã biết checkpoint sẽ nằm ở đâu.
- Bạn đã biết report sẽ nằm ở đâu.
- Bạn đã biết UI sẽ đọc run nào sau khi job hoàn tất.

## 125. Phụ lục G: câu hỏi tự kiểm tra sau khi train xong

- `best_epoch` có hợp lý không.
- `epochs_completed` có thấp hơn `epochs` vì early stopping không.
- `best_loss` có giảm so với baseline trước không.
- `test_accuracy` có tăng không.
- `macro_f1` có tăng không.
- `weighted_f1` có che giấu vấn đề lớp hiếm không.
- Per-class recall của lớp hiếm có cải thiện không.
- `class_weight_map` có phản ánh đúng lệch lớp không.
- `cache_summary` có cho biết cache được reuse không.
- `device` có đúng như kỳ vọng không.
- `export_onnx` có verify pass không.
- `model_manifest.json` có khớp labels hiện tại không.
- `run_index.json` có ghi nhận run mới không.
- `compare_runs` qua API có cho bạn thấy regression không.
- Nếu câu trả lời cho nhiều câu trên là “không rõ”, hãy đọc lại `trainer.py:1270` và các report JSON trước khi chỉnh model.

## 126. Phụ lục H: profile cấu hình máy gợi ý

- Máy văn phòng yếu: `cpu_fast`, `epochs 3-10`, `num_workers 0`, `image_size 160`.
- Máy văn phòng vừa: `cpu_balanced`, `epochs 10-25`, `image_size 224`, `batch_size 16`.
- Máy CPU khá mạnh: `cpu_stronger`, `epochs 10-25`, `batch_size 8`, cân nhắc warm cache kỹ.
- Máy có GPU nhưng chưa rõ driver: bắt đầu `device auto`, đọc log resolve device.
- Máy ít RAM: giảm `batch_size` trước, rồi mới giảm `image_size`.
- Máy ít đĩa: quản lý cache cẩn thận, không giữ nhiều cache format khác nhau.
- Máy dùng Windows: ưu tiên `num_workers 0` trước khi tăng worker.
- Máy demo ngắn hạn: chọn preset ổn định hơn là cấu hình phức tạp.
- Máy production benchmark: luôn lưu benchmark report để so run giữa các profile.
- Nếu chưa biết máy mình thuộc profile nào, chạy smoke test `cpu_fast` trước rồi mới nâng dần.
- Nếu muốn tối ưu sâu hơn, đọc thêm `docs/06` để hiểu trade-off giữa loss, sampler và metric.

## 127. Phụ lục I: mốc kiểm tra sau smoke test

- Lệnh `fit` chạy hết mà không crash.
- `*_training.json` được ghi ra.
- `*_evaluation.json` được ghi ra.
- `run_index.json` có run mới.
- `macro_f1` không bằng `0`.
- `class_weight_map` có giá trị hợp lý.
- `cache_summary` không báo lỗi hàng loạt.
- Nếu các điều kiện trên đều đạt, bạn có thể nâng cấu hình dần.

## 128. Dòng chốt

- Smoke test pass là vé vào vòng tối ưu sâu hơn.
