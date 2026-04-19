# Tài liệu 05: Rust server, API, fetch và giao diện

## 1. File này nói về phần nào của hệ thống

- File này nói về tầng điều phối.
- Tầng điều phối ở dự án này là server Rust cộng với UI Next.js.
- Đây là lớp nối người dùng với Python CLI.
- Đây cũng là lớp giúp bạn không phải gõ command dài trong terminal.

## 2. Kiến trúc một câu

- UI gọi Rust server.
- Rust server spawn Python CLI.
- Python ghi artifact.
- Rust server đọc artifact và trả JSON cho UI.

## 3. Vì sao dùng Rust server thay vì để UI gọi Python trực tiếp

- UI trên trình duyệt không nên trực tiếp spawn process cục bộ.
- Rust server dễ quản lý process, log, status và job history hơn.
- Rust server cũng gom artifact thành API sạch.

## 4. File Rust quan trọng nhất của tầng server

- `localagent/src/bin/server.rs`

## 5. File Rust phụ quan trọng

- `localagent/src/workflow.rs`
- `localagent/src/jobs/types.rs`
- `localagent/src/jobs/commands.rs`
- `localagent/src/jobs/runtime.rs`
- `localagent/src/artifacts.rs`
- `localagent/src/cluster_review.rs`
- `localagent/src/inference.rs`
- `localagent/src/config.rs`

## 6. File UI quan trọng nhất

- `interface/next.config.ts`
- `interface/lib/localagent.ts`
- `interface/components/use-localagent-controller.ts`
- `interface/components/localagent/controller-actions.ts`

## 7. Port mặc định của local server

- `127.0.0.1:8080`

## 8. Vị trí code cấu hình port server

- `RuntimeConfig` trong `localagent/src/config.rs`
- Mặc định `server_host = 127.0.0.1`
- Mặc định `server_port = 8080`

## 9. Rewrite từ Next.js sang local server

- `interface/next.config.ts` rewrite `/api/localagent/:path*` thành `${LOCALAGENT_API_BASE}/:path*`
- Mặc định `LOCALAGENT_API_BASE` là `http://127.0.0.1:8080`

## 10. Hệ quả của rewrite

- Từ phía UI, code chỉ cần gọi `/api/localagent/...`.
- Từ phía người dùng, không cần lo CORS cho local dev theo đường chuẩn của repo.

## 11. Vị trí code của `API_PREFIX`

- `interface/lib/localagent.ts:1`
- `API_PREFIX = "/api/localagent"`

## 12. `fetchJson` làm gì

- Gọi `fetch`.
- Thêm `Content-Type: application/json`.
- Tắt cache bằng `cache: "no-store"`.
- Nếu response lỗi, cố gắng đọc JSON error message.
- Nếu không đọc được JSON, fallback sang text.
- Nếu mọi thứ ổn, parse JSON và trả object typed.

## 13. Vị trí code của `fetchJson`

- `interface/lib/localagent.ts:272`

## 14. Vì sao `fetchJson` quan trọng

- Đây là contract chung của UI.
- Hầu hết action của giao diện đi qua hàm này.
- Nếu API bị đổi schema lỗi, rất nhiều màn sẽ hỏng cùng lúc.

## 15. Hook điều phối chính của UI

- `useLocalAgentController`
- Nằm ở `interface/components/use-localagent-controller.ts`

## 16. Hook này giữ state gì

- `runs`
- `jobs`
- `streamConnected`
- `selectedExperiment`
- `compareExperiment`
- `runDetail`
- `comparison`
- `workflowState`
- `activeJobId`
- `activeLogs`
- `clusterReview`
- `trainingPresets`
- `pipelineCatalog`
- `trainingForm`
- `pipelineForm`

## 17. Action UI gửi request nằm ở đâu

- `useLocalAgentActions`
- Nằm ở `interface/components/localagent/controller-actions.ts`

## 18. `loadCatalog` gọi gì

- `GET /presets/training`
- `GET /presets/pipeline`

## 19. `loadRuns` gọi gì

- `GET /runs`

## 20. `loadRunDetail` gọi gì

- `GET /runs/{experiment_name}`

## 21. `loadComparison` gọi gì

- `GET /runs/{left}/compare?with={right}`

## 22. `loadClusterReview` gọi gì

- `GET /cluster-review?review_file=...`

## 23. `loadWorkflowState` gọi gì

- `GET /workflow/state?review_file=...`

## 24. `loadJobs` gọi gì

- `GET /jobs`
- Và nếu có `activeJobId` thì gọi thêm `GET /jobs/{job_id}/logs`

## 25. `saveClusterReview` gọi gì

- `PUT /cluster-review`

## 26. `submitPipeline` gọi gì

- `POST /jobs/pipeline`

## 27. `submitTraining` gọi gì

- `POST /jobs/training`
- Hoặc `POST /jobs/benchmark`

## 28. `cancelActiveJob` gọi gì

- `POST /jobs/{job_id}/cancel`

## 29. WebSocket log stream dùng endpoint nào

- `/ws/jobs`

## 30. Vị trí code dựng URL WebSocket

- `buildJobsWebSocketUrl` ở `interface/lib/localagent.ts`

## 31. WebSocket event type chính là gì

- `snapshot`
- `job_updated`
- `log_line`
- `resync_required`

## 32. Vị trí code type của WebSocket event

- `JobStreamEvent` ở `interface/lib/localagent.ts`
- Rust counterpart cũng ở `localagent/src/jobs/types.rs`

## 33. `GET /health` trả gì

- `status`
- `service`
- `default_experiment`
- `jobs_tracked`
- `artifacts.training`
- `artifacts.evaluation`
- `artifacts.export`
- `artifacts.benchmark`
- `artifacts.experiment_spec`
- `artifacts.run_index`
- `artifacts.model_manifest`

## 34. `POST /classify` làm gì

- Nhận `sample_ids`.
- Gọi `WasteClassifier.classify_batch`.
- Hiện tại đường `sample_id` này là stub logic trong Rust inference.

## 35. `POST /classify/image` làm gì

- Nhận `image_base64`.
- Có thể nhận `file_name`.
- Có thể nhận `top_k`.
- Chạy ONNX inference thật.
- Trả danh sách prediction có score.

## 36. Vì sao phải phân biệt `classify` và `classify/image`

- `classify` theo `sample_id` hiện chưa map thẳng đến ảnh trong dataset bằng model thật.
- `classify/image` là luồng inference ảnh upload thật sự bằng ONNX.

## 37. Vị trí code của classify ảnh upload

- Route ở `localagent/src/bin/server.rs:113`
- Logic infer ở `localagent/src/inference.rs`

## 38. `GET /jobs` trả gì

- Mảng `jobs`
- Mỗi job là `JobRecord`

## 39. `JobRecord` chứa gì

- `job_id`
- `job_type`
- `command`
- `experiment_name`
- `status`
- `progress_hint`
- `created_at`
- `started_at`
- `finished_at`
- `exit_code`
- `stdout_log_path`
- `stderr_log_path`
- `artifacts`
- `error`
- `cancel_requested`

## 40. Vị trí code schema `JobRecord`

- `localagent/src/jobs/types.rs`

## 41. Job status có các giá trị nào

- `pending`
- `running`
- `completed`
- `failed`
- `cancelled`

## 42. `POST /jobs/pipeline` nhận gì

- `command`
- `raw_dir`
- `manifest_dir`
- `report_dir`
- `min_width`
- `min_height`
- `train_ratio`
- `val_ratio`
- `test_ratio`
- `seed`
- `num_clusters`
- `infer_filename_labels`
- `labels_file`
- `review_file`
- `output`
- `no_progress`

## 43. Vị trí code schema `PipelineJobRequest`

- `localagent/src/jobs/types.rs`

## 44. `POST /jobs/training` nhận gì

- `command`
- `manifest`
- `training_preset`
- `experiment_name`
- `training_backend`
- `model_name`
- `pretrained_backbone`
- `train_backbone`
- `image_size`
- `batch_size`
- `epochs`
- `num_workers`
- `device`
- `cache_dir`
- `resume_from`
- `checkpoint`
- `onnx_output`
- `spec_output`
- `cache_format`
- `use_rust_cache`
- `force_cache`
- `class_bias`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `enable_early_stopping`
- `onnx_opset`
- `export_batch_size`
- `verify_onnx`
- `pseudo_label_threshold`
- `pseudo_label_margin`
- `no_progress`

## 45. Vị trí code schema `TrainingJobRequest`

- `localagent/src/jobs/types.rs`

## 46. `POST /jobs/benchmark` nhận gì

- Toàn bộ trường của `TrainingJobRequest`
- Cộng thêm `compare_to`
- Hoặc `compare_experiment`

## 47. `jobs/commands.rs` làm nhiệm vụ gì

- Validate request từ API.
- Map request JSON sang argv CLI.
- Chặn các trường không hợp lệ.
- Chặn workflow nếu step trước chưa complete.

## 48. `pipeline_args` làm gì

- Tạo argv dạng:
- `uv run python -m localagent.data.pipeline <command> ...`

## 49. `training_args` làm gì

- Tạo argv dạng:
- `uv run python -m localagent.training.train <command> ...`

## 50. Ví dụ mapping UI action sang lệnh thật

- UI `submitPipeline("run-all")`
- Rust build thành `uv run python -m localagent.data.pipeline run-all --no-progress`

## 51. Ví dụ mapping training action sang lệnh thật

- UI `submitTraining("fit")`
- Rust build thành `uv run python -m localagent.training.train fit --experiment-name ... --training-preset ... --no-progress`

## 52. `jobs/runtime.rs` làm gì

- Spawn process `uv`.
- Capture stdout và stderr.
- Ghi log file.
- Lưu `JobRecord` JSON.
- Emit event qua broadcast channel.
- Cập nhật status khi job xong.
- Sync run index.

## 53. Vị trí code spawn process

- `spawn_job` trong `localagent/src/jobs/runtime.rs`

## 54. Tại sao command được spawn bằng `uv`

- Vì Python env của repo đang được quản lý bằng `uv`.
- Dùng `uv run python -m ...` đảm bảo module chạy đúng environment.

## 55. `WorkflowState` trong Rust làm gì

- Đọc artifact summary.
- Đọc manifest CSV.
- Đọc cluster review.
- Tính step nào complete.
- Tính command nào được phép chạy.
- Trả trạng thái đó cho UI.

## 56. Route `GET /workflow/state` dùng gì bên dưới

- `WorkflowState::from_config`

## 57. `ArtifactStore` trong Rust làm gì

- Đọc JSON artifact từ đĩa.
- Tạo `overview`.
- Tạo `training_overview`.
- Tạo `benchmark_overview`.
- Tạo `dashboard_summary`.
- Tạo `run_index`.
- Tạo `run_detail`.
- So sánh benchmark giữa hai run.

## 58. Route `GET /runs` dựa vào gì

- `ArtifactStore.run_index`

## 59. Route `GET /runs/{experiment_name}` dựa vào gì

- `ArtifactStore.run_detail`

## 60. Route `GET /runs/{experiment_name}/compare` dựa vào gì

- `ArtifactStore.compare_runs`

## 61. Route `GET /presets/training` trả gì

- `cpu_fast`
- `cpu_balanced`
- `cpu_stronger`
- Mỗi preset có `model_name`, `image_size`, `batch_size`, `cache_format`, `class_bias`

## 62. Route `GET /presets/pipeline` trả gì

- Danh sách `dataset_commands`
- Danh sách `training_commands`

## 63. Route `GET /artifacts/overview` trả gì

- Một payload tổng hợp overview của experiment.

## 64. Route `GET /artifacts/training-overview` trả gì

- Training history và chart series rút gọn.

## 65. Route `GET /artifacts/benchmarks` trả gì

- Benchmark overview cùng các report liên quan.

## 66. Route `GET /dashboard/summary` trả gì

- `status`
- `cards`
- `charts`
- `dataset_summary`
- `benchmark`
- `export`
- `experiment_spec`
- `model_manifest`

## 67. Vì sao UI thích `dashboard/summary`

- Vì chỉ cần một call là có nhiều dữ liệu cho dashboard.

## 68. Route `GET /dataset/image` làm gì

- Nhận `relative_path`.
- Sanitize path.
- Resolve đường dẫn thật dưới dataset root.
- Đảm bảo path không thoát ra ngoài dataset root.
- Đọc bytes ảnh và trả đúng content type.

## 69. Vì sao route ảnh dataset cần sanitize path

- Để tránh path traversal.
- Để UI không đọc được file tùy ý ngoài dataset.

## 70. Route `GET /cluster-review` làm gì

- Đọc trạng thái review cụm từ manifest CSV và review CSV.
- Trả đầy đủ cluster, đại diện, notes và stale count.

## 71. Route `PUT /cluster-review` làm gì

- Lưu quyết định review từ UI.
- Chặn lưu nếu đang có dataset pipeline job chạy.
- Kiểm stale payload.

## 72. Vì sao không cho lưu cluster review khi dataset job đang chạy

- Vì manifest có thể đang bị thay đổi.
- Điều đó làm review state stale ngay tức thì.

## 73. `next.config.ts` ảnh hưởng WebSocket không

- Không trực tiếp.
- WebSocket URL được build riêng bằng env `NEXT_PUBLIC_LOCALAGENT_WS_BASE` hoặc API base.

## 74. `LocalAgentApiClient` trong Python là gì

- Đây là client nhỏ dùng `httpx`.
- Nó có `health` và `classify`.
- Nằm ở `localagent/python/localagent/api/client.py`.
- Đây không phải lớp chính mà UI dùng.

## 75. `RustBackendBridge` trong Python là gì

- Là cầu nối để Python gọi Rust classifier.
- Nếu extension không load được, nó trả `None`.

## 76. `RustAccelerationBridge` trong Python là gì

- Là cầu nối để Python gọi Rust cache và metric.

## 77. `python_api.rs` expose gì sang Python

- `ping`
- `compute_class_weight_map_json`
- `build_classification_report_json`
- `prepare_image_cache`
- Class `RustBackend`

## 78. Inference Rust thật dùng file nào

- `localagent/src/inference.rs`

## 79. `WasteClassifier` trong Rust load gì

- `model_manifest.json`
- `waste_classifier.onnx`
- normalization mean/std
- label list

## 80. `classify_uploaded_image` preprocess gì

- Decode base64.
- Load ảnh.
- Resize về `image_size` trong manifest.
- Normalize theo mean/std.
- Chạy ONNX Runtime.
- Softmax.
- Sắp xếp prediction giảm dần.

## 81. Vị trí code preprocess ảnh upload

- `preprocess_image` trong `localagent/src/inference.rs`

## 82. Vị trí code softmax trong Rust inference

- `softmax` trong `localagent/src/inference.rs`

## 83. `useLocalAgentController` phản ứng khi job hoàn tất thế nào

- Reload runs.
- Reload run detail.
- Reload comparison nếu có.
- Reload cluster review.
- Reload workflow state.

## 84. Điều đó mang lại gì cho UX

- Người dùng bấm chạy job xong không phải refresh tay quá nhiều.

## 85. `controller-actions.ts` xây payload pipeline thế nào

- Luôn có `command`.
- Luôn có `no_progress`.
- Có thể thêm `labels_file`, `review_file`, `output`, `num_clusters`.

## 86. `controller-actions.ts` xây payload training thế nào

- Truyền `experiment_name`
- Truyền `training_preset`
- Truyền `training_backend`
- Truyền `model_name`
- Truyền `image_size`
- Truyền `batch_size`
- Truyền `epochs`
- Truyền `class_bias`
- Truyền `device`
- Truyền `pseudo_label_threshold`
- Truyền `pseudo_label_margin`
- Truyền `no_progress`

## 87. Vì sao UI default `no_progress = true`

- Vì progress bar terminal không hữu ích khi được stream sang log file.
- Text log sạch thường đọc dễ hơn trên panel log.

## 88. Nếu API trả lỗi, UI làm gì

- `fetchJson` ném `Error`.
- Hook/action bắt lỗi và ghi `connectionError` hoặc `clusterReviewError`.

## 89. Nếu WebSocket mất kết nối, UI làm gì

- Đặt `streamConnected = false`.
- Fallback sang polling `GET /jobs` theo interval.

## 90. Điều đó tốt ở điểm nào

- Dù WebSocket lỗi, UI vẫn có thể tiếp tục theo dõi job ở mức polling.

## 91. Route nào dùng nhiều nhất khi xem dashboard

- `/runs`
- `/runs/{experiment}`
- `/runs/{experiment}/compare`
- `/workflow/state`
- `/cluster-review`
- `/jobs`
- `/jobs/{job_id}/logs`

## 92. Route nào dùng nhiều nhất khi chạy pipeline từ UI

- `/jobs/pipeline`
- `/jobs/training`
- `/jobs/benchmark`
- `/ws/jobs`

## 93. Route nào dùng nhiều nhất khi xem artifact chi tiết

- `/artifacts/dataset`
- `/artifacts/training`
- `/artifacts/evaluation`
- `/artifacts/export`
- `/artifacts/benchmark`
- `/artifacts/model-manifest`
- `/artifacts/experiment-spec`

## 94. Các env đáng chú ý ở UI

- `LOCALAGENT_API_BASE`
- `NEXT_DIST_DIR`
- `NEXT_PUBLIC_LOCALAGENT_WS_BASE`
- `NEXT_PUBLIC_LOCALAGENT_API_BASE`

## 95. Khi nào nên đọc file này trước file training

- Khi bạn không hiểu “bấm nút ở UI” dẫn tới command nào.
- Khi bạn muốn sửa API.
- Khi bạn muốn thêm endpoint mới.

## 96. Khi nào nên đọc file này sau file training

- Khi bạn đã hiểu artifact và muốn biết UI hiển thị chúng ra sao.

## 97. Điều cần nhớ nhất của file này

- Rust server là lớp orchestration.
- UI chỉ là client của Rust server.
- Python vẫn là nơi chạy pipeline thật.
- Artifact là ngôn ngữ chung giữa Python và UI.
- WebSocket chỉ để stream log và cập nhật job, không phải để chạy training logic.

## 98. Phụ lục A: catalog endpoint chính của Rust server

- `POST /classify/image` nằm ở `localagent/src/bin/server.rs:113`.
- Endpoint này nhận ảnh upload và chạy inference ONNX.
- `GET /jobs/{job_id}` nằm ở `server.rs:138`.
- Endpoint này trả metadata của một job cụ thể.
- `POST /jobs/pipeline` nằm ở `server.rs:149`.
- Endpoint này tạo job chạy dataset pipeline Python.
- `POST /jobs/training` nằm ở `server.rs:162`.
- Endpoint này tạo job chạy training command Python.
- `POST /jobs/benchmark` nằm ở `server.rs:175`.
- Endpoint này tạo job benchmark một run hoặc so với run khác.
- `POST /jobs/{job_id}/cancel` nằm ở `server.rs:188`.
- Endpoint này yêu cầu hủy job đang chạy.
- `GET /jobs/{job_id}/logs` nằm ở `server.rs:199`.
- Endpoint này trả tail log stdout và stderr.
- `GET /runs/{experiment_name}` nằm ở `server.rs:242`.
- Endpoint này trả chi tiết artifact của một experiment.
- `GET /runs/{experiment_name}/compare` nằm ở `server.rs:265`.
- Endpoint này so sánh run hiện tại với run khác.
- `GET /presets/training` nằm ở `server.rs:281`.
- Endpoint này trả preset training cho UI.
- `GET /presets/pipeline` nằm ở `server.rs:311`.
- Endpoint này trả danh mục lệnh pipeline và gợi ý workflow.
- `GET /workflow/state` nằm ở `server.rs:435`.
- Endpoint này cho UI biết bước nào đang mở hay đang khóa.
- `GET /dataset/image` nằm ở `server.rs:448`.
- Endpoint này phục vụ ảnh dataset theo `relative_path`.
- `GET /cluster-review` nằm ở `server.rs:523`.
- Endpoint này trả dữ liệu review cluster.
- `PUT /cluster-review` nằm ở `server.rs:535`.
- Endpoint này lưu review cluster đã chỉnh sửa.
- Ngoài ra server còn có các route tổng quan như `/runs`, `/jobs`, `/health`, `/ws/jobs`, `/artifacts/overview`, `/dashboard/summary`.
- Các route này được dùng cho dashboard và điều phối job.

## 99. Phụ lục B: request payload quan trọng và field thật trong code

- Kiểu dữ liệu cho pipeline request nằm ở `localagent/src/jobs/types.rs:119`.
- `PipelineJobRequest.command` nhận enum `PipelineCommand`.
- Enum `PipelineCommand` nằm ở `jobs/types.rs:86`.
- Giá trị CLI tương ứng được map trong `PipelineCommand::as_cli()`.
- Pipeline request có thể mang `raw_dir`.
- Pipeline request có thể mang `manifest_dir`.
- Pipeline request có thể mang `report_dir`.
- Pipeline request có thể mang `min_width`.
- Pipeline request có thể mang `min_height`.
- Pipeline request có thể mang `train_ratio`.
- Pipeline request có thể mang `val_ratio`.
- Pipeline request có thể mang `test_ratio`.
- Pipeline request có thể mang `seed`.
- Pipeline request có thể mang `num_clusters`.
- Pipeline request có thể mang `infer_filename_labels`.
- Pipeline request có thể mang `labels_file`.
- Pipeline request có thể mang `review_file`.
- Pipeline request có thể mang `output`.
- Pipeline request có thể mang `no_progress`.
- Kiểu dữ liệu cho training request nằm ở `jobs/types.rs:169`.
- `TrainingJobRequest.command` là optional vì benchmark flatten thêm training fields.
- Training request có thể mang `manifest`.
- Training request có thể mang `training_preset`.
- Training request có thể mang `experiment_name`.
- Training request có thể mang `training_backend`.
- Training request có thể mang `model_name`.
- Training request có thể mang `pretrained_backbone`.
- Training request có thể mang `train_backbone`.
- Training request có thể mang `image_size`.
- Training request có thể mang `batch_size`.
- Training request có thể mang `epochs`.
- Training request có thể mang `num_workers`.
- Training request có thể mang `device`.
- Training request có thể mang `cache_dir`.
- Training request có thể mang `resume_from`.
- Training request có thể mang `checkpoint`.
- Training request có thể mang `onnx_output`.
- Training request có thể mang `spec_output`.
- Training request có thể mang `cache_format`.
- Training request có thể mang `use_rust_cache`.
- Training request có thể mang `force_cache`.
- Training request có thể mang `class_bias`.
- Training request có thể mang `early_stopping_patience`.
- Training request có thể mang `early_stopping_min_delta`.
- Training request có thể mang `enable_early_stopping`.
- Training request có thể mang `onnx_opset`.
- Training request có thể mang `export_batch_size`.
- Training request có thể mang `verify_onnx`.
- Training request có thể mang `pseudo_label_threshold`.
- Training request có thể mang `pseudo_label_margin`.
- Training request có thể mang `no_progress`.
- Benchmark request nằm ở `jobs/types.rs:204`.
- Benchmark request flatten toàn bộ training request rồi thêm `compare_to` và `compare_experiment`.

## 100. Phụ lục C: UI fetch API thực sự được viết ra sao

- Hằng `API_PREFIX = "/api/localagent"` nằm ở `interface/lib/localagent.ts:1`.
- Hàm `fetchJson()` nằm ở `interface/lib/localagent.ts:272`.
- Hàm này gọi `fetch(`${API_PREFIX}${path}`, ...)`.
- Vì vậy UI không gọi trực tiếp server Rust qua host cứng trong component.
- Next.js rewrite ở `interface/next.config.ts` chuyển `/api/localagent/:path*` sang local agent base URL.
- Điều này giúp frontend và backend tách lớp rõ ràng hơn.
- `buildJobsWebSocketUrl()` nằm ở `interface/lib/localagent.ts:364`.
- Hàm này dựng URL WebSocket tương ứng cho stream log job.
- `controller-actions.ts` là nơi gói request thao tác thực tế.
- `submitPipeline()` nằm ở `interface/components/localagent/controller-actions.ts:297`.
- Hàm này đọc state từ `pipelineForm`.
- Hàm này gắn `labels_file` nếu người dùng đã nhập.
- Hàm này gắn `review_file` nếu người dùng đã nhập.
- Hàm này gắn `output` cho các lệnh export.
- Hàm này gắn `num_clusters` nếu form không trống.
- Sau đó nó gọi `fetchJson<JobRecord>("/jobs/pipeline", ...)`.
- `submitTraining()` nằm ở `controller-actions.ts:340`.
- Hàm này đọc state từ `trainingForm`.
- Hàm này map `training_preset`, `training_backend`, `model_name`, `image_size`, `batch_size`, `epochs`, `class_bias`, `device`, `pseudo_label_threshold`, `pseudo_label_margin`, `no_progress`.
- Nếu command là benchmark, endpoint đổi thành `/jobs/benchmark`.
- Nếu command không phải benchmark, endpoint là `/jobs/training`.
- `saveClusterReview()` nằm ở `controller-actions.ts:273`.
- Hàm này gọi `PUT /cluster-review`.
- `refreshAll()` nằm ở `controller-actions.ts:229`.
- Hàm này nạp catalog, runs, jobs, workflow state và cluster review.
- Vì vậy khi bạn hỏi “fetch API ra sao”, câu trả lời đúng nhất là:
- UI chủ yếu gọi `fetchJson()` trong `lib/localagent.ts`.
- Các action orchestration nằm ở `controller-actions.ts`.
- State và effect nằm ở `use-localagent-controller.ts`.

## 101. Phụ lục D: vòng đời job từ click UI đến file artifact

- Người dùng bấm nút ở giao diện.
- Component gọi action như `submitPipeline()` hoặc `submitTraining()`.
- Action dùng `fetchJson()` gửi request đến `/api/localagent/...`.
- Next.js rewrite chuyển request đó sang Rust server.
- Rust server parse JSON request bằng struct trong `jobs/types.rs`.
- Server chuyển request thành câu lệnh CLI bằng `localagent/src/jobs/commands.rs`.
- CLI thật sự là `uv run python -m localagent.data.pipeline ...` hoặc `uv run python -m localagent.training.train ...`.
- `jobs/runtime.rs` spawn process con.
- Runtime ghi stdout và stderr vào file log.
- Runtime cập nhật `JobRecord` trên đĩa.
- WebSocket phát `JobUpdated` hoặc `LogLine`.
- UI đang mở job sẽ nhận stream này qua `use-localagent-controller.ts`.
- Khi job hoàn tất, UI gọi refresh để nạp artifact mới.
- `ArtifactStore` trong Rust đọc các report JSON/CSV.
- Server đóng gói response cho `/runs`, `/dashboard/summary`, `/artifacts/...`.
- UI render lại dashboard.
- Tất cả luồng này đều không yêu cầu frontend biết chi tiết file system của Python.
- Đó là lý do Rust server rất quan trọng dù training nằm ở Python.

## 102. Phụ lục E: ví dụ payload thực chiến

```json
{
  "command": "run-all",
  "no_progress": true
}
```

- Payload trên là bản tối thiểu để mở Step 1 qua API.

```json
{
  "command": "export-cluster-review",
  "review_file": "artifacts/manifests/cluster_review.csv",
  "output": "artifacts/manifests/cluster_review.csv",
  "num_clusters": 32,
  "no_progress": true
}
```

- Payload trên phản ánh đúng kiểu field mà UI đóng gói cho export cluster review.

```json
{
  "command": "fit",
  "experiment_name": "baseline-waste-sorter",
  "training_preset": "cpu_balanced",
  "training_backend": "pytorch",
  "model_name": "resnet18",
  "image_size": 224,
  "batch_size": 16,
  "epochs": 25,
  "class_bias": "loss",
  "device": "auto",
  "pseudo_label_threshold": 0.85,
  "pseudo_label_margin": 0.15,
  "no_progress": true
}
```

- Payload trên phản ánh các field mặc định đang có trong UI state.

```json
{
  "command": "benchmark",
  "experiment_name": "baseline-waste-sorter",
  "training_preset": "cpu_fast",
  "training_backend": "pytorch",
  "compare_experiment": "older-baseline",
  "no_progress": true
}
```

- Khi benchmark, request đi tới `/jobs/benchmark` chứ không phải `/jobs/training`.

## 103. Phụ lục F: WebSocket và vì sao UI không poll log thô liên tục

- `use-localagent-controller.ts` mở socket bằng `buildJobsWebSocketUrl()`.
- Socket chỉ tập trung vào stream log và trạng thái job.
- Polling log thô liên tục sẽ tốn request hơn và trễ hơn.
- WebSocket giúp UI thấy job đang chạy theo thời gian gần thực.
- Tuy nhiên artifact chi tiết vẫn được nạp lại bằng HTTP sau khi job hoàn thành.
- Đây là kiến trúc hợp lý.
- WebSocket làm tốt việc stream sự kiện.
- HTTP làm tốt việc tải dữ liệu cấu trúc hoàn chỉnh.
- Nếu socket hỏng, UI vẫn có thể resync bằng HTTP.
- `JobStreamEvent` được khai báo ở `localagent/src/jobs/types.rs`.
- Event có thể là `Snapshot`.
- Event có thể là `JobUpdated`.
- Event có thể là `LogLine`.
- Event có thể là `ResyncRequired`.
- Chính kiểu event này giúp UI giữ trạng thái bền hơn so với tự đoán từ text log.

## 104. Phụ lục G: an toàn đường dẫn và chỗ dễ sai

- Route `GET /dataset/image` phải chống path traversal.
- Logic bảo vệ nằm ở `localagent/src/bin/server.rs:448`.
- UI chỉ gửi `relative_path`, không gửi đường dẫn tuyệt đối.
- `buildDatasetImageUrl()` nằm ở `interface/components/dashboard/discovery/discovery-shared.ts:14`.
- Hàm này `encodeURIComponent(relativePath)` trước khi gọi API.
- Nếu bạn sửa naming của `relative_path`, phải sửa cả UI và server.
- Nếu bạn sửa `API_PREFIX`, phải sửa rewrite của Next.js cho đồng bộ.
- Nếu bạn sửa schema artifact mà quên `ArtifactStore`, UI sẽ hiển thị dữ liệu cũ hoặc lỗi parse.
- Nếu bạn sửa job type mà quên `JobRecord`, WebSocket và jobs view có thể hỏng.

## 105. Phụ lục H: lộ trình đọc code API và UI

- Bắt đầu từ `interface/lib/localagent.ts` để hiểu contract phía UI.
- Tiếp theo đọc `controller-actions.ts` để biết request nào được bắn đi.
- Sau đó đọc `use-localagent-controller.ts` để biết lúc nào UI refresh.
- Rồi đọc `localagent/src/bin/server.rs` để thấy route tương ứng.
- Tiếp theo đọc `jobs/types.rs` để biết shape request thực.
- Sau đó đọc `jobs/commands.rs` và `jobs/runtime.rs` để biết route được thực thi thế nào.
- Cuối cùng đọc `artifacts.rs` để hiểu server dựng dashboard từ file cục bộ ra sao.

## 106. Phụ lục I: checklist debug API nhanh

- Endpoint có đúng method không.
- Path có đúng prefix `/api/localagent` ở UI không.
- Rewrite của Next.js có đang trỏ đúng server không.
- Server Rust có đang chạy không.
- Route tương ứng có tồn tại trong `server.rs` không.
- JSON payload có đúng field name như `jobs/types.rs` không.
- Kiểu dữ liệu số có đang bị gửi như chuỗi rỗng không.
- `toNumberString()` ở UI có đang chuyển đổi đúng không.
- Response có đúng type UI mong đợi không.
- Nếu là job endpoint, `jobs/runtime.rs` có spawn process thành công không.
- Nếu là artifact endpoint, `ArtifactStore` có đọc được file thực tế không.
- Nếu là cluster review endpoint, file review có đang stale hoặc bị lock không.
- Nếu là dataset image endpoint, `relative_path` có hợp lệ không.
- Nếu là WebSocket, URL build từ `buildJobsWebSocketUrl()` có đúng không.

## 107. Phụ lục J: mốc kiểm tra sau khi sửa API hoặc UI

- Presets vẫn load được.
- Jobs list vẫn load được.
- Workflow state vẫn load được.
- Cluster review vẫn load được.
- Save cluster review vẫn chạy được.
- Submit pipeline vẫn tạo job được.
- Submit training vẫn tạo job được.
- Log job vẫn stream qua WebSocket.
- Run detail vẫn mở được.
- Compare run vẫn mở được.
- Ảnh dataset vẫn hiện được.
- Nếu các mốc này đều ổn, contract giữa UI và server cơ bản còn nguyên.

## 108. Phụ lục K: câu hỏi chốt khi một request fail

- Fail ở frontend trước khi gửi đi hay fail từ server trả về.
- Payload đã đúng shape `jobs/types.rs` chưa.
- Route tương ứng có còn đúng method và path không.
- Job runtime có spawn process thật không.
- Artifact đích có được ghi sau request đó không.
- Log stdout và stderr nói gì.
- Nếu trả lời được các câu này, bạn sẽ khoanh vùng rất nhanh.

## 109. Dòng chốt

- API tốt là API mà bạn trace được từ click UI xuống artifact cuối cùng.
- UI tốt là UI không giấu mất contract thật của server.
- Server tốt là server giúp debug được cả request lẫn file sinh ra.

## 110. Chốt phụ cuối

- Khi UI lỗi, đừng quên kiểm tra rewrite trước.
- Khi server lỗi, đừng quên kiểm tra payload shape trước.
- Khi artifact thiếu, đừng quên kiểm tra route đọc file trước.
