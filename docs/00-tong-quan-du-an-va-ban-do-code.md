# Tài liệu 00: Tổng quan dự án và bản đồ code Smart Waste Sorting

## 1. Mục đích của file này

- File này là cửa vào chính cho người hoàn toàn mới với repository.
- Mục tiêu của file không phải là dạy machine learning từ số 0.
- Mục tiêu của file là giúp bạn biết dự án này đang làm gì.
- Mục tiêu của file là giúp bạn biết nên đọc code ở đâu trước.
- Mục tiêu của file là giúp bạn biết pipeline dữ liệu, discovery, training và server nối với nhau ra sao.
- Mục tiêu của file là giúp bạn biết chỗ nào là hiện trạng thật của code.
- Mục tiêu của file là giúp bạn biết chỗ nào chỉ là mong muốn hoặc roadmap.
- Mục tiêu của file là giúp bạn không bị lạc giữa `README.md`, `localagent/README.md`, `localagent/python/`, `localagent/src/` và `interface/`.
- Mục tiêu của file là giúp bạn tra cứu nhanh vị trí chức năng trong code.
- Mục tiêu của file là giúp bạn hiểu vì sao hiện tại pipeline có thể chạy được end to end trên CPU.

## 2. Đối tượng nên đọc file này

- Người mới clone repository lần đầu.
- Người chưa biết Rust trong repo này làm gì.
- Người chưa biết Python trong repo này làm gì.
- Người chưa biết GUI `interface/` gọi server thế nào.
- Người chưa biết dữ liệu gốc nằm ở đâu.
- Người chưa biết artifact được ghi ở đâu.
- Người chưa biết project đang dùng supervised learning hay semi-supervised learning.
- Người chưa biết project có thật sự dùng fuzzy logic hay không.
- Người cần tra nhanh từng file chính để bắt đầu đọc code.

## 3. Điều quan trọng phải nói ngay từ đầu

- Dự án hiện tại chạy chính trong thư mục `localagent/`.
- Giao diện điều khiển nằm trong thư mục `interface/`.
- Dữ liệu ảnh thô hiện tại được pipeline Python quét từ `localagent/dataset/`.
- Dự án có một số chỗ nhắc đến `datasets/`, nhưng pipeline dữ liệu chính đang dùng `dataset/`.
- Điều này không phải suy đoán.
- Điều này thể hiện ở `localagent/python/localagent/config/settings.py:111`.
- `DatasetPipelineConfig.raw_dataset_dir` mặc định là `dataset/`.
- Trong khi đó `AgentPaths.dataset_dir` lại mặc định là `datasets/` tại `localagent/python/localagent/config/settings.py:16`.
- Vì vậy nếu bạn là người mới, hãy hiểu rằng dữ liệu đầu vào thật cho pipeline hiện tại là `localagent/dataset/`.
- `datasets/` nên được hiểu là vùng mở rộng hoặc vùng dữ liệu chuẩn hóa về sau.

## 4. Kết luận ngắn nhất về kiến trúc

- Python chịu trách nhiệm cho pipeline dữ liệu, huấn luyện, pseudo-label, evaluate và export ONNX.
- Rust chịu trách nhiệm cho server cục bộ, quản lý job, stream log, đọc artifact, cluster review API và một số tăng tốc.
- Rust cũng cung cấp bridge PyO3 để Python gọi sang phần tăng tốc cache ảnh và tính toán metric.
- Next.js trong `interface/` chỉ là lớp UI.
- UI không huấn luyện trực tiếp.
- UI gửi request HTTP đến local server Rust.
- Server Rust spawn các lệnh `uv run python -m ...`.
- Kết quả của Python được ghi ra `artifacts/` và `models/`.
- Server Rust lại đọc các file JSON/CSV đó để trả API cho UI.

## 5. Dòng chảy tổng thể của hệ thống

- Bước 1 là quét ảnh thô.
- Bước 2 là tạo manifest và report chất lượng.
- Bước 3 là trích embedding từ ảnh.
- Bước 4 là gom cụm trong không gian embedding.
- Bước 5 là xuất file review theo cụm.
- Bước 6 là người dùng duyệt cụm và gán nhãn hoặc loại cụm.
- Bước 7 là promote các quyết định review đó vào manifest.
- Bước 8 là khi manifest đã có nhãn được chấp nhận, training được mở khóa.
- Bước 9 là warm cache ảnh để train nhanh hơn.
- Bước 10 là huấn luyện CNN theo manifest.
- Bước 11 là đánh giá checkpoint tốt nhất.
- Bước 12 là pseudo-label các mẫu chưa được chấp nhận nếu cần.
- Bước 13 là export ONNX và model manifest.
- Bước 14 là benchmark và tổng hợp artifact.
- Bước 15 là UI đọc lại artifact để hiển thị dashboard.

## 6. Điều dự án đang làm thật, không phải mong muốn

- Dự án đang có manifest dataset thực sự.
- Dự án đang có embedding artifact thực sự.
- Dự án đang có cluster summary thực sự.
- Dự án đang có cluster review CSV thực sự.
- Dự án đang có checkpoint `.pt` thực sự.
- Dự án đang có model ONNX thực sự.
- Dự án đang có `model_manifest.json` thực sự.
- Dự án đang có benchmark report thực sự.
- Dự án đang có run index thực sự.
- Điều này có thể kiểm tra trực tiếp ở `localagent/artifacts/` và `localagent/models/`.

## 7. Snapshot hiện trạng artifact của repository

- `localagent/artifacts/reports/summary.json` cho biết dataset hiện có `9999` file.
- `localagent/artifacts/reports/summary.json` cho biết `9989` file hợp lệ.
- `localagent/artifacts/reports/summary.json` cho biết `10` file bị đánh dấu duplicate.
- `localagent/artifacts/reports/summary.json` cho biết `9769` file sẵn sàng để training.
- `localagent/artifacts/reports/summary.json` cho biết training mode hiện tại là `accepted_labels_only`.
- `localagent/artifacts/reports/summary.json` cho biết nhãn hiện tại là `folk`, `glass`, `paper`.
- `localagent/artifacts/reports/summary.json` cho biết nguồn nhãn chấp nhận hiện nay là `cluster_review` và `model_pseudo`.
- `localagent/artifacts/reports/summary.json` cho biết số file `cluster_review` là `9199`.
- `localagent/artifacts/reports/summary.json` cho biết số file `model_pseudo` là `570`.
- `localagent/artifacts/reports/summary.json` cho biết có `790` cluster outlier file.
- `localagent/artifacts/reports/cluster_summary.json` cho biết số cụm là `32`.
- `localagent/artifacts/reports/embedding_summary.json` cho biết vector embedding có chiều `512`.
- `localagent/artifacts/reports/embedding_summary.json` cho biết extractor hiện dùng là `resnet18_imagenet`.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` cho biết preset training gần nhất là `cpu_fast`.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` cho biết backbone gần nhất là `mobilenet_v3_small`.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` cho biết `image_size=160`.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` cho biết `batch_size=32`.
- `localagent/artifacts/reports/baseline-waste-sorter_training.json` cho biết training dừng sớm ở epoch `6`.
- `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json` cho biết accuracy test là khoảng `0.9296`.
- `localagent/models/model_manifest.json` cho biết ONNX đã verify thành công với `max_abs_diff` rất nhỏ.

## 8. Cấu trúc thư mục ở mức root

- `README.md`
- `.gitignore`
- `interface/`
- `localagent/`
- `docs/`

## 9. Vai trò của `README.md` ở root

- `README.md` ở root là bản mô tả tổng quát.
- Nó giúp bạn biết dự án gồm `localagent/` và `interface/`.
- Nó cung cấp sơ đồ pipeline ở mức cao.
- Nó có ví dụ command CLI.
- Nó chưa đủ sâu để thay thế tài liệu chi tiết cho người mới hoàn toàn.
- Vì vậy thư mục `docs/` này tồn tại.

## 10. Vai trò của `localagent/`

- `localagent/` là lõi thực thi của dự án.
- Đây là nơi bạn `cd` vào để chạy đa số command.
- Đây là nơi chứa artifact, model, source Python, source Rust và test.
- Đây là phần cần đọc nếu bạn muốn hiểu pipeline thật sự.

## 11. Vai trò của `interface/`

- `interface/` là lớp trình bày.
- Nó không tự train model.
- Nó không tự scan dataset.
- Nó gọi API của server Rust.
- Nó hiển thị jobs, logs, workflow state, dashboard summary, benchmark comparison và cluster review.

## 12. Vai trò của `docs/`

- `docs/` là bộ tài liệu tiếng Việt chi tiết.
- `docs/` không thay thế source code.
- `docs/` đóng vai trò bản đồ đọc code.
- `docs/` giải thích thuật ngữ, quy trình, artifact và API.

## 13. Cấu trúc quan trọng bên trong `localagent/`

- `localagent/pyproject.toml`
- `localagent/Cargo.toml`
- `localagent/uv.lock`
- `localagent/Cargo.lock`
- `localagent/configs/`
- `localagent/dataset/`
- `localagent/datasets/`
- `localagent/models/`
- `localagent/artifacts/`
- `localagent/python/localagent/`
- `localagent/src/`
- `localagent/tests/`

## 14. Ý nghĩa của `localagent/configs/`

- Đây là chỗ dành cho file cấu hình.
- File mẫu hiện có là `localagent/configs/agent.example.toml`.
- Trong snapshot code hiện tại, hầu hết cấu hình thực tế đi qua dataclass Python và struct Rust mặc định.
- Nói cách khác, config runtime hiện tại chủ yếu là config trong code hơn là config TOML đầy đủ.

## 15. Ý nghĩa của `localagent/dataset/`

- Đây là nơi đặt ảnh đầu vào thô.
- Pipeline quét trực tiếp từ đây.
- Hàm quét thật sự nằm ở `localagent/python/localagent/data/pipeline.py:748`.
- Hàm `_iter_image_paths` sẽ `rglob("*")` thư mục này và lọc theo extension hợp lệ.
- Hàm `_inspect_image` ở `localagent/python/localagent/data/pipeline.py:763` sẽ đọc từng ảnh trong đây.

## 16. Ý nghĩa của `localagent/datasets/`

- Đây là thư mục hiện chưa phải đầu vào mặc định của pipeline dữ liệu chính.
- `AgentPaths.dataset_dir` vẫn tạo thư mục này khi gọi `ensure_layout`.
- Điều này xảy ra tại `localagent/python/localagent/config/settings.py:16` và `:20`.
- Bạn nên xem `datasets/` như vùng dành cho dữ liệu phụ trợ hoặc mở rộng tương lai.
- Nếu không đọc kỹ code, bạn rất dễ nhầm `datasets/` là nơi phải đặt ảnh thô.
- Thực tế hiện tại không phải như vậy.

## 17. Ý nghĩa của `localagent/models/`

- Đây là nơi chứa file model export và metadata cho suy luận.
- Các file hiện hữu gồm `labels.json`, `model_manifest.json`, `waste_classifier.onnx`.
- `labels.json` chứa danh sách lớp theo đúng thứ tự đầu ra model.
- `model_manifest.json` chứa metadata suy luận.
- `waste_classifier.onnx` là model export để Rust ONNX inference đọc.

## 18. Ý nghĩa của `localagent/artifacts/`

- Đây là nơi ghi toàn bộ output trung gian và output huấn luyện.
- Đây là thư mục quan trọng nhất để kiểm tra pipeline đã chạy tới đâu.
- Mọi dashboard của UI gần như đều đọc dữ liệu gián tiếp từ đây.

## 19. Bên trong `localagent/artifacts/`

- `localagent/artifacts/manifests/`
- `localagent/artifacts/reports/`
- `localagent/artifacts/checkpoints/`
- `localagent/artifacts/cache/`
- `localagent/artifacts/jobs/`

## 20. Ý nghĩa của `localagent/artifacts/manifests/`

- Đây là nơi lưu manifest dữ liệu và artifact discovery.
- File quan trọng nhất là `dataset_manifest.parquet`.
- File CSV song song là `dataset_manifest.csv`.
- File `dataset_embeddings.npz` lưu embedding sau bước `embed`.
- File `cluster_review.csv` lưu review cấp cụm.
- File `labeling_template.csv` lưu template gán nhãn thủ công theo sample.

## 21. Ý nghĩa của `localagent/artifacts/reports/`

- Đây là nơi lưu report JSON và CSV cấp pipeline.
- File `summary.json` mô tả tình trạng dữ liệu hiện tại.
- File `embedding_summary.json` mô tả bước embedding.
- File `cluster_summary.json` mô tả bước clustering.
- File `<experiment>_training.json` mô tả huấn luyện.
- File `<experiment>_evaluation.json` mô tả đánh giá.
- File `<experiment>_export.json` mô tả export ONNX.
- File `<experiment>_benchmark.json` mô tả benchmark end to end.
- File `<experiment>_report.json` là bundle tổng hợp artifact.
- File `<experiment>_experiment_spec.json` là snapshot cấu hình run.
- File `<experiment>_confusion_matrix.csv` là confusion matrix dạng CSV.
- File `run_index.json` dùng cho catalog các run.

## 22. Ý nghĩa của `localagent/artifacts/checkpoints/`

- Đây là nơi lưu checkpoint PyTorch.
- File tốt nhất có tên `<experiment>.pt`.
- File gần nhất để resume có tên `<experiment>.last.pt`.
- Cấu trúc đặt tên này được tạo ở `localagent/python/localagent/training/trainer.py:1891` và `:1894`.

## 23. Ý nghĩa của `localagent/artifacts/cache/`

- Đây là nơi lưu ảnh đã resize trước cho training.
- Cache nằm dưới `training/<image_size>px` hoặc `training/<image_size>px-raw`.
- Hàm quyết định tên thư mục cache nằm ở `localagent/python/localagent/training/trainer.py:1944`.
- Cache giúp giảm chi phí decode và resize lặp lại theo epoch.

## 24. Ý nghĩa của `localagent/artifacts/jobs/`

- Đây là nơi server Rust lưu metadata job đã spawn.
- Mỗi job có một file JSON.
- Log stdout và stderr nằm dưới `localagent/artifacts/jobs/logs/`.
- Cơ chế này được quản lý ở `localagent/src/jobs/runtime.rs`.

## 25. Ý nghĩa của `localagent/python/localagent/`

- Đây là package Python chính.
- Các module quan trọng nằm trong `data/`, `training/`, `vision/`, `bridge/`, `api/`, `services/`.

## 26. Ý nghĩa của `localagent/src/`

- Đây là source Rust.
- Nơi này chứa server Actix.
- Nơi này chứa bridge PyO3.
- Nơi này chứa inference ONNX bằng Rust.
- Nơi này chứa job manager.
- Nơi này chứa artifact store.
- Nơi này chứa workflow state cho UI.

## 27. Ý nghĩa của `localagent/tests/`

- Đây là nơi mô tả hành vi đang được test.
- Khi muốn hiểu hành vi mong đợi mà không muốn đọc toàn bộ implementation, hãy đọc test trước.

## 28. File Python khởi đầu nên đọc đầu tiên

- `localagent/python/localagent/config/settings.py`
- `localagent/python/localagent/data/pipeline.py`
- `localagent/python/localagent/data/discovery.py`
- `localagent/python/localagent/training/train.py`
- `localagent/python/localagent/training/trainer.py`
- `localagent/python/localagent/training/manifest_dataset.py`
- `localagent/python/localagent/vision/transforms.py`
- `localagent/python/localagent/bridge/rust_acceleration.py`
- `localagent/python/localagent/bridge/rust_backend.py`

## 29. File Rust khởi đầu nên đọc đầu tiên

- `localagent/src/bin/server.rs`
- `localagent/src/workflow.rs`
- `localagent/src/jobs/types.rs`
- `localagent/src/jobs/commands.rs`
- `localagent/src/jobs/runtime.rs`
- `localagent/src/artifacts.rs`
- `localagent/src/cluster_review.rs`
- `localagent/src/inference.rs`
- `localagent/src/python_api.rs`

## 30. File UI khởi đầu nên đọc đầu tiên

- `interface/next.config.ts`
- `interface/lib/localagent.ts`
- `interface/components/use-localagent-controller.ts`
- `interface/components/localagent/controller-actions.ts`
- `interface/components/dashboard/pipeline-studio-page.tsx`
- `interface/components/dashboard/discovery-studio.tsx`
- `interface/components/dashboard/training-studio.tsx`

## 31. Điểm vào cấu hình mặc định trong Python

- `AgentPaths` ở `localagent/python/localagent/config/settings.py:13`
- `RuntimeConfig` ở `localagent/python/localagent/config/settings.py:34`
- `TrainingConfig` ở `localagent/python/localagent/config/settings.py:55`
- `DatasetPipelineConfig` ở `localagent/python/localagent/config/settings.py:109`

## 32. `AgentPaths` dùng để làm gì

- Nó gom các path gốc mà agent Python hay dùng.
- Nó tạo `project_root`, `dataset_dir`, `model_dir`, `artifact_dir`, `log_dir`, `config_dir`.
- Phương thức `ensure_layout` tạo các thư mục này nếu chưa có.
- Đây là chỗ tiện để đảm bảo workspace có cấu trúc cơ bản.

## 33. `RuntimeConfig` dùng để làm gì

- Nó mô tả thông tin cần cho suy luận và server.
- Nó chứa `model_path`.
- Nó chứa `model_manifest_path`.
- Nó chứa `labels_path`.
- Nó chứa `artifact_dir`.
- Nó chứa `experiment_name`.
- Nó chứa `device`.
- Nó chứa `score_threshold`.
- Nó chứa `server_host`.
- Nó chứa `server_port`.

## 34. `TrainingConfig` dùng để làm gì

- Nó mô tả toàn bộ cấu hình training và export.
- Đây là dataclass quan trọng nhất cho trainer.
- Nếu bạn muốn chỉnh backbone, batch size, early stopping, cache format, pseudo-label threshold hoặc ONNX opset, bạn đang thực chất chỉnh `TrainingConfig`.

## 35. `DatasetPipelineConfig` dùng để làm gì

- Nó mô tả cấu hình quét ảnh và sinh manifest.
- Nó chứa path dữ liệu đầu vào.
- Nó chứa path report.
- Nó chứa ngưỡng kích thước ảnh nhỏ nhất.
- Nó chứa tỉ lệ train/val/test.
- Nó chứa random seed.
- Nó chứa `unknown_label`.
- Nó chứa flag có suy nhãn từ filename hay không.

## 36. Điểm vào CLI pipeline dữ liệu

- Nằm ở `localagent/python/localagent/data/pipeline.py:1361`.
- Hàm `build_parser` định nghĩa subcommand.
- Hàm `build_config` dựng `DatasetPipelineConfig`.
- Hàm `main` dispatch subcommand.

## 37. Điểm vào CLI training

- Nằm ở `localagent/python/localagent/training/train.py:56`.
- Hàm `build_parser` định nghĩa subcommand training.
- Hàm `build_config` dựng `TrainingConfig`.
- Hàm `main` dispatch subcommand sang `WasteTrainer`.

## 38. Đối tượng trung tâm của pipeline dữ liệu

- Là class `DatasetPipeline`.
- Class này bắt đầu ở `localagent/python/localagent/data/pipeline.py:78`.
- Nếu bạn muốn hiểu manifest được tạo ra sao, hãy đọc class này.

## 39. Đối tượng trung tâm của huấn luyện

- Là class `WasteTrainer`.
- Class này bắt đầu ở `localagent/python/localagent/training/trainer.py:40`.
- Nếu bạn muốn hiểu model được build, dataloader được dựng, cache được dùng, checkpoint được ghi và ONNX được export ra sao, hãy đọc class này.

## 40. Đối tượng trung tâm của discovery

- Discovery logic nằm chủ yếu ở `localagent/python/localagent/data/discovery.py`.
- Không có class pipeline lớn ở đây.
- Thay vào đó có dataclass artifact và các hàm thuần như `extract_embeddings` và `cluster_embeddings`.

## 41. Đối tượng trung tâm của server

- Server HTTP nằm ở `localagent/src/bin/server.rs`.
- Đây là file bạn đọc khi muốn biết endpoint nào tồn tại.
- Đây cũng là file bạn đọc khi muốn map UI action sang HTTP route.

## 42. Đối tượng trung tâm của workflow gating

- `WorkflowState` nằm ở `localagent/src/workflow.rs`.
- Đây là nơi quyết định nút nào trong UI được phép bấm.
- Nó đọc `summary.json`, `dataset_manifest.csv`, `cluster_review.csv` và kiểm tra trạng thái step.

## 43. Đối tượng trung tâm của jobs

- `localagent/src/jobs/types.rs` định nghĩa request và record.
- `localagent/src/jobs/commands.rs` build argv cho CLI Python.
- `localagent/src/jobs/runtime.rs` spawn process `uv`, lưu log và cập nhật trạng thái job.

## 44. Đối tượng trung tâm của artifact API

- `ArtifactStore` nằm ở `localagent/src/artifacts.rs`.
- Nó đọc các file report JSON từ `artifacts/reports/`.
- Nó gom thành `overview`, `run_index`, `run_detail`, `dashboard_summary`, `benchmark_overview`.

## 45. Đối tượng trung tâm của cluster review API

- `ClusterReviewStore` nằm ở `localagent/src/cluster_review.rs`.
- Nó tải `cluster_review.csv`.
- Nó kiểm tra stale row.
- Nó lưu lại quyết định review theo cụm.

## 46. Đối tượng trung tâm của inference Rust

- `WasteClassifier` nằm ở `localagent/src/inference.rs`.
- Nó hỗ trợ classify batch stub và classify uploaded image thật bằng ONNX.
- Phần classify theo `sample_id` hiện tại còn là stub logic.
- Phần classify ảnh base64 dùng ONNX thật.

## 47. Đối tượng trung tâm của bridge Python sang Rust

- `RustAccelerationBridge` ở `localagent/python/localagent/bridge/rust_acceleration.py`.
- `RustBackendBridge` ở `localagent/python/localagent/bridge/rust_backend.py`.
- `python_api.rs` ở Rust đăng ký các hàm được expose sang Python.

## 48. Đối tượng trung tâm của UI fetch

- `fetchJson` nằm ở `interface/lib/localagent.ts:272`.
- Đây là wrapper fetch tiêu chuẩn của UI.
- Hầu hết component và hook đi qua đây.

## 49. Rewrite từ UI sang server cục bộ

- `interface/next.config.ts` rewrite `/api/localagent/:path*` sang `http://127.0.0.1:8080/:path*` theo mặc định.
- Vì vậy UI fetch vào `/api/localagent/...`.
- Nhưng server thật đang lắng nghe ở port `8080`.

## 50. Hook quản lý trạng thái UI

- `useLocalAgentController` nằm ở `interface/components/use-localagent-controller.ts`.
- Hook này giữ `runs`, `jobs`, `workflowState`, `clusterReview`, `trainingForm`, `pipelineForm`.
- Hook này mở WebSocket log stream.
- Hook này refresh dữ liệu khi job hoàn tất.

## 51. Action UI gửi request

- `useLocalAgentActions` nằm ở `interface/components/localagent/controller-actions.ts`.
- Hàm `submitPipeline` bắn `POST /jobs/pipeline`.
- Hàm `submitTraining` bắn `POST /jobs/training` hoặc `/jobs/benchmark`.
- Hàm `saveClusterReview` bắn `PUT /cluster-review`.
- Hàm `loadRuns`, `loadRunDetail`, `loadWorkflowState`, `loadClusterReview` đọc các API report.

## 52. Chức năng scan dữ liệu nằm ở đâu

- `DatasetPipeline.scan` ở `localagent/python/localagent/data/pipeline.py:82`.
- `DatasetPipeline.run_scan` ở `localagent/python/localagent/data/pipeline.py:100`.

## 53. Chức năng chia split nằm ở đâu

- `DatasetPipeline.assign_splits` ở `localagent/python/localagent/data/pipeline.py:105`.
- `DatasetPipeline.run_split` ở `localagent/python/localagent/data/pipeline.py:152`.

## 54. Chức năng sinh report dữ liệu nằm ở đâu

- `DatasetPipeline.generate_reports` ở `localagent/python/localagent/data/pipeline.py:158`.
- `DatasetPipeline.run_report` ở `localagent/python/localagent/data/pipeline.py:205`.

## 55. Chức năng embedding nằm ở đâu

- `DatasetPipeline.embed_dataset` ở `localagent/python/localagent/data/pipeline.py:209`.
- `extract_embeddings` ở `localagent/python/localagent/data/discovery.py:72`.
- `_extract_pretrained_embeddings` ở `localagent/python/localagent/data/discovery.py:163`.
- `_extract_handcrafted_embeddings` ở `localagent/python/localagent/data/discovery.py:206`.

## 56. Chức năng clustering nằm ở đâu

- `DatasetPipeline.cluster_dataset` ở `localagent/python/localagent/data/pipeline.py:230`.
- `cluster_embeddings` ở `localagent/python/localagent/data/discovery.py:112`.
- `_spherical_kmeans` ở `localagent/python/localagent/data/discovery.py:266`.
- `_detect_cluster_outliers` ở `localagent/python/localagent/data/discovery.py:301`.

## 57. Chức năng export cluster review nằm ở đâu

- `DatasetPipeline.export_cluster_review` ở `localagent/python/localagent/data/pipeline.py:276`.
- `_build_cluster_review_rows` ở `localagent/python/localagent/data/pipeline.py:950`.

## 58. Chức năng promote cluster labels nằm ở đâu

- `DatasetPipeline.promote_cluster_labels` ở `localagent/python/localagent/data/pipeline.py:318`.
- `_load_cluster_review_assignments` ở `localagent/python/localagent/data/pipeline.py:1239`.

## 59. Chức năng export template gán nhãn thủ công nằm ở đâu

- `DatasetPipeline.export_labeling_template` ở `localagent/python/localagent/data/pipeline.py:397`.

## 60. Chức năng import label thủ công nằm ở đâu

- `DatasetPipeline.import_labels` ở `localagent/python/localagent/data/pipeline.py:464`.
- `_load_label_assignments` ở `localagent/python/localagent/data/pipeline.py:1192`.

## 61. Chức năng validate labels nằm ở đâu

- `DatasetPipeline.validate_labels` ở `localagent/python/localagent/data/pipeline.py:535`.

## 62. Chức năng workflow state mức Python nằm ở đâu

- `DatasetPipeline.workflow_state` ở `localagent/python/localagent/data/pipeline.py:582`.
- Tuy nhiên UI runtime thực tế đọc workflow từ Rust `WorkflowState`.

## 63. Chức năng build model nằm ở đâu

- `WasteTrainer.build_model_stub` ở `localagent/python/localagent/training/trainer.py:128`.
- `_replace_classifier_head` ở `:192`.
- `_freeze_model_backbone` ở `:208`.

## 64. Chức năng build dataset train/val/test nằm ở đâu

- `WasteTrainer.build_datasets` ở `localagent/python/localagent/training/trainer.py:290`.
- `ManifestImageDataset` nằm ở `localagent/python/localagent/training/manifest_dataset.py:11`.

## 65. Chức năng build dataloader nằm ở đâu

- `WasteTrainer.build_dataloaders` ở `localagent/python/localagent/training/trainer.py:312`.

## 66. Chức năng export label index nằm ở đâu

- `WasteTrainer.export_label_index` ở `localagent/python/localagent/training/trainer.py:355`.

## 67. Chức năng export experiment spec nằm ở đâu

- `WasteTrainer.export_experiment_spec` ở `localagent/python/localagent/training/trainer.py:372`.
- `ExperimentSpec` dataclass nằm ở `localagent/python/localagent/training/benchmarking.py:18`.

## 68. Chức năng benchmark nằm ở đâu

- `WasteTrainer.benchmark` ở `localagent/python/localagent/training/trainer.py:385`.
- `compare_benchmark_reports` ở `localagent/python/localagent/training/benchmarking.py:165`.

## 69. Chức năng evaluate nằm ở đâu

- `WasteTrainer.evaluate` ở `localagent/python/localagent/training/trainer.py:869`.
- `_build_classification_report` ở `:678`.
- `_write_evaluation_artifacts` ở `:749`.

## 70. Chức năng export ONNX nằm ở đâu

- `WasteTrainer.export_onnx` ở `localagent/python/localagent/training/trainer.py:959`.
- `_build_model_manifest` ở `:1831`.

## 71. Chức năng pseudo-label nằm ở đâu

- `WasteTrainer.pseudo_label` ở `localagent/python/localagent/training/trainer.py:1126`.
- `_pseudo_label_candidates` ở `:1709`.

## 72. Chức năng fit nằm ở đâu

- `WasteTrainer.fit` ở `localagent/python/localagent/training/trainer.py:1270`.
- `_run_epoch` ở `:2087`.
- `_should_stop_early` ở `:2243`.

## 73. Chức năng warm cache nằm ở đâu

- `WasteTrainer.warm_image_cache` ở `localagent/python/localagent/training/trainer.py:1569`.
- `RustAccelerationBridge.prepare_image_cache` ở `localagent/python/localagent/bridge/rust_acceleration.py`.
- `prepare_image_cache` PyO3 binding ở `localagent/src/python_api.rs:84`.

## 74. Chức năng transform ảnh nằm ở đâu

- `build_training_transforms` ở `localagent/python/localagent/vision/transforms.py`.
- `load_rgb_image` cũng nằm cùng file.

## 75. Chức năng server health nằm ở đâu

- `GET /health` ở `localagent/src/bin/server.rs`.

## 76. Chức năng classify batch sample id nằm ở đâu

- `POST /classify` ở `localagent/src/bin/server.rs`.
- `WasteClassifier.classify_batch` ở `localagent/src/inference.rs`.

## 77. Chức năng classify ảnh base64 nằm ở đâu

- `POST /classify/image` ở `localagent/src/bin/server.rs:113`.
- `WasteClassifier.classify_uploaded_image` ở `localagent/src/inference.rs`.

## 78. Chức năng list job nằm ở đâu

- `GET /jobs` ở `localagent/src/bin/server.rs`.
- `JobManager.list_jobs` ở `localagent/src/jobs/runtime.rs`.

## 79. Chức năng spawn dataset pipeline job nằm ở đâu

- `POST /jobs/pipeline` ở `localagent/src/bin/server.rs:149`.
- `spawn_pipeline_job` ở `localagent/src/jobs/commands.rs`.

## 80. Chức năng spawn training job nằm ở đâu

- `POST /jobs/training` ở `localagent/src/bin/server.rs:162`.
- `spawn_training_job` ở `localagent/src/jobs/commands.rs`.

## 81. Chức năng spawn benchmark job nằm ở đâu

- `POST /jobs/benchmark` ở `localagent/src/bin/server.rs:175`.
- `spawn_benchmark_job` ở `localagent/src/jobs/commands.rs`.

## 82. Chức năng cancel job nằm ở đâu

- `POST /jobs/{job_id}/cancel` ở `localagent/src/bin/server.rs:188`.
- `JobManager.cancel_job` ở `localagent/src/jobs/runtime.rs`.

## 83. Chức năng đọc log job nằm ở đâu

- `GET /jobs/{job_id}/logs` ở `localagent/src/bin/server.rs:199`.
- `JobManager.job_logs` ở `localagent/src/jobs/runtime.rs`.

## 84. Chức năng stream log qua WebSocket nằm ở đâu

- `GET /ws/jobs` ở `localagent/src/bin/server.rs`.
- `buildJobsWebSocketUrl` ở `interface/lib/localagent.ts`.
- `useLocalAgentController` mở WebSocket này ở `interface/components/use-localagent-controller.ts`.

## 85. Chức năng run catalog nằm ở đâu

- `GET /runs` ở `localagent/src/bin/server.rs`.
- `ArtifactStore.run_index` ở `localagent/src/artifacts.rs`.

## 86. Chức năng run detail nằm ở đâu

- `GET /runs/{experiment_name}` ở `localagent/src/bin/server.rs:242`.
- `ArtifactStore.run_detail` ở `localagent/src/artifacts.rs`.

## 87. Chức năng compare runs nằm ở đâu

- `GET /runs/{experiment_name}/compare?with=...` ở `localagent/src/bin/server.rs:265`.
- `ArtifactStore.compare_runs` ở `localagent/src/artifacts.rs`.

## 88. Chức năng training presets nằm ở đâu

- `GET /presets/training` ở `localagent/src/bin/server.rs:281`.
- Preset Python CLI song song nằm ở `localagent/python/localagent/training/train.py`.

## 89. Chức năng pipeline presets nằm ở đâu

- `GET /presets/pipeline` ở `localagent/src/bin/server.rs:311`.

## 90. Chức năng workflow state nằm ở đâu

- `GET /workflow/state` ở `localagent/src/bin/server.rs:435`.
- `WorkflowState::from_config` ở `localagent/src/workflow.rs`.

## 91. Chức năng đọc ảnh dataset cho UI nằm ở đâu

- `GET /dataset/image` ở `localagent/src/bin/server.rs:448`.
- URL được dựng bởi `buildDatasetImageUrl` ở `interface/components/dashboard/discovery/discovery-shared.ts`.

## 92. Chức năng cluster review API nằm ở đâu

- `GET /cluster-review` ở `localagent/src/bin/server.rs:523`.
- `PUT /cluster-review` ở `localagent/src/bin/server.rs:535`.
- Logic lưu và kiểm stale nằm ở `localagent/src/cluster_review.rs`.

## 93. Chức năng artifact dataset API nằm ở đâu

- `GET /artifacts/dataset` ở `localagent/src/bin/server.rs`.

## 94. Chức năng artifact training API nằm ở đâu

- `GET /artifacts/training` ở `localagent/src/bin/server.rs`.
- `GET /artifacts/training-overview` ở `localagent/src/bin/server.rs`.

## 95. Chức năng artifact evaluation API nằm ở đâu

- `GET /artifacts/evaluation` ở `localagent/src/bin/server.rs`.

## 96. Chức năng artifact export API nằm ở đâu

- `GET /artifacts/export` ở `localagent/src/bin/server.rs`.

## 97. Chức năng artifact benchmark API nằm ở đâu

- `GET /artifacts/benchmark` ở `localagent/src/bin/server.rs`.
- `GET /artifacts/benchmarks` ở `localagent/src/bin/server.rs`.

## 98. Chức năng artifact experiment spec API nằm ở đâu

- `GET /artifacts/experiment-spec` ở `localagent/src/bin/server.rs`.

## 99. Chức năng artifact model manifest API nằm ở đâu

- `GET /artifacts/model-manifest` ở `localagent/src/bin/server.rs`.

## 100. Chức năng dashboard summary API nằm ở đâu

- `GET /dashboard/summary` ở `localagent/src/bin/server.rs`.

## 101. Công nghệ Python chính

- `torch`
- `torchvision`
- `polars`
- `numpy`
- `opencv-python-headless`
- `onnx`
- `onnxruntime`
- `pytest`
- `ruff`
- `httpx`

## 102. Công nghệ Rust chính

- `actix-web`
- `actix-ws`
- `tokio`
- `rayon`
- `serde`
- `serde_json`
- `pyo3`
- `ort`

## 103. Công nghệ UI chính

- `Next.js`
- `TypeScript`
- `React`
- `fetch`
- `WebSocket`

## 104. Điều dự án chưa làm thật nhưng người đọc dễ tưởng là đã có

- Chưa có fuzzy logic đúng nghĩa trong code hiện tại.
- Chưa có fuzzy c-means trong code hiện tại.
- Chưa có inference batch theo `sample_id` dùng model thật.
- Chưa có backend training `rust_tch` hoàn chỉnh cho `fit`.
- Chưa có full config TOML runtime được wiring cho toàn pipeline theo kiểu production.

## 105. Vì sao phải nói rõ chuyện fuzzy logic

- Người đọc rất dễ thấy chữ “logic mờ” trong ý tưởng bài toán rồi suy ra code hiện tại đã có fuzzy logic.
- Điều đó không đúng.
- Code hiện tại dùng embedding từ CNN.
- Sau đó dùng `spherical k-means`.
- Sau đó dùng median absolute deviation để phát hiện outlier.
- Sau đó dùng review của con người ở mức cluster.
- Sau đó có thể dùng pseudo-label từ model.
- Nếu bạn muốn fuzzy logic thật sự, bạn cần thêm implementation mới.

## 106. Vị trí chứng minh code hiện tại không dùng fuzzy logic

- `localagent/python/localagent/data/discovery.py:112` có `cluster_embeddings`.
- `localagent/python/localagent/data/discovery.py:266` có `_spherical_kmeans`.
- File này không có fuzzy membership matrix.
- File này không có hệ luật mờ.
- File này không có hàm suy diễn fuzzy.
- File này không có centroid update theo công thức fuzzy c-means.

## 107. Vị trí chứng minh code hiện tại dùng CNN embedding

- `localagent/python/localagent/data/discovery.py:163` load `resnet18`.
- `model.fc = torch.nn.Identity()` biến ResNet18 thành extractor feature.
- Vector output sau đó được chuẩn hóa bằng chuẩn L2.

## 108. Vị trí chứng minh training hiện tại dùng PyTorch

- `localagent/python/localagent/training/trainer.py:767` trả backend capability.
- Khi backend là `pytorch`, trainer báo `supported=True`.
- Khi backend là `rust_tch`, trainer báo `supported=False`.
- `localagent/src/jobs/commands.rs` cũng chặn `fit`, `evaluate`, `export-onnx`, `pseudo-label` nếu backend khác `pytorch`.

## 109. Vị trí chứng minh ONNX export có verify

- `localagent/python/localagent/training/trainer.py:959` là `export_onnx`.
- Trong hàm này có đoạn dùng `onnxruntime` để so sánh logits ONNX với logits PyTorch.
- Report export thực tế nằm ở `localagent/artifacts/reports/baseline-waste-sorter_export.json`.
- `model_manifest.json` cũng lưu trạng thái verify này.

## 110. Vị trí chứng minh UI gọi API qua rewrite

- `interface/next.config.ts` rewrite `/api/localagent/:path*`.
- `interface/lib/localagent.ts:1` khai báo `API_PREFIX = "/api/localagent"`.
- `interface/lib/localagent.ts:272` dùng `fetch(`${API_PREFIX}${path}`)`.

## 111. Đọc code theo thứ tự nào là hợp lý nhất

- Bắt đầu từ file tài liệu này.
- Sau đó đọc `localagent/python/localagent/config/settings.py`.
- Sau đó đọc `localagent/python/localagent/data/pipeline.py`.
- Sau đó đọc `localagent/python/localagent/data/discovery.py`.
- Sau đó đọc `localagent/python/localagent/training/train.py`.
- Sau đó đọc `localagent/python/localagent/training/trainer.py`.
- Sau đó đọc `localagent/src/bin/server.rs`.
- Sau đó đọc `interface/lib/localagent.ts`.
- Sau đó đọc `interface/components/use-localagent-controller.ts`.
- Sau đó đọc test.

## 112. Nếu bạn chỉ muốn chạy pipeline mà chưa muốn đọc sâu code

- Vào `localagent/`.
- Chạy `uv sync --dev`.
- Chạy `uv run maturin develop`.
- Chạy `uv run python -m localagent.data.pipeline run-all`.
- Chạy `uv run python -m localagent.data.pipeline embed`.
- Chạy `uv run python -m localagent.data.pipeline cluster`.
- Chạy `uv run python -m localagent.data.pipeline export-cluster-review`.
- Duyệt `artifacts/manifests/cluster_review.csv`.
- Chạy `uv run python -m localagent.data.pipeline promote-cluster-labels --review-file artifacts/manifests/cluster_review.csv`.
- Chạy `uv run python -m localagent.training.train summary`.
- Chạy `uv run python -m localagent.training.train fit`.
- Chạy `uv run python -m localagent.training.train evaluate`.
- Chạy `uv run python -m localagent.training.train export-onnx`.

## 113. Nếu bạn chỉ muốn xem UI

- Chạy server Rust trong `localagent/`.
- Chạy UI trong `interface/`.
- UI sẽ gọi server local qua rewrite.

## 114. Nếu bạn chỉ muốn xem artifact đã có sẵn

- Mở `localagent/artifacts/reports/summary.json`.
- Mở `localagent/artifacts/reports/baseline-waste-sorter_training.json`.
- Mở `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json`.
- Mở `localagent/artifacts/reports/baseline-waste-sorter_export.json`.
- Mở `localagent/models/model_manifest.json`.

## 115. Nếu bạn chỉ muốn biết dự án đang phân loại bao nhiêu lớp

- Hiện tại artifact export cho thấy ba lớp là `folk`, `glass`, `paper`.
- Nguồn xác nhận là `localagent/models/labels.json`.
- Nguồn xác nhận thứ hai là `localagent/models/model_manifest.json`.

## 116. Nếu bạn chỉ muốn biết kết quả gần nhất

- Accuracy test gần nhất khoảng `92.96%`.
- Macro F1 gần nhất khoảng `0.8074`.
- Weighted F1 gần nhất khoảng `0.9366`.
- Các số này lấy từ `localagent/artifacts/reports/baseline-waste-sorter_evaluation.json`.

## 117. Nếu bạn chỉ muốn biết training mất bao lâu

- Benchmark report gần nhất cho thấy tổng duration khoảng `246.90` giây.
- Stage `fit` chiếm khoảng `241.47` giây.
- Stage `evaluate` chiếm khoảng `3.96` giây.
- Stage `export_onnx` chiếm khoảng `1.38` giây.
- Nguồn là `localagent/artifacts/reports/baseline-waste-sorter_benchmark.json`.

## 118. Nếu bạn chỉ muốn biết current dataset imbalance

- `train_imbalance_ratio` gần nhất khoảng `18.92`.
- Điều đó nghĩa là lớp lớn nhất và lớp nhỏ nhất chênh lệch gần 19 lần.
- Vì vậy class weighting là rất quan trọng.
- Nguồn là `localagent/artifacts/reports/baseline-waste-sorter_training.json`.

## 119. Vì sao người mới hay bị nhầm khái niệm “accepted labels”

- Vì manifest có thể chứa nhãn suy từ filename.
- Nhưng khi trong manifest đã xuất hiện nhãn được chấp nhận từ `curated`, `cluster_review` hoặc `model_pseudo`, trainer chuyển sang chế độ `accepted_labels_only`.
- Khi đó nhãn suy từ filename chỉ còn là weak hint.
- Đây là logic cực kỳ quan trọng của dự án.

## 120. Vị trí code của logic `accepted_labels_only`

- `DatasetPipeline._effective_training_mode` ở `localagent/python/localagent/data/pipeline.py:909`.
- `DatasetPipeline._has_accepted_labels` ở `:912`.
- `DatasetPipeline._training_ready_frame` ở `:1174`.
- `WasteTrainer._has_accepted_labels` ở `localagent/python/localagent/training/trainer.py:1690`.
- `WasteTrainer._effective_training_mode` ở `:1702`.
- `WasteTrainer._labeled_frame` ở `:1666`.

## 121. File nào chứng minh hành vi pipeline qua test

- `localagent/tests/test_pipeline.py`
- `localagent/tests/test_training_data.py`
- `localagent/tests/test_training_fit.py`
- `localagent/tests/test_training_pseudo_label.py`
- `localagent/tests/test_training_artifacts.py`
- `localagent/tests/test_train_cli.py`
- `localagent/tests/test_settings.py`

## 122. Một số test quan trọng nên đọc

- `test_scan_marks_invalid_small_and_duplicate_images`
- `test_run_all_writes_manifest_and_reports`
- `test_embed_cluster_and_export_cluster_review`
- `test_promote_cluster_labels_switches_manifest_to_accepted_labels_only_mode`
- `test_fit_stops_early_when_validation_loss_stalls`
- `test_fit_can_resume_from_latest_checkpoint`
- `test_pseudo_label_updates_manifest_with_confidence_gate`
- `test_export_onnx_writes_manifest_and_export_report`
- `test_benchmark_writes_report_for_pytorch_backend`

## 123. Ý nghĩa kỹ thuật của việc có test

- Test cho biết hành vi mong đợi của code.
- Test cho biết command nào phải bị khóa ở bước nào.
- Test cho biết pseudo-label phải qua cổng confidence và margin.
- Test cho biết cluster review stale row phải bị reset hoặc bị chặn.

## 124. Các tài liệu kế tiếp nên đọc theo thứ tự trong thư mục `docs/`

- `01-pipeline-du-lieu-va-manifest.md`
- `02-discovery-embedding-clustering-va-cluster-review.md`
- `03-huan-luyen-cnn-va-cau-hinh-theo-may.md`
- `04-pseudo-label-onnx-artifact-va-luu-tru-cuc-bo.md`
- `05-rust-server-api-fetch-va-giao-dien.md`
- `06-toan-hoc-ml-dl-va-cong-thuc.md`
- `07-workflow-thuc-chien-checklist-va-khac-phuc-su-co.md`

## 125. Ghi chú quan trọng về thư mục làm việc

- Hầu hết command phải chạy bên trong `localagent/`.
- Nếu chạy từ root, bạn phải `cd localagent` trước.
- UI thì chạy bên trong `interface/`.

## 126. Ghi chú quan trọng về artifact path trong tài liệu này

- Vì `docs/` nằm ở root repository, nhiều đường dẫn trong tài liệu được ghi đầy đủ theo dạng `localagent/...`.
- Khi bạn đang đứng trong `localagent/`, cùng một file sẽ tương ứng với path ngắn hơn như `artifacts/...`.

## 127. Ghi chú quan trọng về line reference

- Line number trong tài liệu này bám theo snapshot code hiện tại lúc viết tài liệu.
- Khi code thay đổi, line number có thể dịch chuyển.
- Nếu line number lệch nhẹ, hãy tìm theo tên hàm trước.

## 128. Thuật ngữ nhanh

- `manifest`: bảng mô tả toàn bộ sample và metadata của chúng.
- `artifact`: file đầu ra trung gian hoặc cuối cùng của pipeline.
- `weak label`: nhãn yếu, chưa đủ tin cậy để xem là ground truth sau khi có nhãn chấp nhận.
- `accepted label`: nhãn đã được chấp nhận từ con người hoặc từ cổng pseudo-label.
- `backbone`: phần CNN trích đặc trưng.
- `head`: lớp phân loại cuối cùng của mạng.
- `embedding`: vector biểu diễn ảnh.
- `cluster`: cụm ảnh giống nhau trong không gian embedding.
- `outlier`: phần tử khác biệt so với phần lớn trong cụm.
- `pseudo label`: nhãn do model hiện tại đề xuất cho dữ liệu chưa gán nhãn.

## 129. Từ điển tra cứu nhanh theo file

- `README.md`
- Vai trò: giải thích tổng quan cấp repository.
- Khi nào đọc: trước khi đọc source.
- `localagent/README.md`
- Vai trò: hướng dẫn tập trung vào localagent.
- Khi nào đọc: trước khi chạy command trong `localagent/`.
- `localagent/python/localagent/config/settings.py`
- Vai trò: default config và path.
- Khi nào đọc: khi muốn biết giá trị mặc định.
- `localagent/python/localagent/data/pipeline.py`
- Vai trò: scan, split, report, import labels, discovery gate.
- Khi nào đọc: khi muốn hiểu dữ liệu.
- `localagent/python/localagent/data/discovery.py`
- Vai trò: embedding và clustering.
- Khi nào đọc: khi muốn hiểu semi-supervised discovery.
- `localagent/python/localagent/training/train.py`
- Vai trò: CLI training.
- Khi nào đọc: khi muốn biết flag CLI.
- `localagent/python/localagent/training/trainer.py`
- Vai trò: trainer thực thi.
- Khi nào đọc: khi muốn hiểu training thật sự.
- `localagent/python/localagent/training/manifest_dataset.py`
- Vai trò: dataset đọc ảnh gốc hoặc cache.
- Khi nào đọc: khi muốn hiểu dataloader.
- `localagent/python/localagent/vision/transforms.py`
- Vai trò: normalize và resize.
- Khi nào đọc: khi muốn hiểu input tensor.
- `localagent/python/localagent/bridge/rust_acceleration.py`
- Vai trò: Python gọi Rust cache và metric.
- Khi nào đọc: khi muốn hiểu bridge tăng tốc.
- `localagent/python/localagent/bridge/rust_backend.py`
- Vai trò: Python gọi Rust inference backend.
- Khi nào đọc: khi muốn hiểu classify stub từ Python.
- `localagent/src/bin/server.rs`
- Vai trò: HTTP API và WebSocket API.
- Khi nào đọc: khi muốn hiểu UI gọi gì.
- `localagent/src/workflow.rs`
- Vai trò: step gating và unlock logic.
- Khi nào đọc: khi nút UI bị khóa.
- `localagent/src/jobs/types.rs`
- Vai trò: schema request và schema job.
- Khi nào đọc: khi muốn biết payload API.
- `localagent/src/jobs/commands.rs`
- Vai trò: map request JSON sang argv CLI Python.
- Khi nào đọc: khi muốn biết UI action sẽ spawn command gì.
- `localagent/src/jobs/runtime.rs`
- Vai trò: spawn process, log, cancel, persist job.
- Khi nào đọc: khi muốn debug job.
- `localagent/src/artifacts.rs`
- Vai trò: đọc report và tạo dashboard payload.
- Khi nào đọc: khi API artifact trả sai.
- `localagent/src/cluster_review.rs`
- Vai trò: load/save cluster review.
- Khi nào đọc: khi UI review cụm lỗi stale.
- `localagent/src/inference.rs`
- Vai trò: ONNX inference bằng Rust.
- Khi nào đọc: khi muốn hiểu classify ảnh.
- `interface/next.config.ts`
- Vai trò: rewrite API.
- Khi nào đọc: khi UI không gọi tới server được.
- `interface/lib/localagent.ts`
- Vai trò: type và fetch helper.
- Khi nào đọc: khi muốn biết contract giữa UI và server.
- `interface/components/use-localagent-controller.ts`
- Vai trò: orchestration state phía client.
- Khi nào đọc: khi debug UI refresh.
- `interface/components/localagent/controller-actions.ts`
- Vai trò: fire API requests.
- Khi nào đọc: khi debug submit job.

## 130. Cảnh báo về hiểu nhầm phổ biến

- Không phải cứ có Rust trong repo là training được viết bằng Rust.
- Không phải cứ có `score_threshold` là batch classify sample id đang dùng model thật.
- Không phải cứ có `datasets/` là pipeline mặc định quét ở đó.
- Không phải cứ có pseudo-label là mọi mẫu chưa nhãn đều được chấp nhận.
- Không phải cứ có cluster là cluster đó đã được người dùng xác nhận.
- Không phải cứ có `cluster_review.csv` là training luôn được mở khóa.

## 131. FAQ ngắn cho người mới

- Hỏi: Tôi nên bắt đầu từ thư mục nào.
- Đáp: Bắt đầu từ `localagent/` vì đây là lõi.
- Hỏi: Tôi có cần chạy UI để train model không.
- Đáp: Không cần, CLI Python đã đủ.
- Hỏi: UI có gọi Python trực tiếp không.
- Đáp: Không, UI gọi Rust server.
- Hỏi: Rust server có train model không.
- Đáp: Không, Rust server spawn CLI Python.
- Hỏi: Có fuzzy logic trong code không.
- Đáp: Không có implementation fuzzy logic thật trong snapshot hiện tại.
- Hỏi: Có clustering trong code không.
- Đáp: Có, bằng spherical k-means.
- Hỏi: Có outlier detection trong code không.
- Đáp: Có, bằng median absolute deviation hoặc quantile fallback.
- Hỏi: Có pseudo-label không.
- Đáp: Có.
- Hỏi: Có export ONNX không.
- Đáp: Có.
- Hỏi: Có verify ONNX không.
- Đáp: Có.
- Hỏi: Có benchmark report không.
- Đáp: Có.
- Hỏi: Có checkpoint resume không.
- Đáp: Có.
- Hỏi: Có log job thời gian thực không.
- Đáp: Có, qua WebSocket.
- Hỏi: Có API xem artifact không.
- Đáp: Có.
- Hỏi: Có test cho pipeline không.
- Đáp: Có.
- Hỏi: Có test cho pseudo-label không.
- Đáp: Có.
- Hỏi: Có test cho benchmark không.
- Đáp: Có.
- Hỏi: Có backend `rust_tch` hoàn chỉnh không.
- Đáp: Chưa.
- Hỏi: Có classify ảnh upload bằng ONNX Rust không.
- Đáp: Có.
- Hỏi: Có classify sample id bằng ONNX thật không.
- Đáp: Chưa, phần đó còn stub.

## 132. Kết luận của file này

- Dự án có kiến trúc ba lớp rõ ràng.
- Python là lõi pipeline dữ liệu và huấn luyện.
- Rust là lớp server, bridge và tăng tốc phụ trợ.
- UI là lớp điều khiển.
- Discovery hiện tại dùng CNN embedding cộng với spherical k-means.
- Dự án hiện chưa có fuzzy logic thật sự.
- Artifact đang tồn tại đầy đủ để bạn xem một workflow hoàn chỉnh.
- Các file tiếp theo trong `docs/` sẽ đi sâu từng phần một.
