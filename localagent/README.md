# Local Smart Waste Agent

`localagent` là phần lõi của dự án Smart Waste Sorting.

Hiện tại module này hỗ trợ:

- quét dataset ảnh thô trong `dataset/`
- tạo manifest Polars
- suy nhãn từ tên file
- phát hiện ảnh lỗi, ảnh quá nhỏ, ảnh trùng
- tạo split `train/val/test`
- xuất báo cáo dữ liệu
- warm cache ảnh bằng Rust để train nhanh hơn
- huấn luyện baseline trực tiếp từ manifest với progress bar

## Cấu trúc quan trọng

```text
localagent/
├── dataset/
├── datasets/
├── artifacts/
│   ├── manifests/
│   ├── reports/
│   ├── checkpoints/
│   └── cache/
├── models/
├── python/localagent/
├── src/
└── tests/
```

## Lệnh thường dùng

```powershell
cd localagent
uv sync --dev
uv run maturin develop
uv run python -m localagent.data.pipeline run-all
uv run python -m localagent.data.pipeline export-labeling-template
uv run python -m localagent.data.pipeline import-labels --labels-file artifacts/manifests/labeling_template.csv
uv run python -m localagent.data.pipeline validate-labels
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train warm-cache
uv run python -m localagent.training.train fit
uv run python -m localagent.training.train evaluate
uv run python -m localagent.training.train export-onnx
uv run python -m localagent.training.train report
```

## Hai pipeline đang có

### 1. Pipeline dữ liệu

Mục tiêu:

- quét ảnh thô trong `dataset/`
- suy nhãn từ tên file
- phát hiện ảnh lỗi, ảnh trùng, ảnh quá nhỏ
- chia `train/val/test`
- sinh manifest và báo cáo

Lệnh:

```powershell
cd localagent
uv run python -m localagent.data.pipeline run-all
```

Khi ảnh thô chưa có nhãn chuẩn, dùng thêm:

```powershell
uv run python -m localagent.data.pipeline export-labeling-template
uv run python -m localagent.data.pipeline import-labels --labels-file artifacts/manifests/labeling_template.csv
uv run python -m localagent.data.pipeline validate-labels
```

Đầu ra chính:

- `artifacts/manifests/dataset_manifest.parquet`
- `artifacts/manifests/dataset_manifest.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/split_summary.csv`
- `artifacts/reports/quality_summary.csv`
- `artifacts/reports/extension_summary.csv`
- `artifacts/reports/label_summary.csv`

### 2. Pipeline huấn luyện

Mục tiêu:

- đọc manifest
- warm cache ảnh bằng Rust
- train baseline từ manifest bằng `MobileNetV3-Small`
- ghi checkpoint và labels

Lệnh:

```powershell
cd localagent
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train warm-cache
uv run python -m localagent.training.train fit
uv run python -m localagent.training.train evaluate
uv run python -m localagent.training.train export-onnx
uv run python -m localagent.training.train report
```

Một số cờ hữu ích:

```powershell
uv run python -m localagent.training.train fit --experiment-name baseline-waste-sorter-e15-cpu --epochs 15
uv run python -m localagent.training.train fit --cache-format raw --class-bias both --epochs 1000 --disable-early-stopping
uv run python -m localagent.training.train fit --experiment-name waste-e25-fast --training-preset cpu_fast --epochs 25
uv run python -m localagent.training.train fit --experiment-name waste-e25-balanced --training-preset cpu_balanced --epochs 25
uv run python -m localagent.training.train fit --experiment-name waste-e25-stronger --training-preset cpu_stronger --epochs 25
uv run python -m localagent.training.train fit --experiment-name waste-e25-fast --training-preset cpu_fast --epochs 25 --resume-from artifacts/checkpoints/waste-e25-fast.last.pt
uv run python -m localagent.training.train fit --experiment-name waste-e25-balanced --training-preset cpu_balanced --epochs 25 --resume-from artifacts/checkpoints/waste-e25-balanced.last.pt
uv run python -m localagent.training.train fit --experiment-name waste-e25-stronger --training-preset cpu_stronger --epochs 25 --resume-from artifacts/checkpoints/waste-e25-stronger.last.pt
uv run python -m localagent.training.train fit --model-name mobilenet_v3_small --cache-format raw --class-bias loss --epochs 25
uv run python -m localagent.training.train fit --model-name mobilenet_v3_large --cache-format raw --class-bias loss --epochs 25
uv run python -m localagent.training.train fit --model-name resnet18 --cache-format raw --class-bias loss --epochs 25
uv run python -m localagent.training.train fit --model-name efficientnet_b0 --cache-format raw --class-bias loss --epochs 25
uv run python -m localagent.training.train fit --no-pretrained
uv run python -m localagent.training.train fit --train-backbone
uv run python -m localagent.training.train fit --class-bias loss
uv run python -m localagent.training.train fit --class-bias sampler
uv run python -m localagent.training.train fit --class-bias both
uv run python -m localagent.training.train fit --early-stopping-patience 2
```

Khi dùng `--no-progress`, trainer sẽ tắt progress bar dạng thanh nhưng vẫn in snapshot tiến độ theo batch và summary theo epoch để theo dõi các run dài.

Trainer hiện sẽ tự lưu:

- checkpoint tốt nhất ở `artifacts/checkpoints/<experiment_name>.pt`
- checkpoint resume mới nhất ở `artifacts/checkpoints/<experiment_name>.last.pt`
- benchmark theo lớp ở `artifacts/reports/<experiment_name>_evaluation.json`
- confusion matrix ở `artifacts/reports/<experiment_name>_confusion_matrix.csv`
- báo cáo train ở `artifacts/reports/<experiment_name>_training.json`
- báo cáo export ở `artifacts/reports/<experiment_name>_export.json`
- model manifest ở `models/model_manifest.json`

### 3. API artifact từ Actix

Server Rust hiện có thể đọc trực tiếp JSON artifact để interface gọi:

```powershell
cd localagent
cargo run --bin localagent-server
```

Các endpoint chính:

- `GET /health`
- `POST /classify`
- `GET /dashboard/summary?experiment=<name>`
- `GET /artifacts/overview?experiment=<name>`
- `GET /artifacts/training?experiment=<name>`
- `GET /artifacts/training-overview?experiment=<name>`
- `GET /artifacts/evaluation?experiment=<name>`
- `GET /artifacts/export?experiment=<name>`
- `GET /artifacts/benchmarks?experiment=<name>`
- `GET /artifacts/model-manifest`

### 4. Rust đang hỗ trợ gì cho training

- Rust hỗ trợ mạnh ở phần warm-cache ảnh, đọc artifact JSON và API benchmark/report.
- Rust hiện cũng hỗ trợ tính `class_weight_map` và classification report cho benchmark qua bridge.
- Phần forward/backward, update weight/bias theo epoch vẫn do PyTorch native backend đảm nhiệm.
- Nếu muốn Rust tham gia trực tiếp vào optimizer/backprop, bước tiếp theo sẽ là tách trainer sang stack như `tch`, `burn` hoặc `candle`, không chỉ thêm helper quanh pipeline hiện tại.

Khi resume, hãy giữ nguyên `--experiment-name` để trainer đọc đúng file `artifacts/checkpoints/<experiment_name>.last.pt`.

Rust hiện vẫn hỗ trợ chung cho các backbone CNN này ở phần warm-cache, đọc dữ liệu và pipeline I/O; phần forward/backward của CNN vẫn do PyTorch trong Python đảm nhiệm.

Preset đang có:

- `cpu_fast`: `mobilenet_v3_small`, `image_size=160`, `batch_size=32`, `cache_format=raw`, `class_bias=loss`
- `cpu_balanced`: `resnet18`, `image_size=224`, `batch_size=16`, `cache_format=raw`, `class_bias=loss`
- `cpu_stronger`: `efficientnet_b0`, `image_size=224`, `batch_size=8`, `cache_format=raw`, `class_bias=loss`

Tùy chọn test nhanh trên CPU:

```powershell
uv run python -m localagent.training.train fit --image-size 160 --epochs 3 --batch-size 32 --num-workers 0
```

Cache ảnh được ghi vào:

```text
artifacts/cache/training/<image_size>px/
```

Nếu có ảnh lỗi khi warm cache, báo cáo sẽ nằm ở:

```text
artifacts/reports/training_cache_failures_<image_size>px.json
```

Sau bước Rust cache, trainer sẽ tự fallback bằng OpenCV cho đúng các ảnh lỗi đó. Nếu cứu được, cache sẽ được bổ sung và report sẽ chỉ còn những ảnh thật sự chưa đọc được.

Progress bar hiện có ở:

- bước scan dataset
- bước warm cache ảnh bằng Rust
- từng epoch/batch khi train

Early stopping hiện được bật mặc định để không chạy thừa epoch khi validation loss ngừng cải thiện.

## Ghi chú trước khi test

- Nếu `summary` hiển thị `resolved_device` là `cpu` thì mô hình đang train bằng CPU. Điều này đúng với máy không có CUDA và sẽ chậm hơn đáng kể so với GPU.
- Rust hiện hỗ trợ chủ yếu ở bước build cache ảnh và giảm chi phí decode/resize lặp lại. Phần train tensor vẫn do PyTorch đảm nhiệm.
- Model mặc định hiện là `mobilenet_v3_small`, ưu tiên pretrained weights nếu tải được.
- Nếu pretrained weights không tải được, trainer sẽ tự fallback về random init thay vì dừng hẳn.
- Mặc định backbone bị freeze để train nhẹ hơn trên CPU. Dùng `--train-backbone` nếu bạn muốn fine-tune toàn bộ.
- Lần đầu chạy `warm-cache` có thể tốn thời gian. Những lần sau sẽ nhanh hơn nếu cache đã tồn tại và bạn không dùng `--force-cache`.
- `fit` sẽ tự gọi warm cache nếu Rust bridge sẵn sàng. Bạn vẫn có thể chạy `warm-cache` thủ công trước để kiểm tra riêng bước này.
- Nếu `warm-cache` báo lỗi, trainer sẽ tự thử cứu bằng OpenCV trước. Chỉ các lỗi còn lại sau bước đó mới được giữ trong `training_cache_failures_<image_size>px.json`.
- Nếu tên file không đủ rõ để suy nhãn, một phần dữ liệu sẽ bị gắn `unknown` và không được đưa vào `train/val/test`.
- Các ảnh bị lỗi, trùng hoặc quá nhỏ chỉ bị đánh dấu trong manifest; ảnh gốc vẫn giữ nguyên trong `dataset/`.
- Để test nhanh trên Windows, nên bắt đầu với `--image-size 160 --epochs 3 --batch-size 32 --num-workers 0`.
- Nếu bạn thay đổi `image_size`, cache sẽ được tạo sang thư mục khác tương ứng.

## Đầu ra chính

- `artifacts/manifests/dataset_manifest.parquet`
- `artifacts/manifests/dataset_manifest.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/label_summary.csv`
- `artifacts/cache/training/<image_size>px/`
- `models/labels.json`
- `artifacts/checkpoints/*.pt`

README đầy đủ của repository nằm ở thư mục root.
