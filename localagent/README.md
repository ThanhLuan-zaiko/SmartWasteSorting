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
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train warm-cache
uv run python -m localagent.training.train fit
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
```

Một số cờ hữu ích:

```powershell
uv run python -m localagent.training.train fit --no-pretrained
uv run python -m localagent.training.train fit --train-backbone
uv run python -m localagent.training.train fit --early-stopping-patience 2
```

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
