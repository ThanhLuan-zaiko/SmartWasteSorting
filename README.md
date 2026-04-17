# Smart Waste Sorting

## Giới thiệu

Đây là repository cho hệ thống phân loại rác thông minh. Ở trạng thái hiện tại, phần chạy chính là `localagent/`, còn `interface/` vẫn là giao diện mẫu để phát triển UI sau này.

Repository được tổ chức theo hướng:

- `localagent/`: lõi xử lý cục bộ bằng Python + Rust cho pipeline dữ liệu, huấn luyện, suy luận và backend nội bộ
- `interface/`: giao diện Next.js tham khảo, chưa phải trọng tâm vận hành ở giai đoạn này

Nếu bạn mới clone repository, hãy bắt đầu từ `localagent/`.

## Cấu trúc thư mục

```text
SmartWasteSorting/
├─ README.md
├─ .gitignore
├─ interface/
│  ├─ app/
│  ├─ package.json
│  ├─ bun.lock
│  └─ các tệp cấu hình Next.js / TypeScript
└─ localagent/
   ├─ pyproject.toml
   ├─ Cargo.toml
   ├─ uv.lock
   ├─ Cargo.lock
   ├─ dataset/
   ├─ datasets/
   ├─ configs/
   ├─ models/
   ├─ artifacts/
   │  ├─ manifests/
   │  ├─ reports/
   │  ├─ checkpoints/
   │  └─ cache/
   ├─ logs/
   ├─ python/localagent/
   ├─ src/
   └─ tests/
```

## Vai trò của từng phần

### `localagent/`

Đây là phần lõi của hệ thống phân loại rác.

- Python dùng `uv` để quản lý môi trường, dependency, pipeline dữ liệu, huấn luyện và test
- Rust dùng `cargo` để cung cấp backend hiệu năng cao, bridge `pyo3` và các tiện ích tăng tốc cục bộ

Các khu vực quan trọng:

- `dataset/`: nơi đặt ảnh thô ban đầu
- `datasets/`: vùng dữ liệu chuẩn hóa hoặc dữ liệu phụ trợ về sau
- `models/`: nơi đặt model và metadata như `labels.json`
- `artifacts/manifests/`: manifest `.parquet` và `.csv`
- `artifacts/reports/`: báo cáo thống kê dữ liệu
- `artifacts/checkpoints/`: checkpoint huấn luyện `.pt`
- `artifacts/cache/`: cache ảnh đã resize sẵn để train nhanh hơn
- `python/localagent/`: package Python chính
- `src/`: mã nguồn Rust
- `tests/`: test cho pipeline và training

### `interface/`

Đây là giao diện Next.js ở mức mẫu. Bạn có thể bỏ qua nếu đang tập trung vào dữ liệu, huấn luyện mô hình hoặc backend local.

## Công nghệ chính

### Python trong `localagent`

- `torch`, `torchvision`
- `polars`, `numpy`
- `opencv-python-headless`
- `onnx`, `onnxruntime`
- `maturin`
- `httpx`
- `pytest`, `ruff`, `mypy`

### Rust trong `localagent`

- `actix-web`
- `tokio`
- `rayon`
- `serde`, `serde_json`
- `anyhow`, `thiserror`
- `tracing`, `tracing-subscriber`
- `pyo3`
- `ort`

Ngoài phần bridge và server, Rust hiện còn hỗ trợ bước build cache ảnh huấn luyện song song để giảm chi phí đọc, decode và resize lặp lại trong nhiều epoch.

## Yêu cầu môi trường

Bạn nên cài sẵn:

- Python `>= 3.11`
- `uv`
- Rust toolchain và `cargo`
- `bun` nếu muốn chạy thử `interface/`

## Bắt đầu nhanh sau khi clone

Từ root của repository:

```powershell
cd localagent
uv sync --dev
uv run maturin develop
```

Lệnh trên sẽ:

- tạo `.venv/`
- cài dependency Python
- build bridge Rust cho Python trong môi trường hiện tại

## Các lệnh phát triển cơ bản trong `localagent`

### Chạy test Python

```powershell
cd localagent
uv run pytest
```

### Chạy lint Python

```powershell
cd localagent
uv run ruff check python/localagent tests
```

### Chạy local HTTP server từ Rust

```powershell
cd localagent
cargo run --bin localagent-server
```

## Hai pipeline chính

Hiện tại `localagent` xoay quanh 2 pipeline chạy nối tiếp nhau:

1. Pipeline dữ liệu: quét ảnh thô, làm sạch, suy nhãn, chia `train/val/test`, sinh manifest và báo cáo.
2. Pipeline huấn luyện: đọc manifest, warm cache ảnh bằng Rust, dựng `DataLoader`, huấn luyện baseline và ghi checkpoint.

Luồng chuẩn để test end-to-end là:

```powershell
cd localagent
uv sync --dev
uv run maturin develop
uv run python -m localagent.data.pipeline run-all
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train warm-cache
uv run python -m localagent.training.train fit
```

## Pipeline dữ liệu ảnh thô

Pipeline hiện được thiết kế cho trường hợp dataset có rất nhiều ảnh thô, để lộn xộn trong cùng một thư mục và chưa chia sẵn `train/val/test`.

### Đầu vào mặc định

Đặt ảnh thô vào:

```text
localagent/dataset/
```

Pipeline sẽ đọc đệ quy trong thư mục này và xử lý các ảnh có phần mở rộng hợp lệ như `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

### Pipeline hiện làm gì

- quét toàn bộ dataset
- đọc metadata ảnh bằng OpenCV
- tạo manifest bằng Polars
- suy nhãn tự động từ tên file
- phát hiện ảnh lỗi không decode được
- phát hiện ảnh quá nhỏ
- phát hiện ảnh trùng nội dung
- tạo split `train/val/test` theo nhãn
- xuất báo cáo phân bố dữ liệu
- hiển thị thanh tiến trình trong CLI khi scan

### Cách suy nhãn hiện tại

Pipeline suy nhãn từ prefix của tên file trước phần số ở cuối tên.

Ví dụ:

- `battery_123.jpg` → `battery`
- `Miscellaneous Trash_12.jpg` → `miscellaneous_trash`
- `R_1.jpg` → `r`

### Chạy pipeline

Từ thư mục `localagent/`:

```powershell
uv run python -m localagent.data.pipeline scan
uv run python -m localagent.data.pipeline split
uv run python -m localagent.data.pipeline report
```

Hoặc chạy toàn bộ:

```powershell
uv run python -m localagent.data.pipeline run-all
```

Nếu muốn tắt progress bar:

```powershell
uv run python -m localagent.data.pipeline run-all --no-progress
```

### Đầu ra của pipeline

Sau khi chạy, pipeline sẽ sinh ra:

- `localagent/artifacts/manifests/dataset_manifest.parquet`
- `localagent/artifacts/manifests/dataset_manifest.csv`
- `localagent/artifacts/reports/summary.json`
- `localagent/artifacts/reports/split_summary.csv`
- `localagent/artifacts/reports/quality_summary.csv`
- `localagent/artifacts/reports/extension_summary.csv`
- `localagent/artifacts/reports/label_summary.csv`

### Ý nghĩa thực tế của pipeline dữ liệu

Pipeline này là bước chuẩn hóa dataset trước khi train. Nó giúp bạn:

- giữ nguyên dữ liệu gốc trong `localagent/dataset/`
- biết ảnh nào bị lỗi decode, quá nhỏ hoặc trùng
- có một manifest trung tâm để training không cần tự chia cây thư mục
- có báo cáo nhanh để kiểm tra chất lượng dataset trước khi train

Nói ngắn gọn, đây là pipeline biến một thư mục ảnh thô lộn xộn thành dữ liệu có cấu trúc để huấn luyện.

## Huấn luyện từ manifest

Sau khi chạy `run-all`, bạn có thể huấn luyện trực tiếp từ manifest mà không cần tự chia thư mục `train/val/test`.

### Xem nhanh kế hoạch huấn luyện

```powershell
cd localagent
uv run python -m localagent.training.train summary
```

Lệnh này sẽ in ra:

- số mẫu hợp lệ có nhãn
- số lớp
- tên các lớp
- số lượng theo `train`, `val`, `test`
- thiết bị train được resolve thực tế
- trạng thái Rust acceleration

### Xuất file nhãn cho suy luận

```powershell
cd localagent
uv run python -m localagent.training.train export-labels
```

Kết quả sẽ ghi vào:

```text
localagent/models/labels.json
```

### Warm cache ảnh bằng Rust

Để chuẩn bị cache ảnh đã resize sẵn trước khi train:

```powershell
cd localagent
uv run python -m localagent.training.train warm-cache
```

Bạn cũng có thể đổi kích thước cache:

```powershell
uv run python -m localagent.training.train warm-cache --image-size 160
```

Cache sẽ được ghi vào:

```text
localagent/artifacts/cache/training/<image_size>px/
```

Mỗi ảnh hợp lệ trong manifest sẽ có một ảnh cache tương ứng theo `sample_id`. Khi chạy `fit`, dataset sẽ ưu tiên đọc ảnh cache này thay vì decode và resize lại ảnh gốc ở từng epoch.

Nếu Rust gặp ảnh không đọc được theo backend `image`, hệ thống sẽ ghi báo cáo lỗi vào:

```text
localagent/artifacts/reports/training_cache_failures_<image_size>px.json
```

Ngay sau đó, trainer sẽ tự thử một vòng fallback bằng OpenCV chỉ cho các ảnh bị lỗi từ Rust. Nếu OpenCV đọc được, cache sẽ được bổ sung và file report sẽ được thu gọn chỉ còn các lỗi thực sự chưa cứu được.

### Huấn luyện baseline từ manifest

```powershell
cd localagent
uv run python -m localagent.training.train fit
```

CLI hiện sẽ:

1. đọc manifest
2. warm cache ảnh bằng Rust theo kích thước train hiện tại
3. xây `DataLoader` từ manifest
4. dựng model `MobileNetV3-Small`
5. mặc định cố gắng dùng pretrained weights và freeze backbone để giảm tải CPU
6. hiển thị progress bar theo từng epoch/batch
7. tự dừng sớm bằng early stopping nếu validation loss không còn cải thiện đủ tốt
8. lưu checkpoint và `labels.json`

### Pipeline huấn luyện hiện hoạt động thế nào

Pipeline huấn luyện hiện có 4 lớp trách nhiệm:

- `summary`: đọc manifest và cho bạn thấy dữ liệu có đủ điều kiện train hay chưa
- `warm-cache`: dùng Rust để chuẩn bị ảnh resize sẵn, giảm chi phí I/O và resize lặp lại
- `fit`: train mô hình baseline từ manifest với backbone nhẹ
- checkpoint/output: lưu kết quả train để dùng cho bước suy luận sau

Nếu CLI đang hiển thị progress bar, đó là bình thường. Hiện progress bar có ở:

- bước scan dataset
- bước build cache ảnh bằng Rust
- từng epoch/batch của `fit`

### Một số tùy chọn hữu ích

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
uv run python -m localagent.training.train fit --image-size 160 --epochs 5
uv run python -m localagent.training.train fit --batch-size 32 --num-workers 0
uv run python -m localagent.training.train fit --force-cache
uv run python -m localagent.training.train fit --no-rust-cache
uv run python -m localagent.training.train fit --cache-format raw
uv run python -m localagent.training.train fit --no-pretrained
uv run python -m localagent.training.train fit --train-backbone
uv run python -m localagent.training.train fit --class-bias loss
uv run python -m localagent.training.train fit --class-bias sampler
uv run python -m localagent.training.train fit --class-bias both
uv run python -m localagent.training.train fit --early-stopping-patience 2
uv run python -m localagent.training.train fit --no-progress
```

Khi dùng `--no-progress`, trainer sẽ tắt progress bar dạng thanh nhưng vẫn in snapshot tiến độ theo batch và summary theo epoch để theo dõi các run dài.

Trainer hiện sẽ tự lưu:

- checkpoint tốt nhất ở `artifacts/checkpoints/<experiment_name>.pt`
- checkpoint resume mới nhất ở `artifacts/checkpoints/<experiment_name>.last.pt`
- benchmark theo lớp ở `artifacts/reports/<experiment_name>_evaluation.json`
- confusion matrix ở `artifacts/reports/<experiment_name>_confusion_matrix.csv`

Khi resume, hãy giữ nguyên `--experiment-name` để trainer đọc đúng file `artifacts/checkpoints/<experiment_name>.last.pt`.

Rust hiện vẫn hỗ trợ chung cho các backbone CNN này ở phần warm-cache, đọc dữ liệu và pipeline I/O; phần forward/backward của CNN vẫn do PyTorch trong Python đảm nhiệm.

Preset đang có:

- `cpu_fast`: `mobilenet_v3_small`, `image_size=160`, `batch_size=32`, `cache_format=raw`, `class_bias=loss`
- `cpu_balanced`: `resnet18`, `image_size=224`, `batch_size=16`, `cache_format=raw`, `class_bias=loss`
- `cpu_stronger`: `efficientnet_b0`, `image_size=224`, `batch_size=8`, `cache_format=raw`, `class_bias=loss`

### Gợi ý test nhanh trước khi train lâu

Nếu bạn muốn kiểm tra toàn bộ pipeline trước khi chạy lâu trên CPU:

```powershell
cd localagent
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train warm-cache --image-size 160
uv run python -m localagent.training.train fit --image-size 160 --epochs 3 --batch-size 32 --num-workers 0
```

Lệnh trên giúp bạn xác minh:

- manifest đọc được
- Rust bridge hoạt động
- cache ảnh được tạo đúng
- progress bar hiển thị ổn
- training loop chạy hết mà không phải chờ quá lâu

### Ghi chú quan trọng trước khi test

- Nếu máy không có CUDA thì `summary` sẽ báo `resolved_device: cpu`. Đây là bình thường. Khi đó `fit` vẫn chạy được nhưng sẽ chậm hơn đáng kể so với GPU.
- Rust hiện không thay thế toàn bộ phần train của PyTorch. Rust đang hỗ trợ mạnh ở bước warm cache ảnh và tăng tốc tiền xử lý lặp lại, không thay thế phép tính forward/backward của mô hình.
- Backbone mặc định hiện là `mobilenet_v3_small`. Đây là model nhẹ hơn baseline CNN cũ và phù hợp hơn để thử trên CPU.
- Mặc định trainer sẽ cố tải pretrained weights. Nếu không tải được do không có cache hoặc không có mạng, hệ thống sẽ tự fallback về random init và vẫn tiếp tục train.
- Mặc định backbone bị freeze để giảm thời gian train trên CPU. Nếu muốn fine-tune toàn bộ model, dùng `--train-backbone`.
- Lần chạy `warm-cache` đầu tiên có thể mất thời gian vì phải đọc ảnh gốc, resize và ghi cache vào `artifacts/cache/training/<image_size>px/`. Những lần sau sẽ nhanh hơn nếu không dùng `--force-cache`.
- Nếu `warm-cache` báo có lỗi, trainer sẽ tự thử cứu tiếp bằng OpenCV. Chỉ các ảnh còn lỗi sau bước fallback này mới ở lại trong `training_cache_failures_<image_size>px.json`.
- Khi chạy `fit`, CLI có thể in progress bar khá liên tục ở từng batch. Đây là hành vi mong đợi, không phải lỗi.
- Nếu bạn chỉ muốn xác minh pipeline hoạt động, hãy giảm `image_size` và `epochs` trước. Đừng chạy cấu hình nặng ngay trên CPU.
- Dataset chỉ train tốt khi pipeline suy nhãn được từ tên file. Nếu tên file không mang cấu trúc kiểu `label_123.jpg`, nhiều mẫu có thể rơi vào `label=unknown` hoặc `split=excluded`.
- Ảnh lỗi, ảnh trùng và ảnh quá nhỏ không bị di chuyển khỏi dữ liệu gốc. Chúng chỉ bị đánh dấu trong manifest và bị loại khỏi tập train.
- `run-all` chỉ chuẩn hóa dữ liệu và sinh báo cáo. Nó không train mô hình. Sau `run-all`, bạn vẫn cần chạy `summary`, `warm-cache`, rồi mới `fit`.
- Nếu bạn đổi `--image-size`, hệ thống sẽ tạo một nhánh cache mới tương ứng. Ví dụ `160px` và `224px` là hai bộ cache tách biệt.
- Trên Windows, `num_workers=0` thường ổn định hơn cho bước test nhanh đầu tiên. Sau khi xác nhận pipeline chạy đúng, bạn mới nên thử tăng `num_workers`.

### Lưu ý quan trọng về tốc độ

Nếu máy không có CUDA, phần train vẫn chạy bằng PyTorch trên CPU. Rust hiện giúp tăng tốc chủ yếu ở khâu:

- build cache ảnh song song
- giảm decode/resize lặp lại qua nhiều epoch
- cải thiện trải nghiệm CLI với progress bar rõ ràng

Nó không thay thế GPU cho phần toán tử học sâu. Nếu vẫn thấy chậm, hướng tối ưu tiếp theo thường là:

- giảm `image_size`
- giảm `epochs`
- tăng hoặc giảm `batch_size` tùy RAM
- dùng GPU/CUDA nếu có

## Cách chạy `interface`

Nếu muốn mở giao diện mẫu:

```powershell
cd interface
bun install
bun run dev
```

Hoặc dùng `npm`:

```powershell
cd interface
npm install
npm run dev
```

## Quy ước Git và `.gitignore`

Repository chỉ dùng một file `.gitignore` duy nhất ở root.

Hiện tại `.gitignore` đã cấu hình để không đưa lên GitHub:

- cache và build output của Next.js
- `node_modules/`
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `target/` của Rust
- artifact model như `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`, `*.pdb`
- dataset thô trong `localagent/dataset/`
- dữ liệu thô tương lai trong `localagent/datasets/raw/`
- manifest, report và cache sinh ra trong `localagent/artifacts/`

Các file lock quan trọng vẫn được giữ lại:

- `localagent/uv.lock`
- `localagent/Cargo.lock`
- `interface/bun.lock`

## Luồng làm việc khuyến nghị

1. Clone repository.
2. Vào `localagent/`.
3. Chạy `uv sync --dev`.
4. Chạy `uv run maturin develop`.
5. Đặt ảnh thô vào `localagent/dataset/`.
6. Chạy `uv run python -m localagent.data.pipeline run-all`.
7. Kiểm tra manifest và các báo cáo.
8. Chạy `uv run python -m localagent.training.train summary`.
9. Chạy `uv run python -m localagent.training.train warm-cache`.
10. Chạy `uv run python -m localagent.training.train fit`.

## Ghi chú hiện trạng

- `localagent/` là phần vận hành chính hiện tại.
- `interface/` vẫn là phần mẫu.
- Pipeline dữ liệu ưu tiên an toàn dữ liệu gốc: chỉ đọc, phân tích, đánh dấu và xuất manifest/report.
- Huấn luyện hiện là baseline đầu tiên để xác nhận flow ảnh thô → manifest → split → DataLoader → checkpoint.
- Rust đang được dùng theo hướng bổ trợ thực dụng cho hiệu năng và CLI, không thay thế hoàn toàn phần train tensor của PyTorch.
- Nếu bạn thấy train lâu bất thường, hãy kiểm tra lần lượt: `resolved_device`, kích thước ảnh, số epoch, batch size, số mẫu hợp lệ trong manifest và trạng thái cache ảnh.
