# Smart Waste Sorting

## Giới thiệu

Đây là repository cho hệ thống phân loại rác thông minh. Hiện tại dự án được tổ chức thành hai phần:

- `localagent/`: phần lõi chạy cục bộ, kết hợp Python và Rust để xử lý dữ liệu, huấn luyện, suy luận và backend nội bộ.
- `interface/`: giao diện Next.js ở mức mẫu tham khảo, chưa phải phần trọng tâm để vận hành mô hình ở thời điểm hiện tại.

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
   │  └─ checkpoints/
   ├─ logs/
   ├─ python/localagent/
   ├─ src/
   └─ tests/
```

## Vai trò của từng phần

### `localagent/`

Đây là phần lõi của hệ thống phân loại rác.

- Python dùng `uv` để quản lý môi trường, dependency, pipeline dữ liệu, tiền xử lý ảnh, orchestration, test và huấn luyện.
- Rust dùng `cargo` để cung cấp local HTTP server, runtime hiệu năng cao, ONNX runtime và extension cho Python qua `pyo3`.

Các khu vực quan trọng trong `localagent/`:

- `dataset/`: nơi đặt dữ liệu ảnh thô ban đầu, chưa cần chia thư mục.
- `datasets/`: vùng dữ liệu chuẩn hóa hoặc dữ liệu dùng cho các bước khác sau này nếu cần.
- `configs/`: cấu hình chạy cục bộ.
- `models/`: nơi đặt model và metadata như `labels.json`.
- `artifacts/manifests/`: nơi pipeline sinh manifest `.parquet` và `.csv`.
- `artifacts/reports/`: nơi pipeline sinh báo cáo thống kê dữ liệu.
- `artifacts/checkpoints/`: nơi lưu checkpoint huấn luyện.
- `python/localagent/`: package Python chính.
- `src/`: mã nguồn Rust.
- `tests/`: test cho pipeline dữ liệu và training.

### `interface/`

Đây là giao diện Next.js để tham khảo hoặc phát triển UI sau này. Bạn có thể bỏ qua nếu đang tập trung vào xử lý dữ liệu, train model hoặc xây backend local.

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
```

Lệnh trên sẽ:

- tạo `.venv/`
- cài dependency Python
- dùng `uv.lock` để cố định phiên bản môi trường

## Các lệnh phát triển cơ bản trong `localagent`

### Chạy test Python

```powershell
cd localagent
uv run pytest
```

### Chạy lint Python

```powershell
cd localagent
uv run ruff check python tests
```

### Build bridge Rust cho Python

```powershell
cd localagent
uv run maturin develop
```

### Chạy local HTTP server từ Rust

```powershell
cd localagent
cargo run --bin localagent-server
```

## Pipeline dữ liệu ảnh thô

Pipeline dữ liệu được thiết kế cho trường hợp dataset rất nhiều ảnh, để lộn xộn trong cùng một thư mục và chưa chia sẵn `train/val/test`.

### Đầu vào mặc định

Bạn đặt ảnh thô vào:

```text
localagent/dataset/
```

Pipeline sẽ đọc đệ quy trong thư mục này và xử lý các ảnh có phần mở rộng hợp lệ như `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

### Pipeline làm gì

Pipeline hiện hỗ trợ các bước:

- scan toàn bộ dataset
- đọc metadata ảnh bằng OpenCV
- tạo manifest bằng Polars
- suy nhãn tự động từ tên file
- phát hiện ảnh lỗi không decode được
- phát hiện ảnh quá nhỏ
- phát hiện ảnh trùng nội dung
- tạo split `train/val/test` theo nhãn
- xuất báo cáo phân bố dữ liệu

### Cách suy nhãn hiện tại

Pipeline hiện suy nhãn từ prefix của tên file trước số thứ tự ở cuối tên.

Ví dụ:

- `battery_123.jpg` → `battery`
- `Miscellaneous Trash_12.jpg` → `miscellaneous_trash`
- `R_1.jpg` → `r`

Nhãn được chuẩn hóa về chữ thường và thay ký tự không phải chữ/số bằng dấu gạch dưới.

### Cách chạy pipeline

Từ thư mục `localagent/`:

```powershell
uv run python -m localagent.data.pipeline scan
uv run python -m localagent.data.pipeline split
uv run python -m localagent.data.pipeline report
```

Hoặc chạy toàn bộ một lượt:

```powershell
uv run python -m localagent.data.pipeline run-all
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

### Ý nghĩa của manifest

Manifest là bảng dữ liệu trung tâm để dùng cho các bước tiếp theo. Nó chứa các cột quan trọng như:

- `sample_id`
- `image_path`
- `relative_path`
- `file_name`
- `extension`
- `file_size`
- `width`, `height`, `channels`
- `decode_ok`
- `raw_label`
- `label`
- `content_hash`
- `is_duplicate`
- `duplicate_of`
- `is_too_small`
- `is_valid`
- `quarantine_reason`
- `split`

### Hành vi hiện tại của pipeline

- Ảnh lỗi, ảnh quá nhỏ hoặc ảnh trùng sẽ không bị di chuyển đi chỗ khác.
- Pipeline chỉ đánh dấu trạng thái ngay trong manifest.
- Chỉ các ảnh `is_valid=true` và có `label` khác `unknown` mới được gán vào `train`, `val`, `test`.
- Ảnh không hợp lệ hoặc chưa suy nhãn được sẽ có `split="excluded"`.

Điều này giúp bạn an toàn khi thử pipeline nhiều lần vì dữ liệu gốc không bị sửa.

## Huấn luyện từ manifest

Sau khi chạy `run-all`, bạn có thể huấn luyện trực tiếp từ manifest mà không cần tự tạo cây thư mục `train/val/test`.

### Xem nhanh kế hoạch huấn luyện

```powershell
cd localagent
uv run python -m localagent.training.train summary
```

Lệnh này sẽ đọc manifest và in ra:

- số mẫu hợp lệ có nhãn
- số lớp
- tên các lớp
- số lượng mẫu theo `train`, `val`, `test`

### Xuất file nhãn cho suy luận

```powershell
cd localagent
uv run python -m localagent.training.train export-labels
```

Kết quả sẽ ghi vào:

```text
localagent/models/labels.json
```

### Huấn luyện baseline từ manifest

```powershell
cd localagent
uv run python -m localagent.training.train fit
```

Huấn luyện hiện tại dùng một CNN baseline nhỏ để kiểm tra pipeline đầu-cuối:

- đọc ảnh trực tiếp từ manifest
- dùng split đã được tạo trong manifest
- build `labels.json`
- lưu checkpoint vào `localagent/artifacts/checkpoints/`

## Cách chạy `interface`

Nếu muốn mở giao diện mẫu:

```powershell
cd interface
bun install
bun run dev
```

Nếu không dùng `bun`, có thể thay bằng:

```powershell
cd interface
npm install
npm run dev
```

## Quy ước Git và `.gitignore`

Repository này chỉ dùng một file `.gitignore` duy nhất ở root.

Hiện tại `.gitignore` đã được cấu hình để không đưa lên GitHub:

- cache và build output của Next.js
- `node_modules/`
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `target/` của Rust
- model artifact như `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`, `*.pdb`
- dataset thô trong `localagent/dataset/`
- dữ liệu thô tương lai trong `localagent/datasets/raw/`
- manifest và report sinh ra trong `localagent/artifacts/manifests/` và `localagent/artifacts/reports/`
- checkpoint huấn luyện dạng `.pt`

Các file lock quan trọng vẫn được giữ lại:

- `localagent/uv.lock`
- `localagent/Cargo.lock`
- `interface/bun.lock`

## Luồng làm việc khuyến nghị

Nếu bạn là người mới vào dự án, thứ tự nên là:

1. Clone repository.
2. Vào `localagent/`.
3. Chạy `uv sync --dev`.
4. Đặt ảnh thô vào `localagent/dataset/`.
5. Chạy `uv run python -m localagent.data.pipeline run-all`.
6. Kiểm tra manifest và các báo cáo.
7. Chạy `uv run python -m localagent.training.train summary`.
8. Chạy `uv run python -m localagent.training.train fit` khi sẵn sàng huấn luyện.

## Ghi chú hiện trạng

- `localagent/` đã có scaffold module-first cho pipeline phân loại rác bằng hình ảnh.
- `interface/` hiện vẫn là phần mẫu, chưa ràng buộc chặt với `localagent`.
- Pipeline dữ liệu ưu tiên an toàn dữ liệu gốc: chỉ đọc, phân tích, đánh dấu và xuất manifest/report.
- Training hiện là baseline đầu tiên để xác nhận toàn bộ flow từ ảnh thô → manifest → split → dataloader → checkpoint.
