# Smart Waste Sorting

## Giới thiệu

Đây là repository cho hệ thống phân loại rác thông minh. Ở trạng thái hiện tại, dự án được tách thành hai phần:

- `localagent/`: phần lõi chạy cục bộ, kết hợp Python và Rust để phục vụ huấn luyện, suy luận và backend nội bộ cho bài toán phân loại rác bằng hình ảnh.
- `interface/`: giao diện Next.js đang ở mức mẫu tham khảo. Thư mục này chưa phải trọng tâm vận hành chính ở thời điểm hiện tại.

Nếu bạn mới clone repository, hãy bắt đầu từ `localagent/`. Đây là nơi chứa phần hệ thống quan trọng nhất cho pipeline AI hiện tại.

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
   ├─ configs/
   ├─ datasets/
   ├─ models/
   ├─ artifacts/
   ├─ logs/
   ├─ python/localagent/
   ├─ src/
   └─ tests/
```

## Vai trò của từng phần

### `localagent/`

Đây là phần lõi của dự án, được thiết kế theo hướng Python và Rust chạy song hành:

- Python dùng `uv` để quản lý môi trường, dependency, pipeline dữ liệu, tiền xử lý ảnh, orchestration, test và cầu nối với backend Rust.
- Rust dùng `cargo` để cung cấp các thành phần hiệu năng cao như local HTTP server, ONNX runtime, xử lý song song và extension cho Python qua `pyo3`.

Các thư mục quan trọng trong `localagent/`:

- `configs/`: chứa file cấu hình mẫu và các cấu hình cục bộ cho agent.
- `datasets/`: nơi đặt dữ liệu huấn luyện hoặc dữ liệu kiểm thử cục bộ.
- `models/`: nơi đặt file mô hình và metadata như nhãn lớp.
- `artifacts/`: nơi chứa các output sinh ra trong quá trình train, export hoặc thử nghiệm.
- `logs/`: nơi chứa log khi chạy cục bộ.
- `python/localagent/`: package Python chính của hệ thống.
- `src/`: mã nguồn Rust cho backend, runtime và bridge với Python.
- `tests/`: test hiện tại cho scaffold Python.

### `interface/`

Đây là giao diện Next.js để tham khảo hoặc phát triển UI sau này. Bạn có thể bỏ qua nếu đang tập trung vào:

- dựng agent cục bộ
- huấn luyện mô hình
- export model
- chạy suy luận hoặc backend nội bộ

## Công nghệ và dependency chính

### Python trong `localagent`

Phần Python hiện được cấu hình qua `uv` với các gói chính:

- `torch`, `torchvision`
- `polars`, `numpy`
- `opencv-python-headless`
- `onnx`, `onnxruntime`
- `maturin`
- `httpx`
- `pytest`, `ruff`, `mypy`

### Rust trong `localagent`

Phần Rust hiện dùng các crate chính:

- `actix-web`
- `tokio`
- `rayon`
- `serde`, `serde_json`
- `anyhow`, `thiserror`
- `tracing`, `tracing-subscriber`
- `pyo3`
- `ort`

## Yêu cầu môi trường

Bạn nên cài sẵn các công cụ sau trước khi chạy:

- Python `>= 3.11`
- `uv`
- Rust toolchain và `cargo`
- `bun` nếu muốn chạy thử `interface/`

## Bắt đầu nhanh sau khi clone

Từ thư mục root của repository:

```powershell
cd localagent
uv sync --dev
```

Lệnh trên sẽ:

- tạo môi trường ảo `.venv/`
- cài toàn bộ dependency Python
- dùng `uv.lock` để cố định phiên bản đang dùng trong dự án

## Cách chạy `localagent`

### 1. Chạy test Python

```powershell
cd localagent
uv run pytest
```

### 2. Chạy kiểm tra lint Python

```powershell
cd localagent
uv run ruff check python tests
```

### 3. Build bridge Rust cho Python khi cần phát triển tích hợp

```powershell
cd localagent
uv run maturin develop
```

Lệnh này build extension Rust để Python có thể import `localagent._rust`.

### 4. Chạy local HTTP server từ Rust

```powershell
cd localagent
cargo run --bin localagent-server
```

Server hiện được scaffold để phục vụ backend cục bộ cho bước suy luận hoặc orchestration sau này.

## Cách chạy `interface`

Nếu bạn muốn mở giao diện mẫu để tham khảo:

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

## Quy ước làm việc với dữ liệu và model

Các thư mục sau trong `localagent/` được dành cho dữ liệu chạy cục bộ:

- `datasets/`
- `models/`
- `artifacts/`
- `logs/`

Bạn có thể lưu dữ liệu ảnh, model tạm, output export, checkpoint hoặc log trong các thư mục này khi làm việc cục bộ.

## Quy ước Git và file không commit

Repository này chỉ dùng một file `.gitignore` duy nhất ở root dự án.

Các nhóm file tự sinh đã được cấu hình để không đưa lên GitHub, bao gồm:

- cache và build output của Next.js
- `node_modules/`
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `target/` của Rust
- log runtime
- artifact mô hình như `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`

Các file lock quan trọng vẫn được giữ lại để đồng bộ môi trường:

- `localagent/uv.lock`
- `localagent/Cargo.lock`
- `interface/bun.lock`

## Luồng làm việc khuyến nghị

Nếu bạn là người mới vào dự án, thứ tự nên là:

1. Clone repository.
2. Đi vào `localagent/`.
3. Chạy `uv sync --dev`.
4. Chạy `uv run pytest`.
5. Chạy `cargo run --bin localagent-server` nếu cần backend local.
6. Chỉ vào `interface/` nếu bạn cần tham khảo hoặc phát triển giao diện.

## Ghi chú hiện trạng

- `localagent/` đã có scaffold module-first cho pipeline phân loại rác bằng hình ảnh.
- `interface/` hiện vẫn là phần mẫu, chưa ràng buộc chặt với `localagent`.
- README này được viết để giúp người mới clone hiểu nhanh cấu trúc repository và cách bắt đầu đúng chỗ.
