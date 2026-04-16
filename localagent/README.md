# Local Smart Waste Agent

`localagent` là phần lõi của dự án Smart Waste Sorting.

Hiện tại module này hỗ trợ:

- quét dataset ảnh thô trong `dataset/`
- tạo manifest Polars
- suy nhãn từ tên file
- phát hiện ảnh lỗi, ảnh quá nhỏ, ảnh trùng
- tạo split `train/val/test`
- xuất báo cáo dữ liệu
- huấn luyện baseline trực tiếp từ manifest

## Cấu trúc quan trọng

```text
localagent/
├── dataset/
├── datasets/
├── artifacts/
│   ├── manifests/
│   ├── reports/
│   └── checkpoints/
├── models/
├── python/localagent/
├── src/
└── tests/
```

## Lệnh thường dùng

```powershell
cd localagent
uv sync --dev
uv run python -m localagent.data.pipeline run-all
uv run python -m localagent.training.train summary
uv run python -m localagent.training.train fit
```

## Đầu ra chính

- `artifacts/manifests/dataset_manifest.parquet`
- `artifacts/manifests/dataset_manifest.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/label_summary.csv`
- `models/labels.json`
- `artifacts/checkpoints/*.pt`

README đầy đủ của repository nằm ở thư mục root.
