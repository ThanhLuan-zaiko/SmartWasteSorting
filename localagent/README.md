# Local Smart Waste Agent

Scaffold `localagent` nay tap trung vao agent phan loai rac bang hinh anh theo huong lai Python + Rust:

- Python (`uv`) quan ly du lieu, vision transforms, huan luyen, orchestration va client giao tiep.
- Rust (`cargo`) cung cap loi hieu nang, ONNX runtime, Python extension (`pyo3`/`maturin`) va HTTP server cuc bo.

## Cau truc

```text
localagent/
├── Cargo.toml
├── pyproject.toml
├── configs/
├── datasets/
├── models/
├── artifacts/
├── logs/
├── python/localagent/
│   ├── api/
│   ├── bridge/
│   ├── config/
│   ├── data/
│   ├── domain/
│   ├── inference/
│   ├── services/
│   ├── training/
│   ├── utils/
│   └── vision/
├── src/
│   ├── bin/
│   ├── config.rs
│   ├── domain.rs
│   ├── error.rs
│   ├── inference.rs
│   ├── lib.rs
│   ├── python_api.rs
│   └── telemetry.rs
└── tests/
```

## Vai tro hai ngon ngu

- Python la lop dieu phoi chinh cho dataset indexing, trainer, vision preprocessing va phat lenh huan luyen/suy luan.
- Rust la backend cuc bo cho inference, kha nang phuc vu API noi bo va sau nay co the them thread pool, SIMD, ONNX runtime.
- `maturin` + `pyo3` giu vai tro cau noi de Python goi truc tiep Rust khi can latency thap.

## Lenh khoi tao tiep theo

```powershell
cd localagent
uv sync
uv run pytest
uv run maturin develop
cargo run --bin localagent-server
```

Thu muc `interface` hien tai khong duoc rang buoc vao scaffold nay.
