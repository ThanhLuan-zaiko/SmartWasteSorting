# Tài liệu 07: Workflow thực chiến, checklist vận hành và khắc phục sự cố

## 1. Mục đích của file này

- File này là sổ tay thao tác thực chiến cho người mới.
- File này giả định bạn chưa biết gì về project.
- File này không thay thế các file lý thuyết ở `00` tới `06`.
- File này gom những gì bạn cần làm theo thứ tự.
- File này ưu tiên workflow chạy được hơn là giải thích hàn lâm.
- File này bám theo code hiện tại trong repo.
- File này chỉ rõ lệnh nên chạy ở đâu.
- File này chỉ rõ artifact nào phải xuất hiện sau mỗi bước.
- File này chỉ rõ bước nào bị khóa nếu bước trước chưa xong.
- File này chỉ rõ khi lỗi thì nên đọc file nào trước.
- File này cũng chỉ rõ chỗ tương ứng trong code để bạn tự kiểm chứng.

## 2. Ai nên đọc file này trước

- Người vừa clone repo lần đầu.
- Người đang có dataset ảnh nhưng chưa biết đưa vào workflow thế nào.
- Người muốn train model mà chưa hiểu tại sao UI khóa nút.
- Người muốn biết file `.json`, `.csv`, `.onnx` nằm ở đâu.
- Người muốn biết server Rust làm gì.
- Người muốn biết frontend `interface/` fetch API ra sao.
- Người muốn biết nên chạy bằng CLI hay UI.
- Người muốn debug job fail mà không phải mò toàn bộ code.
- Người muốn sửa code nhưng chưa biết test nào bảo vệ hành vi hiện có.

## 3. Bản đồ cực ngắn của toàn dự án

- Python pipeline nằm trong `localagent/python/localagent/`.
- Rust server nằm trong `localagent/src/`.
- Frontend Next.js nằm trong `interface/`.
- Raw dataset hiện tại được pipeline quét từ `localagent/dataset/`.
- Manifest và review file nằm trong `localagent/artifacts/manifests/`.
- Report JSON/CSV nằm trong `localagent/artifacts/reports/`.
- Cache ảnh training nằm trong `localagent/artifacts/cache/training/`.
- Model ONNX và metadata inference nằm trong `localagent/models/`.
- Checkpoint training nằm dưới vùng artifact/checkpoint của local agent.
- Entry point dataset pipeline là `localagent/python/localagent/data/pipeline.py:1429`.
- Entry point training CLI là `localagent/python/localagent/training/train.py:219`.
- Entry point server Rust là `localagent/src/bin/server.rs`.
- Entry point fetch phía UI là `interface/lib/localagent.ts:272`.
- Nơi UI điều phối hành động là `interface/components/localagent/controller-actions.ts`.
- Nơi UI giữ state và WebSocket là `interface/components/use-localagent-controller.ts`.

## 4. Điều phải hiểu trước khi chạy bất cứ thứ gì

- Project này không phải một script đơn lẻ.
- Đây là một workflow nhiều bước.
- Step 1 là quét dataset và tạo manifest.
- Step 2 là discovery bằng embedding và clustering.
- Step 3 là training classifier.
- Sau Step 3 có pseudo-label, evaluate, export ONNX và report.
- UI chỉ điều phối.
- Python mới là nơi chạy logic dữ liệu và training thật.
- Rust server là lớp orchestration và API.
- Artifact là “hợp đồng dữ liệu” giữa Python, Rust và UI.
- Nếu artifact không được ghi đúng, UI sẽ thiếu dữ liệu dù lệnh đã chạy.
- Nếu manifest sai, các bước sau sai dây chuyền.
- Nếu cluster review stale, promote sẽ bị bỏ qua.
- Nếu model export ra ONNX nhưng verify fail, đừng dùng file đó để suy luận.
- Nếu bạn đọc thấy “logic mờ” ở đâu đó trong mô tả cũ, hãy nhớ code hiện tại chưa có fuzzy logic.
- Code hiện tại dùng CNN embedding cộng spherical k-means cộng MAD outlier cộng human review.

## 5. Các file code nên mở trước khi thao tác

- Mở `localagent/python/localagent/config/settings.py`.
- Mở `localagent/python/localagent/data/pipeline.py`.
- Mở `localagent/python/localagent/data/discovery.py`.
- Mở `localagent/python/localagent/training/train.py`.
- Mở `localagent/python/localagent/training/trainer.py`.
- Mở `localagent/src/bin/server.rs`.
- Mở `localagent/src/jobs/types.rs`.
- Mở `localagent/src/artifacts.rs`.
- Mở `interface/lib/localagent.ts`.
- Mở `interface/components/localagent/controller-actions.ts`.
- Mở `interface/components/use-localagent-controller.ts`.
- Mở `localagent/tests/test_pipeline.py`.
- Mở `localagent/tests/test_train_cli.py`.
- Mở `localagent/tests/test_training_fit.py`.
- Mở `localagent/tests/test_training_artifacts.py`.
- Mở `localagent/tests/test_training_pseudo_label.py`.

## 6. Dấu hiệu repo đang ở trạng thái “đã có dữ liệu mẫu”

- Có thư mục `localagent/dataset/`.
- Có `localagent/artifacts/manifests/dataset_manifest.parquet`.
- Có `localagent/artifacts/reports/summary.json`.
- Có `localagent/artifacts/reports/embedding_summary.json`.
- Có `localagent/artifacts/reports/cluster_summary.json`.
- Có `localagent/models/waste_classifier.onnx`.
- Có `localagent/models/model_manifest.json`.
- Có `localagent/artifacts/reports/run_index.json`.
- Nếu các file trên đã tồn tại, repo có thể đã từng được chạy trước đó.
- Tuy nhiên bạn vẫn nên kiểm tra xem chúng có còn khớp dữ liệu hiện tại hay không.

## 7. Preflight kiểm tra môi trường làm việc

- Đảm bảo bạn đang đứng trong root repo khi đọc file này.
- Đảm bảo thư mục hiện tại là nơi có `README.md`, `interface/` và `localagent/`.
- Đảm bảo Python environment của `uv` có thể chạy.
- Đảm bảo Rust toolchain có thể build server.
- Đảm bảo `bun` có sẵn nếu bạn muốn chạy frontend.
- Đảm bảo dataset không bị đặt nhầm ở `datasets/` nếu bạn đang mong pipeline quét `dataset/`.
- Đảm bảo còn đủ dung lượng đĩa cho cache ảnh và artifact.
- Đảm bảo bạn hiểu Windows path trong report có thể là đường dẫn tuyệt đối trên máy local.
- Đảm bảo bạn không xóa nhầm manifest cũ nếu đang so sánh run giữa nhiều lần train.
- Đảm bảo bạn không vô tình dùng cluster review cũ sau khi đã cluster lại.

## 8. Cấu hình đường dẫn mặc định cần nhớ

- `AgentPaths.dataset_dir` ở `localagent/python/localagent/config/settings.py:16` mặc định là `datasets/`.
- `DatasetPipelineConfig.raw_dataset_dir` ở `settings.py:111` mặc định là `dataset/`.
- Trong workflow hiện tại, bạn nên kiểm tra dữ liệu thật đang được quét từ đâu.
- `summary.json.dataset_root` là cách nhanh nhất để xác nhận raw dataset thực sự đã dùng.
- Đừng đoán đường dẫn từ README cũ hoặc ký ức cũ.
- Hãy kiểm chứng bằng config và artifact thật.

## 9. Workflow rất ngắn nếu chỉ muốn chạy end-to-end bằng CLI

- Chạy `uv run python -m localagent.data.pipeline run-all`.
- Chạy `uv run python -m localagent.data.pipeline embed`.
- Chạy `uv run python -m localagent.data.pipeline cluster`.
- Chạy `uv run python -m localagent.data.pipeline export-cluster-review`.
- Review file `artifacts/manifests/cluster_review.csv`.
- Chạy `uv run python -m localagent.data.pipeline promote-cluster-labels --review-file artifacts/manifests/cluster_review.csv`.
- Chạy `uv run python -m localagent.training.train summary`.
- Chạy `uv run python -m localagent.training.train warm-cache`.
- Chạy `uv run python -m localagent.training.train fit`.
- Chạy `uv run python -m localagent.training.train pseudo-label`.
- Chạy `uv run python -m localagent.training.train evaluate`.
- Chạy `uv run python -m localagent.training.train export-onnx`.
- Chạy `uv run python -m localagent.training.train report`.
- Đây là skeleton workflow.
- Thực tế bạn sẽ phải dừng ở vài điểm để đọc artifact.

## 10. Workflow rất ngắn nếu muốn chạy server và UI

- Chạy server Rust bằng `cargo run --bin localagent-server` trong `localagent/`.
- Lệnh này được ghi ở `README.md` và `localagent/README.md`.
- Bin name được khai báo ở `localagent/Cargo.toml:12`.
- Chạy frontend bằng `bun run dev` trong `interface/`.
- Script `dev` được khai báo ở `interface/package.json:6`.
- Sau đó mở UI ở cổng dev mặc định của Next.js.
- UI sẽ gọi API qua `API_PREFIX = "/api/localagent"` ở `interface/lib/localagent.ts:1`.
- Rewrite của Next.js sẽ chuyển request tới local agent server.

## 11. Bước 0 thực chiến: xác định mục tiêu của lần chạy này

- Nếu bạn chỉ muốn xác minh repo còn chạy được, ưu tiên CLI tối giản.
- Nếu bạn muốn demo cho người khác, dùng UI.
- Nếu bạn muốn nghiên cứu thuật toán, đọc code và chạy CLI từng bước.
- Nếu bạn muốn benchmark model, dùng command benchmark hoặc route job benchmark.
- Nếu bạn muốn sửa code, đừng chạy mọi thứ cùng lúc ngay từ đầu.
- Hãy xác định bạn đang làm một trong các mục tiêu sau.
- Mục tiêu A là dựng manifest mới.
- Mục tiêu B là tạo cluster review mới.
- Mục tiêu C là train baseline mới.
- Mục tiêu D là sinh pseudo-label cho tập outlier.
- Mục tiêu E là export ONNX cho suy luận.
- Mục tiêu F là debug server hoặc UI.

## 12. Bước 1 thực chiến: kiểm tra raw dataset

- Mở `localagent/dataset/`.
- Kiểm tra số thư mục con.
- Kiểm tra naming có hợp lý không.
- Kiểm tra có file lạ không phải ảnh hay không.
- Kiểm tra có nhiều bản copy trùng nội dung hay không.
- Kiểm tra có ảnh quá nhỏ hay không.
- Kiểm tra có ảnh hỏng mà Windows preview không mở được hay không.
- Kiểm tra xem nhãn đang nằm ở tên thư mục hay tên file.
- Nếu trông dataset lộn xộn, đừng kỳ vọng `infer_filename_labels` cứu được hết.
- Nếu dataset quá bẩn, Step 1 sẽ phản ánh điều đó qua report.

## 13. Bước 1 thực chiến: chạy `run-all`

```bash
uv run python -m localagent.data.pipeline run-all --no-progress
```

- Entry point CLI là `localagent/python/localagent/data/pipeline.py:1429`.
- Parser lệnh nằm ở `pipeline.py:1361`.
- Build config từ flag nằm ở `pipeline.py:1400`.
- `run_all()` nằm ở `pipeline.py:639`.
- `run_all()` thực chất chạy `scan`, `split`, `report`.
- Sau lệnh này bạn kỳ vọng có manifest và report cơ bản.
- Nếu lệnh này fail, chưa nên đi tiếp sang Step 2.

## 14. Sau `run-all` phải thấy gì

- `localagent/artifacts/manifests/dataset_manifest.parquet`.
- `localagent/artifacts/manifests/dataset_manifest.csv`.
- `localagent/artifacts/reports/summary.json`.
- `localagent/artifacts/reports/split_summary.csv`.
- `localagent/artifacts/reports/quality_summary.csv`.
- `localagent/artifacts/reports/extension_summary.csv`.
- `localagent/artifacts/reports/label_summary.csv`.
- Nếu thiếu một trong các file trên, hãy đọc log trước khi làm bước khác.

## 15. Cách đọc `summary.json` trong 2 phút

- Nhìn `dataset_root` để xác định pipeline quét đúng chỗ chưa.
- Nhìn `total_files` để biết lượng file đã quét.
- Nhìn `valid_files` để biết còn bao nhiêu ảnh usable.
- Nhìn `invalid_files` để biết ảnh lỗi có nhiều không.
- Nhìn `duplicate_files` để biết duplicate có đáng lo không.
- Nhìn `training_ready_files` để biết còn bao nhiêu ảnh đủ điều kiện train.
- Nhìn `label_counts` để biết phân bố lớp.
- Nhìn `split_counts` để biết train/val/test có bị lệch quá không.
- Nhìn `effective_training_mode` để biết pipeline đang ở chế độ nhãn nào.
- Nhìn `clustered_files` để biết Step 2 đã từng chạy chưa.
- Nhìn `cluster_outlier_files` để biết discovery đã gắn cờ bao nhiêu outlier.

## 16. Nếu `run-all` fail thì đọc đâu

- Đọc stdout hoặc stderr của CLI nếu bạn chạy trực tiếp.
- Nếu chạy qua server, đọc `GET /jobs/{job_id}/logs` ở `localagent/src/bin/server.rs:199`.
- Nếu lỗi liên quan file ảnh, đọc `_inspect_image()` ở `pipeline.py:763`.
- Nếu lỗi liên quan split, đọc `assign_splits()` ở `pipeline.py:105`.
- Nếu lỗi liên quan report, đọc `generate_reports()` ở `pipeline.py:158`.
- Nếu lỗi liên quan schema manifest, đọc `_ensure_manifest_columns()` ở `pipeline.py:680`.

## 17. Khi nào nên chạy riêng `scan`, `split`, `report`

- Chạy riêng `scan` khi bạn nghi dữ liệu nguồn thay đổi nhưng chưa muốn ghi split mới.
- Chạy riêng `split` khi bạn chỉ muốn đổi ratio hoặc seed.
- Chạy riêng `report` khi manifest đã có và bạn chỉ muốn refresh thống kê.
- Chạy `run-all` khi bạn đang từ trạng thái thô.
- Về code, `scan()` ở `pipeline.py:82`, `run_split()` ở `pipeline.py:152`, `run_report()` ở `pipeline.py:205`.

## 18. Bước 1 phụ: export template gán nhãn thủ công

```bash
uv run python -m localagent.data.pipeline export-labeling-template
```

- Lệnh này hữu ích khi bạn cần vòng gán nhãn tay.
- Hàm thực thi nằm ở `pipeline.py:397`.
- File đích mặc định thường là `artifacts/manifests/labeling_template.csv`.
- Sau khi người review sửa file, bạn dùng `import-labels`.

## 19. Bước 1 phụ: import labels

```bash
uv run python -m localagent.data.pipeline import-labels --labels-file artifacts/manifests/labeling_template.csv
```

- Hàm thực thi nằm ở `pipeline.py:464`.
- Code đọc CSV nằm ở `pipeline.py:1338`.
- Code đọc JSON nằm ở `pipeline.py:1342`.
- Code đọc JSONL nằm ở `pipeline.py:1348`.
- Sau import, bạn nên chạy `validate-labels`.

## 20. Bước 1 phụ: validate labels

```bash
uv run python -m localagent.data.pipeline validate-labels
```

- Hàm thực thi nằm ở `pipeline.py:535`.
- Lệnh này giúp phát hiện trường hợp manifest chỉ có một lớp hoặc nhãn không usable.
- Nếu validate fail, đừng train.
- Test tham chiếu là `localagent/tests/test_pipeline.py` phần validate labels.

## 21. Bước 2 thực chiến: trích xuất embedding

```bash
uv run python -m localagent.data.pipeline embed
```

- `embed_dataset()` nằm ở `pipeline.py:209`.
- `extract_embeddings()` nằm ở `localagent/python/localagent/data/discovery.py:72`.
- Đây là bước dùng CNN pretrained để sinh vector đặc trưng.
- Bản hiện tại ưu tiên `resnet18_imagenet`.
- Nếu extractor fail, có fallback handcrafted.
- Sau khi chạy xong bạn kỳ vọng có `dataset_embeddings.npz`.
- Bạn cũng kỳ vọng có `embedding_summary.json`.
- Nếu `embedding_summary.json.fallback_reason` khác `null`, hãy đọc kỹ lý do.

## 22. Bước 2 thực chiến: cluster embeddings

```bash
uv run python -m localagent.data.pipeline cluster
```

- `cluster_dataset()` nằm ở `pipeline.py:230`.
- `cluster_embeddings()` nằm ở `discovery.py:112`.
- Thuật toán cluster là spherical k-means.
- Thuật toán không phải fuzzy c-means.
- Sau khi cluster xong, manifest sẽ có `cluster_id`, `cluster_distance`, `cluster_size`, `is_cluster_outlier`.
- Sau khi cluster xong, bạn kỳ vọng có `cluster_summary.json`.

## 23. Khi nào nên truyền `--num-clusters`

- Khi bạn đã nhìn `cluster_review.csv` cũ và thấy cluster quá to, lẫn nhiều lớp.
- Khi bạn thấy một cụm gom quá nhiều hình ảnh không cùng kiểu.
- Khi bạn muốn ép chi tiết hơn cho human review.
- Không nên tăng cluster quá nhiều chỉ vì nghĩ “nhiều là tốt”.
- Cluster quá nhỏ sẽ làm review rời rạc và mất lợi ích gom nhóm.
- Logic resolve mặc định nằm ở `_resolve_cluster_count()` trong `discovery.py:253`.

## 24. Bước 2 thực chiến: export cluster review

```bash
uv run python -m localagent.data.pipeline export-cluster-review --output artifacts/manifests/cluster_review.csv
```

- `export_cluster_review()` nằm ở `pipeline.py:276`.
- File review mặc định thường là `artifacts/manifests/cluster_review.csv`.
- File này là nơi con người quyết định cluster mang nhãn gì.
- File này dùng fingerprint để chống stale row.
- Đừng chỉnh fingerprint bằng tay.

## 25. Cách review `cluster_review.csv`

- Mở file CSV bằng công cụ bạn quen dùng.
- Nhìn `cluster_id`.
- Nhìn preview summary.
- Nhìn số lượng mẫu trong cluster.
- Nhìn outlier summary.
- Chọn nhãn phù hợp cho cluster.
- Nếu cluster quá lẫn, đừng gán bừa.
- Nếu cluster không chắc, hãy để trạng thái cần review.
- Nếu cluster rõ ràng, gán nhãn và trạng thái accepted theo workflow hiện tại của file review.
- Sau khi sửa xong, lưu file và chuẩn bị promote.

## 26. Bước 2 thực chiến: promote cluster labels

```bash
uv run python -m localagent.data.pipeline promote-cluster-labels --review-file artifacts/manifests/cluster_review.csv
```

- `promote_cluster_labels()` nằm ở `pipeline.py:318`.
- Hàm này cập nhật manifest từ review file.
- Hàm này chỉ áp row còn current.
- Nếu row stale, nó sẽ bỏ qua.
- Nếu bạn vừa re-cluster mà dùng review cũ, expect nhiều row bị bỏ qua.
- Sau bước này, `summary.json.effective_training_mode` thường chuyển sang `accepted_labels_only`.

## 27. Step 2 thành công thì workflow state phải thay đổi ra sao

- `workflow_state()` nằm ở `pipeline.py:582`.
- Rust server cũng tổng hợp workflow ở `localagent/src/workflow.rs`.
- Step 3 chỉ mở khi số cluster reviewed hợp lệ đủ điều kiện.
- Nếu UI vẫn khóa Step 3, đọc `workflow/state`.
- Nếu CLI fit bị chặn qua server, đọc logic gating trong `jobs/commands.rs`.

## 28. Dấu hiệu Step 2 thành công

- `embedding_summary.json` tồn tại.
- `cluster_summary.json` tồn tại.
- Manifest có cột cluster đã được điền.
- `cluster_review.csv` tồn tại.
- `summary.json.clustered_files` lớn hơn 0.
- `summary.json.cluster_summary_exists = true`.
- `summary.json.embedding_artifact_exists = true`.
- `summary.json.cluster_preview_total` lớn hơn 0.

## 29. Step 2 thất bại thì đọc đâu

- Đọc `discovery.py:72` nếu fail lúc embed.
- Đọc `discovery.py:163` nếu nghi extractor pretrained có vấn đề.
- Đọc `discovery.py:206` nếu nghi fallback feature đang kích hoạt.
- Đọc `discovery.py:266` nếu fail lúc cluster.
- Đọc `discovery.py:301` nếu fail lúc outlier detection.
- Đọc `pipeline.py:276` nếu fail lúc export cluster review.
- Đọc `pipeline.py:318` nếu fail lúc promote.
- Đọc `localagent/tests/test_pipeline.py:272` để xem behavior mong đợi.
- Đọc `localagent/tests/test_pipeline.py:444` nếu nghi stale review logic.

## 30. Điều tuyệt đối không nên làm ở Step 2

- Không sửa tay `dataset_embeddings.npz`.
- Không đổi cột manifest bằng tay nếu chưa hiểu schema.
- Không promote cluster review cũ sau khi đã cluster lại.
- Không nói repo đang dùng fuzzy logic nếu chưa thêm code cho nó.
- Không train ngay khi cluster review chưa ổn mà vẫn mong kết quả tốt.

## 31. Bước 3 thực chiến: đọc `summary` của training trước

```bash
uv run python -m localagent.training.train summary
```

- Command này cho bạn biết config huấn luyện hiện tại.
- Nó đi qua `localagent/python/localagent/training/train.py`.
- Nó dùng `WasteTrainer.summarize_training_plan()` ở `trainer.py:47`.
- Đây là cách tốt để kiểm tra preset, model, image size, cache format, class bias trước khi fit.

## 32. Bước 3 thực chiến: warm cache

```bash
uv run python -m localagent.training.train warm-cache
```

- `warm_image_cache()` nằm ở `trainer.py:1569`.
- Nếu repo có bridge Rust sẵn sàng, warm cache có thể dùng hỗ trợ Rust.
- Nếu không, có fallback tương ứng.
- Cache giúp giảm chi phí decode và resize lặp lại qua nhiều epoch.
- Sau command này bạn kỳ vọng có thư mục cache dưới `artifacts/cache/training/`.

## 33. Khi nào nên warm cache

- Khi dataset lớn.
- Khi ảnh đầu vào cần resize lặp lại nhiều lần.
- Khi bạn train nhiều run với cùng `image_size`.
- Khi CPU là nút thắt chính.
- Không nhất thiết phải warm cache nếu bạn chỉ test rất nhỏ.
- Nhưng workflow chuẩn của repo vẫn coi warm cache là bước hợp lý trước fit.

## 34. Bước 3 thực chiến: fit baseline

```bash
uv run python -m localagent.training.train fit --training-preset cpu_fast --experiment-name baseline-waste-sorter --no-progress
```

- `fit()` nằm ở `trainer.py:1270`.
- Preset `cpu_fast` nằm ở `train.py:18`.
- Preset này dùng `mobilenet_v3_small`.
- Preset này phù hợp để có baseline nhanh trên CPU.
- Nếu muốn cân bằng hơn, dùng `cpu_balanced`.
- Nếu muốn model mạnh hơn trên CPU, dùng `cpu_stronger`.

## 35. Những flag quan trọng nhất khi `fit`

- `--training-preset`.
- `--experiment-name`.
- `--training-backend`.
- `--model-name`.
- `--no-pretrained`.
- `--train-backbone`.
- `--image-size`.
- `--batch-size`.
- `--epochs`.
- `--num-workers`.
- `--device`.
- `--cache-dir`.
- `--resume-from`.
- `--class-bias`.
- `--early-stopping-patience`.
- `--early-stopping-min-delta`.
- `--disable-early-stopping`.
- `--no-progress`.

## 36. Cách chọn preset nhanh nhất

- Máy yếu, cần baseline sớm: `cpu_fast`.
- Muốn trade-off hợp lý hơn: `cpu_balanced`.
- Muốn thử backbone mạnh hơn trên CPU: `cpu_stronger`.
- Muốn kiểm soát tay hoàn toàn: bỏ preset và truyền từng flag.

## 37. Cách chọn `class-bias`

- Dataset cân bằng khá tốt: có thể `none`.
- Dataset lệch vừa: `loss`.
- Dataset lệch nặng và muốn batch cân bằng hơn: `sampler`.
- Dataset rất lệch và chấp nhận can thiệp mạnh: `both`.
- Code liên quan nằm ở `_build_train_sampler()` và `_loss_class_weights()` trong `trainer.py`.

## 38. Khi nào nên bật `train-backbone`

- Khi dataset đủ lớn.
- Khi bạn chấp nhận runtime dài hơn.
- Khi backbone frozen không còn cải thiện metric.
- Khi máy đủ mạnh.
- Trên CPU yếu, đừng bật nếu chưa có lý do rõ.

## 39. Nếu muốn resume run cũ

```bash
uv run python -m localagent.training.train fit --experiment-name waste-e25-fast --training-preset cpu_fast --epochs 25 --resume-from artifacts/checkpoints/waste-e25-fast.last.pt
```

- Mẫu command này đã có trong `README.md`.
- Logic resume nằm ở `_load_resume_state()` trong `trainer.py:505`.
- Nếu resume không hoạt động, kiểm tra checkpoint path trước khi đổ lỗi cho trainer.

## 40. Sau `fit` phải thấy gì

- `*_training.json`.
- `*_evaluation.json`.
- `*_confusion_matrix.csv`.
- Checkpoint `.best.pt` hoặc `.last.pt` tương ứng.
- `run_index.json` được cập nhật.
- Trong snapshot hiện có còn có `baseline-waste-sorter_training.json`.

## 41. Cách đọc report training trong 2 phút

- Nhìn `training_preset`.
- Nhìn `model.model_name`.
- Nhìn `image_size`.
- Nhìn `batch_size`.
- Nhìn `epochs_completed`.
- Nhìn `best_epoch`.
- Nhìn `best_loss`.
- Nhìn `stopped_early`.
- Nhìn `test_accuracy`.
- Nhìn `evaluation_summary.macro_f1`.
- Nhìn `class_weight_map`.
- Nhìn `cache_summary`.

## 42. Bước 3 phụ: pseudo-label

```bash
uv run python -m localagent.training.train pseudo-label
```

- `pseudo_label()` nằm ở `trainer.py:1126`.
- Bước này thường áp cho các mẫu chưa label hoặc outlier.
- Logic accept dựa trên confidence threshold và margin threshold.
- Nếu bạn muốn rất chặt, tăng threshold.
- Nếu bạn muốn nhận nhiều mẫu hơn, hạ threshold nhưng phải chấp nhận rủi ro.

## 43. Bước 3 phụ: evaluate

```bash
uv run python -m localagent.training.train evaluate
```

- `evaluate()` nằm ở `trainer.py:869`.
- Đây là cách refresh evaluation report từ checkpoint đã lưu.
- Hữu ích khi bạn muốn kiểm tra lại metric mà không train lại.

## 44. Bước 3 phụ: export ONNX

```bash
uv run python -m localagent.training.train export-onnx
```

- `export_onnx()` nằm ở `trainer.py:959`.
- Lệnh này ghi `waste_classifier.onnx`.
- Lệnh này còn ghi `labels.json`.
- Lệnh này còn ghi `model_manifest.json`.
- Nếu verify bật, code sẽ so ONNX với PyTorch.
- Nếu verify fail, đừng deploy file đó.

## 45. Bước 3 phụ: export spec

```bash
uv run python -m localagent.training.train export-spec
```

- Artifact spec rất hữu ích cho reproducibility.
- Nó cho bạn biết run đã dùng config nào.
- Dù không phải artifact phục vụ inference, nó cực quan trọng cho quản trị experiment.

## 46. Bước 3 phụ: report bundle

```bash
uv run python -m localagent.training.train report
```

- Report bundle hợp nhất nhiều artifact của cùng run.
- UI rất thích kiểu artifact này vì dễ đọc tổng quan.
- Logic build bundle nằm ở `build_artifact_report()` tại `trainer.py:1099`.

## 47. Benchmark dùng khi nào

- Khi bạn muốn so sánh preset.
- Khi bạn muốn so sánh backend.
- Khi bạn muốn biết chi phí thời gian của fit, evaluate, export.
- Command benchmark đã có trong `README`.
- Route server cho benchmark là `POST /jobs/benchmark`.

## 48. Vì sao benchmark quan trọng

- Nó tránh việc bạn cảm giác “model này nhanh hơn” mà không có số.
- Nó tách thời gian thành từng stage.
- Nó cho phép so với run khác qua API compare.
- Nó giúp chọn cấu hình cho máy yếu.

## 49. Cách hiểu `run_index.json`

- Đây là chỉ mục run để UI render danh sách experiment.
- Nếu run mới không lên UI, hãy kiểm tra file này.
- Rust `ArtifactStore` đọc file này trong `localagent/src/artifacts.rs`.
- Nếu file này hỏng format, dashboard run sẽ lỗi.

## 50. Cách hiểu `model_manifest.json`

- File này mô tả input contract của inference.
- Nó chứa labels.
- Nó chứa image size.
- Nó chứa normalization mean/std.
- Nó chứa kết quả verify ONNX.
- Rust inference dựa vào file này để preprocess ảnh.

## 51. Chạy server Rust đúng cách

- Vào thư mục `localagent/`.
- Chạy `cargo run --bin localagent-server`.
- Nếu build fail, xem `Cargo.toml` và dependency local environment.
- Server routes chính nằm trong `src/bin/server.rs`.
- Job orchestration đi qua `src/jobs/runtime.rs`.
- Artifact read model đi qua `src/artifacts.rs`.

## 52. Chạy frontend Next.js đúng cách

- Vào thư mục `interface/`.
- Chạy `bun run dev`.
- Nếu UI không gọi được API, kiểm tra rewrite của Next.js.
- Kiểm tra `interface/next.config.ts`.
- Kiểm tra `API_PREFIX` trong `interface/lib/localagent.ts`.
- Kiểm tra server Rust đã chạy chưa.

## 53. Workflow qua UI nếu bạn không muốn dùng CLI

- Mở dashboard.
- Đợi UI load presets, jobs, runs và workflow state.
- Nếu UI trống dữ liệu, xem server có đang chạy không.
- Chạy Step 1 bằng action pipeline tương ứng.
- Theo dõi log job ở panel job.
- Chờ job completed.
- Refresh state nếu cần.
- Chạy Step 2.
- Xuất cluster review.
- Review file cluster.
- Save cluster review nếu dùng luồng chỉnh qua UI.
- Promote cluster labels.
- Chờ workflow mở Step 3.
- Chạy training.
- Theo dõi logs.
- Chạy pseudo-label nếu cần.
- Chạy export ONNX nếu cần.

## 54. UI lấy dữ liệu từ đâu

- Preset training từ `GET /presets/training`.
- Pipeline catalog từ `GET /presets/pipeline`.
- Runs từ `GET /runs`.
- Run detail từ `GET /runs/{experiment_name}`.
- Compare từ `GET /runs/{experiment_name}/compare`.
- Workflow state từ `GET /workflow/state`.
- Cluster review từ `GET /cluster-review`.
- Job logs từ `GET /jobs/{job_id}/logs`.
- Ảnh dataset từ `GET /dataset/image`.

## 55. File nào trong UI nên đọc đầu tiên

- `interface/lib/localagent.ts` để hiểu type và fetch helper.
- `interface/components/localagent/controller-actions.ts` để hiểu request nào được gửi đi.
- `interface/components/use-localagent-controller.ts` để hiểu state.
- `interface/components/dashboard/discovery/discovery-shared.ts` để hiểu ảnh dataset được dựng URL ra sao.

## 56. `fetchJson()` làm gì

- Nó bọc `fetch`.
- Nó gắn `API_PREFIX`.
- Nó xử lý request JSON thống nhất.
- Nó là chỗ UI dùng cho đa số API call.
- Nếu API call hỏng hàng loạt ở UI, đây là chỗ nên đọc đầu tiên.

## 57. `submitPipeline()` làm gì

- Nó đọc `pipelineForm`.
- Nó dựng payload pipeline job request.
- Nó điền `labels_file` nếu có.
- Nó điền `review_file` nếu có.
- Nó điền `output` cho export.
- Nó điền `num_clusters` nếu người dùng nhập.
- Nó gọi `POST /jobs/pipeline`.
- Nó reload danh sách job sau khi tạo job.

## 58. `submitTraining()` làm gì

- Nó đọc `trainingForm`.
- Nó dựng payload training request.
- Nó có thể gửi tới `/jobs/training`.
- Nó có thể gửi tới `/jobs/benchmark`.
- Nó set selected experiment theo `experiment_name`.
- Nó reload jobs sau khi tạo job.

## 59. `saveClusterReview()` làm gì

- Nó gọi `PUT /cluster-review`.
- Nó gửi payload save request.
- Nó cập nhật state review sau khi server lưu thành công.
- Nếu save fail, xem server có đang chạy pipeline job active không.

## 60. `refreshAll()` làm gì

- Nó nạp presets.
- Nó nạp runs.
- Nó nạp workflow state.
- Nó nạp cluster review.
- Nó nạp jobs.
- Đây là nút “resync” quan trọng khi UI có dấu hiệu lệch trạng thái.

## 61. WebSocket trong UI dùng cho việc gì

- Nó stream log job.
- Nó stream trạng thái job update.
- Nó không stream artifact chi tiết.
- Artifact chi tiết vẫn nên nạp lại qua HTTP sau khi job xong.
- Kiểu event nằm ở `localagent/src/jobs/types.rs`.

## 62. Nếu CLI chạy được nhưng UI lỗi

- Kiểm tra server Rust có chạy không.
- Kiểm tra frontend có rewrite đúng không.
- Kiểm tra `API_PREFIX`.
- Kiểm tra route server còn tên cũ hay đã đổi.
- Kiểm tra type trong `interface/lib/localagent.ts`.
- Kiểm tra network tab của trình duyệt.
- Kiểm tra `run_index.json` và report files có thật sự được ghi chưa.

## 63. Nếu UI chạy được nhưng job fail

- Kiểm tra stdout/stderr log của job.
- Kiểm tra `jobs/runtime.rs`.
- Kiểm tra `jobs/commands.rs`.
- Kiểm tra command bị sinh ra có đúng không.
- Kiểm tra workflow gating có chặn command đó không.
- Kiểm tra đường dẫn input file có đúng không.

## 64. Workflow chuẩn nhất cho người mới hoàn toàn

- Đọc `docs/00`.
- Đọc `docs/01`.
- Chạy `run-all`.
- Mở `summary.json`.
- Chạy `embed`.
- Chạy `cluster`.
- Export cluster review.
- Review cluster.
- Promote cluster labels.
- Chạy `summary` training.
- Chạy `warm-cache`.
- Chạy `fit` với `cpu_fast`.
- Mở training report.
- Chạy `evaluate`.
- Chạy `export-onnx`.
- Mở `model_manifest.json`.

## 65. Workflow chuẩn nhất cho người muốn dùng UI

- Chạy server Rust.
- Chạy frontend.
- Mở dashboard.
- Xác nhận workflow state.
- Chạy job pipeline Step 1.
- Đợi completed.
- Chạy embed.
- Chạy cluster.
- Export cluster review.
- Review cluster.
- Save hoặc promote review.
- Chạy fit.
- Chạy pseudo-label nếu cần.
- Chạy export ONNX.
- Xem run detail và compare nếu cần.

## 66. Workflow chuẩn nhất cho người muốn nghiên cứu thuật toán

- Đọc `data/discovery.py`.
- Đọc `training/trainer.py`.
- Chạy `run-all`.
- Chạy `embed`.
- Mở `embedding_summary.json`.
- Chạy `cluster`.
- Mở `cluster_summary.json`.
- Mở `cluster_review.csv`.
- Đọc `summary.json`.
- Chạy `fit` với baseline.
- Mở `evaluation.json`.
- Mở `confusion_matrix.csv`.
- Đọc `model_manifest.json`.

## 67. Workflow chuẩn nhất cho người muốn deploy inference

- Xác minh training report đủ tốt.
- Chạy `export-onnx`.
- Xác minh `verification.verified = true`.
- Giữ lại `waste_classifier.onnx`.
- Giữ lại `labels.json`.
- Giữ lại `model_manifest.json`.
- Thử route `POST /classify/image`.
- Kiểm tra preprocess trong `localagent/src/inference.rs`.

## 68. Checklist trước khi sửa code

- Bạn biết mình đang sửa Python hay Rust hay UI.
- Bạn đã đọc test liên quan chưa.
- Bạn đã biết artifact nào chịu ảnh hưởng chưa.
- Bạn đã biết route hoặc CLI nào sẽ bị đổi chưa.
- Bạn đã biết UI type nào sẽ phải cập nhật chưa.
- Bạn đã biết có workflow gating nào phụ thuộc logic đó chưa.
- Bạn đã biết có file report nào thay đổi format chưa.
- Bạn đã biết có thể gây stale cluster review hay không chưa.

## 69. Checklist trước khi sửa dataset pipeline

- Đọc `pipeline.py`.
- Đọc `test_pipeline.py`.
- Đọc `workflow.rs`.
- Đọc `jobs/commands.rs`.
- Đọc UI pipeline action nếu đổi naming flag.
- Kiểm tra manifest schema hiện tại.
- Xác định cột mới có làm vỡ CSV/Parquet hay không.

## 70. Checklist trước khi sửa discovery

- Đọc `discovery.py`.
- Đọc `pipeline.py` phần `embed_dataset()` và `cluster_dataset()`.
- Đọc `cluster_review.rs`.
- Đọc `interface/lib/localagent.ts` nếu response cluster review thay đổi.
- Đọc `test_pipeline.py:272`.
- Đọc `test_pipeline.py:444`.

## 71. Checklist trước khi sửa training

- Đọc `train.py`.
- Đọc `trainer.py`.
- Đọc `manifest_dataset.py`.
- Đọc `vision/transforms.py`.
- Đọc `test_train_cli.py`.
- Đọc `test_training_fit.py`.
- Đọc `test_training_artifacts.py`.
- Đọc `test_training_pseudo_label.py`.

## 72. Checklist trước khi sửa server Rust

- Đọc `server.rs`.
- Đọc `jobs/types.rs`.
- Đọc `jobs/commands.rs`.
- Đọc `jobs/runtime.rs`.
- Đọc `artifacts.rs`.
- Đọc `cluster_review.rs`.
- Đọc `inference.rs`.
- Kiểm tra UI type nếu API shape thay đổi.

## 73. Checklist trước khi sửa frontend

- Đọc `interface/lib/localagent.ts`.
- Đọc `controller-actions.ts`.
- Đọc `use-localagent-controller.ts`.
- Đọc `discovery-shared.ts`.
- Xác nhận server response shape hiện tại.
- Xác nhận rewrite config hiện tại.

## 74. Lỗi thường gặp: Step 2 không mở

- Step 1 chưa hoàn tất.
- Manifest chưa được ghi.
- `summary.json` chưa được sinh.
- `workflow_state()` trả step chưa unlock.
- Server đang đọc artifact cũ.
- Bạn chưa refresh UI.
- Bạn đang xem job cũ fail nhưng tưởng đã completed.

## 75. Lỗi thường gặp: Step 3 không mở

- Cluster review chưa được promote.
- Promote dùng file review stale.
- `accepted_label_source_counts` chưa tăng.
- `reviewed_cluster_count` chưa đủ.
- `workflow.rs` vẫn đánh dấu Step 3 locked.
- UI chưa refresh state.

## 76. Lỗi thường gặp: `fit` accuracy cao nhưng mô hình dở

- Dataset lệch lớp mạnh.
- Accuracy bị lớp đông che mắt.
- Macro F1 thấp hơn nhiều.
- Confusion matrix cho thấy lớp hiếm bị hút vào lớp đông.
- Bạn đang nhìn sai metric.

## 77. Lỗi thường gặp: pseudo-label nhận quá ít mẫu

- Threshold quá cao.
- Margin quá cao.
- Model chưa đủ tốt.
- Tập ứng viên quá khó.
- Outlier cluster quá bẩn.
- Bạn đang kỳ vọng pseudo-label làm thay human review.

## 78. Lỗi thường gặp: export ONNX xong nhưng inference fail

- `model_manifest.json` không khớp.
- `labels.json` bị lệch labels.
- Verify ONNX đã bị skip.
- Preprocess ở Rust side không khớp image size.
- File `.onnx` bị ghi đè bởi run khác.

## 79. Lỗi thường gặp: UI không hiện ảnh dataset

- `relative_path` trong manifest sai.
- Route `/dataset/image` không đọc được file đó.
- Bạn đã di chuyển raw dataset sau khi tạo manifest.
- Encode URL không đúng.
- Server path sanitization chặn đường dẫn.

## 80. Lỗi thường gặp: cluster review lưu không được

- Có dataset pipeline job đang chạy.
- File review đang stale.
- Payload save không khớp schema.
- UI dùng path review file khác với file bạn đang mở tay.

## 81. Lỗi thường gặp: benchmark so không được

- Experiment compare không tồn tại.
- Run index chưa có run cần so.
- Benchmark report chưa được ghi.
- Backend đang chọn `rust_tch` cho operation chưa hỗ trợ.

## 82. Lỗi thường gặp: cache không giúp nhanh hơn

- Bạn đổi `image_size` mỗi run.
- Bạn đổi `cache_format`.
- Bạn bật `force_cache`.
- Cache đang được tạo lại liên tục.
- Dung lượng đĩa làm I/O chậm.

## 83. Lỗi thường gặp: resume không tiếp tục từ checkpoint mong muốn

- Sai path `resume_from`.
- Checkpoint đã bị xóa.
- Experiment name không khớp ý định của bạn.
- Bạn đang nghĩ `checkpoint` và `resume_from` là một.
- Thực ra `resume_from` dùng cho tiếp tục train.
- `checkpoint` thường dùng để evaluate hoặc export cụ thể.

## 84. Lỗi thường gặp: report có mà UI không thấy

- `run_index.json` chưa refresh.
- `ArtifactStore` chưa đọc lại.
- Experiment name không khớp run bạn mong đợi.
- Server cần refresh hoặc request lại.
- UI đang giữ selected experiment cũ.

## 85. Lỗi thường gặp: sửa code xong test cũ fail dây chuyền

- Bạn đổi schema report mà quên test artifact.
- Bạn đổi CLI flag mà quên test CLI.
- Bạn đổi logic workflow mà quên test pipeline gating.
- Bạn đổi API shape mà quên UI type.
- Bạn đổi artifact path mà quên `ArtifactStore`.

## 86. Nếu phải debug từ log lên code

- Tìm command đang chạy.
- Xác định job type.
- Xác định command CLI tương ứng.
- Mở entry point Python hoặc Rust liên quan.
- Tìm hàm ném lỗi đầu tiên.
- So với artifact được kỳ vọng nhưng chưa có.
- Đọc test mô tả behavior gần nhất.

## 87. Nếu phải debug từ code xuống artifact

- Xem hàm ghi file nằm ở đâu.
- Xem path helper dựng đường dẫn ra sao.
- Xem server đọc file đó ở đâu.
- Xem UI gọi endpoint nào để lấy file đó.
- Xem test nào xác minh file đó phải tồn tại.

## 88. Nếu phải debug từ UI xuống server

- Xem `controller-actions.ts`.
- Xem `fetchJson()`.
- Xem network request path.
- Xem route tương ứng trong `server.rs`.
- Xem struct request trong `jobs/types.rs`.
- Xem `jobs/commands.rs` sinh CLI gì.
- Xem `jobs/runtime.rs` spawn process gì.

## 89. Nếu phải debug từ server xuống Python

- Xem route gọi job nào.
- Xem enum command map sang CLI gì.
- Xem args được nối ra sao.
- Chạy CLI đó bằng tay nếu cần.
- So kết quả CLI trực tiếp với job qua server.

## 90. Nếu phải debug từ Python xuống dữ liệu

- Mở manifest.
- Mở summary.
- Mở cluster review.
- Mở training report.
- Kiểm tra vài ảnh thật bằng mắt.
- Đừng chỉ nhìn code mà không nhìn dữ liệu.

## 91. Checklist “first success” trong 30 phút

- Xác nhận raw dataset tồn tại.
- Chạy `run-all`.
- Mở `summary.json`.
- Chạy `embed`.
- Chạy `cluster`.
- Export cluster review.
- Promote cluster labels nếu review đã sẵn.
- Chạy `summary` training.
- Chạy `fit --training-preset cpu_fast --epochs 3`.
- Mở `*_training.json`.
- Chạy `export-onnx`.
- Mở `model_manifest.json`.

## 92. Checklist “baseline sạch” trong nửa ngày

- Review kỹ `summary.json`.
- Dọn bớt ảnh lỗi nếu invalid nhiều.
- Dọn duplicate nếu duplicate nhiều.
- Review cluster cẩn thận.
- Promote cluster labels.
- Warm cache.
- Fit `cpu_fast`.
- Đọc evaluation.
- So confusion matrix.
- Nếu hợp lý, fit `cpu_balanced`.
- So run bằng compare.
- Export ONNX của run tốt nhất.

## 93. Checklist “chuẩn bị demo”

- Server Rust chạy ổn.
- Frontend chạy ổn.
- Dashboard load được runs.
- Cluster review có dữ liệu.
- Một run baseline có sẵn training report.
- ONNX verify đã pass.
- Route classify image hoạt động.
- Bạn biết file nào để trình bày metric.

## 94. Checklist “chuẩn bị chỉnh thuật toán clustering”

- Bạn hiểu repo hiện đang dùng spherical k-means.
- Bạn biết repo chưa dùng fuzzy logic.
- Bạn biết cluster review schema hiện là hard assignment.
- Bạn biết cần sửa `discovery.py`.
- Bạn biết có thể cần sửa `pipeline.py`.
- Bạn biết có thể cần sửa `cluster_review.rs`.
- Bạn biết có thể cần sửa UI types.
- Bạn biết test hiện tại nào sẽ fail sau khi đổi logic.

## 95. Checklist “chuẩn bị chỉnh training config”

- Bạn biết preset hiện tại ở `train.py`.
- Bạn biết model available ở `trainer.py`.
- Bạn biết image transforms ở `vision/transforms.py`.
- Bạn biết dataset class ở `manifest_dataset.py`.
- Bạn biết artifact report nào phản ánh config cuối cùng.

## 96. Checklist “chuẩn bị deploy local inference”

- `waste_classifier.onnx` tồn tại.
- `labels.json` tồn tại.
- `model_manifest.json` tồn tại.
- Verify ONNX pass.
- Route `POST /classify/image` đã được test thủ công.
- Bạn biết input image size yêu cầu.
- Bạn biết normalization preset yêu cầu.

## 97. Test map cho Step 1

- `test_scan_marks_invalid_small_and_duplicate_images`.
- `test_run_all_writes_manifest_and_reports`.
- `test_scan_infers_normalized_label_names`.
- `test_pipeline_can_export_template_and_import_curated_labels`.
- `test_validate_labels_warns_when_manifest_has_single_class`.
- Các test này sống trong `localagent/tests/test_pipeline.py`.

## 98. Test map cho Step 2

- `test_embed_cluster_and_export_cluster_review`.
- `test_export_cluster_review_preserves_matching_saved_rows`.
- `test_export_cluster_review_resets_stale_rows`.
- `test_promote_cluster_labels_switches_manifest_to_accepted_labels_only_mode`.
- `test_promote_cluster_labels_skips_stale_review_rows`.
- `test_discovery_cli_requires_completed_step_one`.

## 99. Test map cho Step 3

- `test_build_config_uses_default_experiment_name`.
- `test_build_config_accepts_custom_experiment_name`.
- `test_build_config_applies_cpu_fast_preset`.
- `test_fit_stops_early_when_validation_loss_stalls`.
- `test_fit_can_resume_from_latest_checkpoint`.
- `test_fit_handles_keyboard_interrupt_and_saves_latest_checkpoint`.
- `test_fit_prints_epoch_progress_when_progress_bars_are_disabled`.

## 100. Test map cho artifact và export

- `test_fit_writes_evaluation_report_and_confusion_matrix`.
- `test_evaluate_uses_saved_checkpoint_and_writes_report`.
- `test_export_onnx_writes_manifest_and_export_report`.
- `test_benchmark_writes_report_for_pytorch_backend`.
- `test_benchmark_marks_rust_backend_as_unsupported`.
- `test_pseudo_label_updates_manifest_with_confidence_gate`.

## 101. Nếu chỉ được đọc 10 hàm trong Python

- `DatasetPipeline.run_all`.
- `DatasetPipeline.embed_dataset`.
- `DatasetPipeline.cluster_dataset`.
- `DatasetPipeline.export_cluster_review`.
- `DatasetPipeline.promote_cluster_labels`.
- `extract_embeddings`.
- `cluster_embeddings`.
- `WasteTrainer.fit`.
- `WasteTrainer.evaluate`.
- `WasteTrainer.export_onnx`.

## 102. Nếu chỉ được đọc 10 hàm trong Rust và UI

- Route `POST /jobs/pipeline` trong `server.rs`.
- Route `POST /jobs/training` trong `server.rs`.
- Route `GET /workflow/state` trong `server.rs`.
- Route `GET /cluster-review` trong `server.rs`.
- Route `PUT /cluster-review` trong `server.rs`.
- `ArtifactStore` trong `artifacts.rs`.
- `submitPipeline` trong `controller-actions.ts`.
- `submitTraining` trong `controller-actions.ts`.
- `saveClusterReview` trong `controller-actions.ts`.
- `fetchJson` trong `localagent.ts`.

## 103. Mẫu chuỗi lệnh CLI cho workflow chuẩn

```bash
uv run python -m localagent.data.pipeline run-all --no-progress
uv run python -m localagent.data.pipeline embed --no-progress
uv run python -m localagent.data.pipeline cluster --no-progress
uv run python -m localagent.data.pipeline export-cluster-review --output artifacts/manifests/cluster_review.csv --no-progress
uv run python -m localagent.data.pipeline promote-cluster-labels --review-file artifacts/manifests/cluster_review.csv --no-progress
uv run python -m localagent.training.train summary --no-progress
uv run python -m localagent.training.train warm-cache --training-preset cpu_fast --no-progress
uv run python -m localagent.training.train fit --training-preset cpu_fast --experiment-name baseline-waste-sorter --no-progress
uv run python -m localagent.training.train pseudo-label --experiment-name baseline-waste-sorter --no-progress
uv run python -m localagent.training.train evaluate --experiment-name baseline-waste-sorter --no-progress
uv run python -m localagent.training.train export-onnx --experiment-name baseline-waste-sorter --no-progress
uv run python -m localagent.training.train report --experiment-name baseline-waste-sorter --no-progress
```

## 104. Mẫu chuỗi lệnh cho máy yếu

```bash
uv run python -m localagent.training.train fit --training-preset cpu_fast --image-size 160 --batch-size 32 --epochs 3 --num-workers 0 --no-progress
```

- Mẫu này đã được README gợi ý ở nhiều chỗ.
- Đây là cấu hình hợp lý để smoke test workflow.

## 105. Mẫu chuỗi lệnh cho máy muốn cân bằng hơn

```bash
uv run python -m localagent.training.train fit --training-preset cpu_balanced --epochs 25 --no-progress
```

- Với preset này, bạn nên kỳ vọng runtime dài hơn `cpu_fast`.
- Bù lại backbone `resnet18` thường là mốc tham chiếu dễ hiểu hơn.

## 106. Mẫu chuỗi lệnh cho xuất ONNX an toàn

```bash
uv run python -m localagent.training.train export-onnx --experiment-name baseline-waste-sorter --no-progress
```

- Sau command này, mở `*_export.json`.
- Xác nhận `verification.verified = true`.
- Xác nhận `max_abs_diff` nhỏ.
- Xác nhận `model_manifest.json` đã được cập nhật.

## 107. Quy tắc vàng khi nhìn metric

- Không nhìn mỗi accuracy.
- Luôn nhìn macro F1.
- Luôn mở confusion matrix.
- Luôn xem lớp hiếm có bị bỏ quên không.
- Luôn nhìn class weight map nếu dataset lệch.
- Luôn so với baseline cũ nếu có.

## 108. Quy tắc vàng khi nhìn cluster

- Không nghĩ cluster là nhãn thật.
- Không nghĩ cùng cluster thì 100% cùng lớp.
- Không nghĩ outlier là lỗi thuật toán.
- Không nghĩ cluster review cũ dùng lại được sau khi cluster lại.
- Không nhầm hard cluster với fuzzy membership.

## 109. Quy tắc vàng khi nhìn artifact

- Artifact không chỉ là output để lưu.
- Artifact là giao diện giữa các tầng của hệ thống.
- Nếu bạn sửa format artifact, bạn đang sửa giao diện hệ thống.
- Hãy kiểm tra Python ghi gì.
- Hãy kiểm tra Rust đọc gì.
- Hãy kiểm tra UI kỳ vọng gì.

## 110. FAQ: Tại sao README chưa đủ

- Vì workflow nhiều bước.
- Vì README phải ngắn hơn docs chuyên sâu.
- Vì người mới cần bản đồ code, artifact và failure mode.
- Vì chỉ đọc README khó biết chức năng nằm ở đâu trong code.

## 111. FAQ: Dự án này có thật sự dùng logic mờ không

- Không, ở code hiện tại thì chưa.
- Repo hiện tại dùng CNN embeddings.
- Sau đó dùng spherical k-means.
- Sau đó dùng MAD outlier detection.
- Sau đó dùng human review.
- Nếu muốn fuzzy logic, phải thêm implementation mới.

## 112. FAQ: Tại sao phải có Rust server nếu Python đã train được

- Vì UI cần API ổn định.
- Vì cần orchestration job.
- Vì cần WebSocket stream log.
- Vì cần route inference và artifact aggregation.
- Vì frontend không nên tự biết mọi chi tiết file system Python.

## 113. FAQ: Có thể bỏ UI và chỉ dùng CLI không

- Có.
- Workflow core vẫn chạy được bằng CLI.
- UI chỉ giúp điều phối và quan sát.
- Với nghiên cứu thuật toán, CLI thường còn rõ hơn.

## 114. FAQ: Có thể bỏ Rust server và chỉ dùng Python không

- Có cho workflow CLI.
- Không tiện nếu bạn muốn dashboard và job orchestration hiện tại.
- Nhiều phần UI phụ thuộc API của Rust server.

## 115. FAQ: Tại sao lại có cả `dataset/` và `datasets/`

- Vì config có hai khái niệm đường dẫn khác nhau.
- Nhưng raw dataset của dataset pipeline hiện mặc định là `dataset/`.
- Hãy kiểm tra `summary.json.dataset_root` thay vì đoán.

## 116. FAQ: Tại sao `summary.json` rất quan trọng

- Vì nó cho bạn biết snapshot trạng thái workflow hiện tại.
- Vì nó gom nhiều chỉ số quan trọng vào một chỗ.
- Vì server và UI cũng dùng nó để quyết định trạng thái một phần.

## 117. FAQ: Tại sao phải warm cache

- Vì decode và resize ảnh lặp lại rất tốn CPU.
- Vì cache giúp nhiều epoch rẻ hơn.
- Vì repo đã có cơ chế lưu cache cục bộ để tận dụng điều này.

## 118. FAQ: Tại sao phải export ONNX nếu đã có checkpoint PyTorch

- Vì Rust inference side dùng ONNX.
- Vì ONNX là format triển khai chéo môi trường dễ hơn.
- Vì `model_manifest.json` gói hợp đồng suy luận đi cùng ONNX.

## 119. FAQ: Có nên tin pseudo-label hoàn toàn không

- Không.
- Pseudo-label là công cụ mở rộng tập nhãn với điều kiện.
- Threshold càng thấp, rủi ro càng cao.
- Nó phù hợp như bước tăng coverage có kiểm soát.

## 120. FAQ: Có nên train trước rồi mới cluster không

- Workflow hiện tại đi theo hướng cluster review mở khóa accepted labels trước.
- Bạn vẫn có thể có nhãn tay từ template import.
- Nhưng nếu hỏi workflow chuẩn của repo hiện tại, cluster review đứng trước fit chuẩn.

## 121. FAQ: Có cần GPU không

- Không bắt buộc.
- Preset hiện tại đã thiết kế để chạy trên CPU.
- Nhưng GPU vẫn giúp rút ngắn thời gian nếu môi trường hỗ trợ.

## 122. FAQ: Chọn `cpu_fast` hay `cpu_balanced`

- Cần baseline nhanh: `cpu_fast`.
- Cần mốc tham chiếu tốt hơn: `cpu_balanced`.
- Muốn đào sâu và chấp nhận chậm: `cpu_stronger`.

## 123. FAQ: Tại sao run mới không hiện trong UI

- Có thể `run_index.json` chưa refresh.
- Có thể job fail trước khi ghi report.
- Có thể `experiment_name` khác run bạn tưởng.
- Có thể UI đang giữ selected experiment cũ.

## 124. FAQ: Có cần đọc mọi file docs theo thứ tự không

- Không bắt buộc.
- Nhưng nếu mới hoàn toàn thì nên đọc `00`, `01`, `02`, `03`, rồi quay lại file này.
- File này phù hợp để thao tác.
- Các file kia phù hợp để hiểu sâu.

## 125. FAQ: Nếu tôi chỉ muốn biết code nằm ở đâu

- Step 1 ở `data/pipeline.py`.
- Step 2 thuật toán ở `data/discovery.py`.
- Step 3 ở `training/train.py` và `training/trainer.py`.
- API ở `src/bin/server.rs`.
- Job types ở `src/jobs/types.rs`.
- UI fetch ở `interface/lib/localagent.ts`.
- UI action ở `controller-actions.ts`.

## 126. FAQ: Nếu tôi muốn thay thuật toán cluster

- Hãy bắt đầu ở `discovery.py`.
- Sau đó đọc `pipeline.py` phần cluster review.
- Sau đó đọc `cluster_review.rs`.
- Sau đó đọc UI type cluster review.
- Sau đó cập nhật test pipeline.

## 127. FAQ: Nếu tôi muốn thay backbone CNN

- Hãy đọc `_replace_classifier_head()` ở `trainer.py:192`.
- Hãy kiểm tra danh sách model đã hỗ trợ.
- Hãy xem transform hiện tại có còn phù hợp không.
- Hãy cập nhật test training data nếu cần.

## 128. FAQ: Nếu tôi muốn thay định nghĩa metric

- Hãy đọc `_build_classification_report()` ở `trainer.py:678`.
- Hãy kiểm tra report JSON hiện tại.
- Hãy kiểm tra UI đang đọc metric nào.
- Hãy cập nhật tài liệu report nếu format đổi.

## 129. FAQ: Nếu tôi muốn đổi API path

- Đổi trong `server.rs`.
- Đổi trong `interface/lib/localagent.ts` hoặc `controller-actions.ts`.
- Đổi type nếu shape response đổi.
- Kiểm tra rewrite của Next.js nếu prefix đổi.

## 130. FAQ: Nếu tôi muốn bỏ Rust server

- Bạn mất dashboard và orchestration hiện tại.
- CLI Python vẫn dùng được.
- Bạn sẽ phải tự thay lớp API cho UI nếu vẫn giữ frontend.

## 131. FAQ: Tôi nên tin file nào hơn khi nhiều file mâu thuẫn

- Tin code hơn mô tả.
- Tin artifact hiện tại hơn giả định.
- Tin `summary.json.dataset_root` hơn trí nhớ.
- Tin `jobs/types.rs` hơn ví dụ payload cũ.

## 132. FAQ: Tại sao docs luôn nhắc “đừng đoán”

- Vì repo này có nhiều tầng.
- Vì có nhiều artifact trung gian.
- Vì config path có điểm dễ nhầm.
- Vì workflow gating phụ thuộc state thực tế.
- Vì nói sai một bước là người mới bị lạc ngay.

## 133. Checklist cuối cùng trước khi kết luận “repo chạy ổn”

- Step 1 chạy xong.
- Step 2 chạy xong.
- Cluster review đã được promote hoặc có luồng nhãn tay thay thế.
- Step 3 fit được.
- Evaluation report có số.
- Export ONNX verify pass.
- UI load run được.
- Route classify image hoạt động nếu bạn cần inference.

## 134. Checklist cuối cùng trước khi kết luận “docs đủ dùng cho người mới”

- Người mới biết phải chạy lệnh nào trước.
- Người mới biết file nào phải xuất hiện sau mỗi bước.
- Người mới biết lỗi thì nhìn log ở đâu.
- Người mới biết code chức năng nằm ở đâu.
- Người mới biết repo hiện chưa có fuzzy logic.
- Người mới biết pipeline CNN + clustering + review + training vận hành ra sao.

## 135. Chốt file này

- Nếu bạn làm đúng thứ tự, repo hiện tại có thể chạy end-to-end.
- Nếu bạn bỏ qua manifest và artifact, bạn sẽ rất dễ lạc.
- Nếu bạn cần hiểu sâu hơn, quay lại `docs/00` tới `docs/06`.
- Nếu bạn cần thao tác thực chiến hằng ngày, quay lại file này trước.
