# Tài liệu 01: Pipeline dữ liệu và manifest

## 1. File này dùng để làm gì

- File này giải thích toàn bộ Step 1 của workflow.
- Step 1 trong dự án này là dataset pipeline.
- Dataset pipeline không huấn luyện model.
- Dataset pipeline có nhiệm vụ biến thư mục ảnh thô thành manifest có cấu trúc.
- Dataset pipeline cũng sinh report để các bước sau biết dữ liệu đã sẵn sàng hay chưa.
- Nếu bạn không hiểu file này, bạn sẽ rất khó hiểu vì sao training bị khóa.

## 2. Vị trí code chính của Step 1

- `localagent/python/localagent/data/pipeline.py`
- Class chính là `DatasetPipeline` tại `localagent/python/localagent/data/pipeline.py:78`.
- Parser CLI nằm ở `localagent/python/localagent/data/pipeline.py:1361`.
- Hàm `main` dispatch command nằm ở `localagent/python/localagent/data/pipeline.py:1429`.

## 3. Dataset pipeline đang nhận đầu vào từ đâu

- Đầu vào mặc định là `localagent/dataset/`.
- Giá trị mặc định nằm ở `DatasetPipelineConfig.raw_dataset_dir`.
- Vị trí cấu hình là `localagent/python/localagent/config/settings.py:111`.
- Đây là thư mục ảnh thô.
- Pipeline không chỉnh sửa ảnh gốc tại đây.

## 4. Dataset pipeline tạo đầu ra ở đâu

- Manifest nằm ở `localagent/artifacts/manifests/`.
- Report nằm ở `localagent/artifacts/reports/`.
- Cụ thể path mặc định được expose qua property trong `DatasetPipelineConfig`.
- Các property đó nằm ở `localagent/python/localagent/config/settings.py`.

## 5. Lý do phải có manifest

- Training không đọc trực tiếp thư mục ảnh theo kiểu thư mục nhãn con.
- Training đọc một bảng manifest.
- Bảng manifest lưu path ảnh.
- Bảng manifest lưu nhãn hiện tại.
- Bảng manifest lưu nguồn nhãn.
- Bảng manifest lưu trạng thái annotation.
- Bảng manifest lưu split.
- Bảng manifest lưu cluster metadata.
- Bảng manifest lưu hash nội dung để phát hiện duplicate.
- Bảng manifest là hợp đồng dữ liệu chung cho scan, discovery, training và report.

## 6. Các subcommand của Step 1

- `scan`
- `split`
- `report`
- `run-all`
- `export-labeling-template`
- `import-labels`
- `validate-labels`
- `embed`
- `cluster`
- `export-cluster-review`
- `promote-cluster-labels`

## 7. Subcommand nào thật sự thuộc Step 1 nền tảng

- `scan`
- `split`
- `report`
- `run-all`
- `export-labeling-template`
- `import-labels`
- `validate-labels`

## 8. Subcommand nào là discovery nhưng vẫn đi qua cùng CLI

- `embed`
- `cluster`
- `export-cluster-review`
- `promote-cluster-labels`

## 9. Vì sao discovery vẫn nằm chung file pipeline

- Vì tất cả các bước đó đều thao tác trên cùng một manifest.
- Vì cùng một class `DatasetPipeline` quản lý logic metadata của ảnh.
- Vì training chỉ được mở khóa sau khi manifest đạt trạng thái mong muốn.

## 10. Chạy nhanh Step 1 đầy đủ

```powershell
cd localagent
uv run python -m localagent.data.pipeline run-all
```

## 11. `run-all` làm gì

- Gọi `scan`.
- Gọi `assign_splits`.
- Ghi manifest parquet.
- Ghi manifest csv.
- Ghi `summary.json`.
- Ghi các CSV summary phụ.

## 12. Vị trí code của `run-all`

- `DatasetPipeline.run_all` ở `localagent/python/localagent/data/pipeline.py:639`.

## 13. Quy trình nội bộ của `scan`

- Dò toàn bộ file ảnh hợp lệ trong `raw_dataset_dir`.
- Mỗi file được inspect metadata.
- Sau khi inspect xong toàn bộ, pipeline đánh dấu duplicate theo hash nội dung.
- Sau đó pipeline áp quality flag.
- Cuối cùng pipeline dựng `polars.DataFrame` theo schema chuẩn.

## 14. Vị trí code của từng bước `scan`

- `_iter_image_paths` ở `localagent/python/localagent/data/pipeline.py:748`
- `_inspect_image` ở `localagent/python/localagent/data/pipeline.py:763`
- `_mark_duplicates` ở `localagent/python/localagent/data/pipeline.py:826`
- `_apply_quality_flags` ở `localagent/python/localagent/data/pipeline.py:837`
- `_frame_from_records` ở `localagent/python/localagent/data/pipeline.py:856`

## 15. `_iter_image_paths` làm gì

- Kiểm tra thư mục dataset có tồn tại không.
- Nếu không tồn tại thì ném `FileNotFoundError`.
- Dùng `rglob("*")` để quét đệ quy.
- Chỉ giữ file có extension hợp lệ.
- Sắp xếp kết quả theo relative path viết thường.

## 16. Extension mặc định được chấp nhận

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`

## 17. Vị trí cấu hình extension hợp lệ

- `DatasetPipelineConfig.allowed_extensions` trong `localagent/python/localagent/config/settings.py`.

## 18. `_inspect_image` làm gì

- Tính `relative_path`.
- Suy nhãn từ filename nếu được bật.
- Đọc kích thước ảnh bằng OpenCV.
- Ghi `width`, `height`, `channels`.
- Ghi trạng thái decode.
- Tính hash nội dung file.
- Khởi tạo toàn bộ field còn lại của record với giá trị mặc định phù hợp.

## 19. Vì sao `_inspect_image` chưa đánh dấu duplicate ngay

- Vì duplicate phải so sánh hash của tất cả file.
- Do đó phải inspect hết rồi mới `_mark_duplicates`.

## 20. `_mark_duplicates` làm gì

- Duy trì map `content_hash -> canonical_sample_id`.
- File đầu tiên với hash đó là bản gốc.
- File sau với cùng hash bị đánh dấu duplicate.
- Duplicate lưu `duplicate_of = canonical_sample_id`.

## 21. `_apply_quality_flags` làm gì

- Nếu decode lỗi thì gắn `quarantine_reason = decode_failed`.
- Nếu ảnh quá nhỏ thì gắn `quarantine_reason = too_small`.
- Nếu ảnh trùng thì gắn `quarantine_reason = duplicate`.
- Nếu không có lý do cách ly thì `is_valid = True`.

## 22. Thế nào là ảnh quá nhỏ

- Nhỏ hơn `min_width`.
- Hoặc nhỏ hơn `min_height`.
- Mặc định `min_width = 32`.
- Mặc định `min_height = 32`.

## 23. Vị trí cấu hình ngưỡng ảnh nhỏ

- `localagent/python/localagent/config/settings.py`
- `DatasetPipelineConfig.min_width`
- `DatasetPipelineConfig.min_height`

## 24. `_infer_label` làm gì

- Nếu `infer_labels_from_filename` bị tắt thì trả `unknown`.
- Nếu bật, nó dùng regex `LABEL_PATTERN`.
- Regex mặc định bỏ phần số thứ tự ở cuối tên.
- Sau đó gọi `_normalize_label`.

## 25. Regex suy nhãn từ filename

- Regex là `(.+?)[_\\- ]\\d+$`.
- Nghĩa là lấy phần trước hậu tố số như `_1`, `-2`, ` 3`.
- Ví dụ `Glass_12.jpg` có raw label là `Glass`.
- Ví dụ `Miscellaneous Trash_1.jpg` có raw label là `Miscellaneous Trash`.

## 26. `_normalize_label` làm gì

- Đưa chữ về `casefold`.
- Thay chuỗi không phải chữ hoặc số bằng `_`.
- Bỏ `_` ở đầu và cuối.

## 27. Hệ quả của `_normalize_label`

- `Glass Bottle` thành `glass_bottle`.
- `MISC-TRASH` thành `misc_trash`.
- `PAPER` thành `paper`.

## 28. `_build_sample_id` làm gì

- Tạo `safe_stem` từ tên file.
- Tạo digest SHA1 ngắn từ `relative_path`.
- Ghép thành `safe_stem-digest`.
- Điều này giúp sample id ổn định và khó trùng.

## 29. `_hash_file` làm gì

- Mở file ở chế độ nhị phân.
- Đọc theo chunk 1MB.
- Tính hash bằng thuật toán cấu hình.
- Mặc định là `sha256`.

## 30. `assign_splits` làm gì

- Chuẩn hóa frame về đủ cột manifest.
- Lấy các sample training-ready hiện tại.
- Gom sample id theo từng label.
- Shuffle theo random seed.
- Cắt theo tỉ lệ train/val/test trên từng lớp.
- Ghi split cuối vào cột `split`.
- Sample không đủ điều kiện train sẽ bị `split = excluded`.

## 31. Vị trí code của `assign_splits`

- `localagent/python/localagent/data/pipeline.py:105`

## 32. Vì sao split được gán theo từng lớp

- Để tránh một lớp bị dồn hết vào một split.
- Đây là stratified split đơn giản dựa trên label hiện tại.

## 33. `has_accepted_labels` ảnh hưởng split thế nào

- Nếu manifest chưa có accepted labels, các nhãn `filename` vẫn có thể đi train.
- Nếu manifest đã có accepted labels, sample chỉ được train nếu nhãn đến từ nguồn accepted.
- Vì thế split có thể đổi sau khi import curated labels hoặc promote cluster labels.

## 34. Nguồn nhãn nào được xem là accepted

- `curated`
- `cluster_review`
- `model_pseudo`

## 35. Trạng thái annotation nào được xem là accepted

- `labeled`
- `pseudo_labeled`

## 36. Vị trí code định nghĩa accepted source

- `ACCEPTED_LABEL_SOURCES` trong `localagent/python/localagent/data/pipeline.py`.

## 37. Vị trí code định nghĩa accepted annotation status

- `ACCEPTED_ANNOTATION_STATUSES` trong `localagent/python/localagent/data/pipeline.py`.

## 38. `generate_reports` làm gì

- Tính thống kê tổng file.
- Tính thống kê file hợp lệ.
- Tính thống kê duplicate, decode fail, too small.
- Tính split count.
- Tính label count.
- Tính label source count.
- Tính review status count.
- Tính extension count.
- Tính dimension stats.
- Tính snapshot cluster preview.
- Ghi `summary.json`.
- Ghi các CSV thống kê phụ.

## 39. Vị trí code của `generate_reports`

- `localagent/python/localagent/data/pipeline.py:158`

## 40. Report JSON quan trọng nhất của Step 1

- `localagent/artifacts/reports/summary.json`

## 41. CSV phụ được sinh bởi Step 1

- `split_summary.csv`
- `quality_summary.csv`
- `extension_summary.csv`
- `label_summary.csv`

## 42. Ý nghĩa của `summary.json`

- Đây là snapshot trạng thái dữ liệu mà toàn hệ thống cùng nhìn vào.
- Workflow state ở Rust dùng file này.
- Training unlock logic cũng dựa gián tiếp vào file này.
- UI hiển thị dashboard dữ liệu cũng dựa vào file này.

## 43. `validate_labels` làm gì

- Kiểm tra số lớp hiện tại.
- Kiểm tra có sample training-ready hay chưa.
- Cảnh báo khi toàn bộ nhãn vẫn chỉ từ filename.
- Cảnh báo khi vẫn còn `unlabeled`.
- Trả `class_names`, `train_label_counts`, `warnings`.

## 44. Vị trí code của `validate_labels`

- `localagent/python/localagent/data/pipeline.py:535`

## 45. `export_labeling_template` làm gì

- Tạo một CSV theo từng sample để người dùng chỉnh nhãn.
- CSV này chứa `sample_id`, `relative_path`, `suggested_label`, `current_label`, `status`, `notes`.
- Đây là cách annotate thủ công ở mức sample.

## 46. Vị trí code của `export_labeling_template`

- `localagent/python/localagent/data/pipeline.py:397`

## 47. `import_labels` làm gì

- Đọc file `.csv`, `.json` hoặc `.jsonl`.
- Match theo `sample_id`.
- Cập nhật `label`, `label_source`, `annotation_status`, `review_status`, `annotated_at`.
- Re-assign split sau khi cập nhật.
- Ghi lại manifest và report.

## 48. Vị trí code của `import_labels`

- `localagent/python/localagent/data/pipeline.py:464`

## 49. Định dạng file labels được chấp nhận

- `.csv`
- `.json`
- `.jsonl`
- `.ndjson`

## 50. Vị trí code đọc từng định dạng labels

- `_read_label_csv` ở `localagent/python/localagent/data/pipeline.py:1338`
- `_read_label_json` ở `:1342`
- `_read_label_jsonl` ở `:1348`

## 51. Cảnh báo khi import label

- Nếu file label chứa `sample_id` không tồn tại trong manifest, pipeline sẽ báo lỗi.
- Nếu record bị đánh dấu `labeled` nhưng label rỗng, pipeline sẽ báo lỗi.

## 52. Phân tích manifest schema

- Manifest schema là cốt lõi của Step 1.
- Nó được định nghĩa hằng `MANIFEST_SCHEMA`.
- Vị trí là đầu file `localagent/python/localagent/data/pipeline.py`.

## 53. Cột `sample_id`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: định danh ổn định của sample.
- Nguồn sinh: `_build_sample_id`.
- Ý nghĩa thực tế: khóa join giữa manifest, label template và các bước cập nhật nhãn.

## 54. Cột `image_path`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: đường dẫn tuyệt đối hoặc đầy đủ tới ảnh nguồn.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: trainer và discovery dùng path này để đọc ảnh.

## 55. Cột `relative_path`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: đường dẫn tương đối so với thư mục dataset.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: UI dùng giá trị này để build URL `GET /dataset/image`.

## 56. Cột `file_name`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: tên file gốc.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: hiển thị thân thiện cho người dùng.

## 57. Cột `extension`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: phần mở rộng file.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: thống kê định dạng dữ liệu.

## 58. Cột `file_size`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: kích thước file tính theo byte.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: hỗ trợ kiểm tra dữ liệu bất thường.

## 59. Cột `width`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: chiều rộng ảnh.
- Nguồn sinh: OpenCV decode.
- Ý nghĩa thực tế: dùng để phát hiện ảnh quá nhỏ và thống kê dataset.

## 60. Cột `height`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: chiều cao ảnh.
- Nguồn sinh: OpenCV decode.
- Ý nghĩa thực tế: dùng để phát hiện ảnh quá nhỏ và thống kê dataset.

## 61. Cột `channels`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: số kênh của ảnh.
- Nguồn sinh: OpenCV decode.
- Ý nghĩa thực tế: giúp debug dữ liệu grayscale hay RGB.

## 62. Cột `decode_ok`

- Kiểu dữ liệu: `pl.Boolean`.
- Mục đích: decode ảnh có thành công không.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: ảnh decode lỗi sẽ bị loại khỏi training-ready.

## 63. Cột `decode_error`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: ghi lỗi decode nếu có.
- Nguồn sinh: `_inspect_image`.
- Ý nghĩa thực tế: phục vụ debug file hỏng.

## 64. Cột `raw_label`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: nhãn thô đọc trực tiếp từ filename trước normalize.
- Nguồn sinh: `_infer_label`.
- Ý nghĩa thực tế: lưu lại dấu vết trước chuẩn hóa.

## 65. Cột `curated_label`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: nhãn con người gán trực tiếp.
- Nguồn sinh: `import_labels` hoặc các bước cập nhật sau.
- Ý nghĩa thực tế: phân biệt với nhãn yếu hoặc pseudo-label.

## 66. Cột `suggested_label`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: nhãn gợi ý hiện tại.
- Nguồn sinh: ban đầu thường từ filename, về sau có thể từ pseudo-label hoặc cluster review.
- Ý nghĩa thực tế: hỗ trợ người dùng review nhanh.

## 67. Cột `suggested_label_source`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: nguồn của nhãn gợi ý.
- Nguồn sinh: `filename`, `cluster_review`, `model_pseudo`.
- Ý nghĩa thực tế: người dùng biết gợi ý đến từ đâu.

## 68. Cột `label`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: nhãn hiện đang được dùng cho workflow.
- Nguồn sinh: filename, curated, cluster_review hoặc model_pseudo.
- Ý nghĩa thực tế: đây là cột mà trainer sẽ nhìn.

## 69. Cột `label_source`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: lưu nguồn gốc của `label`.
- Nguồn sinh: pipeline cập nhật theo từng bước.
- Ý nghĩa thực tế: cực kỳ quan trọng để biết nhãn có được chấp nhận hay chưa.

## 70. Cột `annotation_status`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: trạng thái annotation của sample.
- Giá trị thường gặp: `inferred`, `labeled`, `unlabeled`, `excluded`, `pseudo_labeled`.
- Ý nghĩa thực tế: dùng để loại sample khỏi training nếu chưa đủ điều kiện.

## 71. Cột `annotated_at`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: timestamp lúc sample được gán hoặc cập nhật nhãn.
- Nguồn sinh: `import_labels`, `promote_cluster_labels`, `pseudo_label`.
- Ý nghĩa thực tế: audit trail đơn giản.

## 72. Cột `pseudo_label_score`

- Kiểu dữ liệu: `pl.Float64`.
- Mục đích: điểm confidence của pseudo-label gần nhất.
- Nguồn sinh: `WasteTrainer.pseudo_label`.
- Ý nghĩa thực tế: giúp phân tích chất lượng pseudo-label.

## 73. Cột `pseudo_label_margin`

- Kiểu dữ liệu: `pl.Float64`.
- Mục đích: chênh lệch giữa xác suất top 1 và top 2.
- Nguồn sinh: `WasteTrainer.pseudo_label`.
- Ý nghĩa thực tế: dùng làm cổng chấp nhận pseudo-label.

## 74. Cột `review_status`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: trạng thái review chi tiết hơn.
- Giá trị thường gặp: `unreviewed`, `curated`, `cluster_accepted`, `cluster_excluded`, `pseudo_accepted`, `pseudo_rejected`.
- Ý nghĩa thực tế: dễ debug tiến trình human-in-the-loop.

## 75. Cột `cluster_id`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: id cụm của sample.
- Nguồn sinh: `cluster_dataset`.
- Ý nghĩa thực tế: dùng cho cluster review và phân tích discovery.

## 76. Cột `cluster_distance`

- Kiểu dữ liệu: `pl.Float64`.
- Mục đích: khoảng cách của sample tới centroid cụm.
- Nguồn sinh: `cluster_embeddings`.
- Ý nghĩa thực tế: hỗ trợ đánh giá sample có điển hình trong cụm hay không.

## 77. Cột `cluster_size`

- Kiểu dữ liệu: `pl.Int64`.
- Mục đích: kích thước cụm chứa sample.
- Nguồn sinh: `cluster_embeddings`.
- Ý nghĩa thực tế: UI có thể dùng để ưu tiên review cụm lớn.

## 78. Cột `is_cluster_outlier`

- Kiểu dữ liệu: `pl.Boolean`.
- Mục đích: sample có bị xem là outlier trong cụm không.
- Nguồn sinh: `_detect_cluster_outliers`.
- Ý nghĩa thực tế: outlier có thể bị bỏ qua khi promote nhãn cụm.

## 79. Cột `content_hash`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: hash nội dung file ảnh.
- Nguồn sinh: `_hash_file`.
- Ý nghĩa thực tế: phát hiện duplicate chắc chắn hơn tên file.

## 80. Cột `is_duplicate`

- Kiểu dữ liệu: `pl.Boolean`.
- Mục đích: đánh dấu file trùng nội dung.
- Nguồn sinh: `_mark_duplicates`.
- Ý nghĩa thực tế: duplicate bị loại khỏi `is_valid`.

## 81. Cột `duplicate_of`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: sample id chuẩn mà sample hiện tại trùng theo nội dung.
- Nguồn sinh: `_mark_duplicates`.
- Ý nghĩa thực tế: truy ngược file gốc.

## 82. Cột `is_too_small`

- Kiểu dữ liệu: `pl.Boolean`.
- Mục đích: ảnh có nhỏ hơn ngưỡng hay không.
- Nguồn sinh: `_apply_quality_flags`.
- Ý nghĩa thực tế: ảnh quá nhỏ bị cách ly.

## 83. Cột `is_valid`

- Kiểu dữ liệu: `pl.Boolean`.
- Mục đích: sample có hợp lệ cho workflow tiếp theo không.
- Nguồn sinh: `_apply_quality_flags`.
- Ý nghĩa thực tế: filter nền tảng cho discovery và training.

## 84. Cột `quarantine_reason`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: lý do sample bị loại khỏi hợp lệ.
- Giá trị thường gặp: `decode_failed`, `too_small`, `duplicate`.
- Ý nghĩa thực tế: phân tích chất lượng dữ liệu.

## 85. Cột `split`

- Kiểu dữ liệu: `pl.String`.
- Mục đích: sample thuộc `train`, `val`, `test` hay `excluded`.
- Nguồn sinh: `assign_splits`.
- Ý nghĩa thực tế: trainer dựa vào đây để build dataset.

## 86. Vì sao `_ensure_manifest_columns` quan trọng

- Manifest có thể được tạo từ các phiên bản code khác nhau.
- `_ensure_manifest_columns` giúp backfill cột mới.
- Nhờ đó code vẫn đọc được manifest cũ tương đối an toàn.

## 87. Vị trí code của `_ensure_manifest_columns`

- `localagent/python/localagent/data/pipeline.py:680`

## 88. Quy tắc xác định training-ready frame của dataset pipeline

- `is_valid` phải là `True`.
- `label` không được là `unknown`.
- `annotation_status` phải nằm trong nhóm cho phép.
- Nếu manifest đã có accepted labels, `label_source` cũng phải là accepted source.

## 89. Vị trí code của `_training_ready_frame`

- `localagent/python/localagent/data/pipeline.py:1174`

## 90. `summary.json` hiện tại của repo nói gì

- Tổng file: `9999`.
- File hợp lệ: `9989`.
- File duplicate: `10`.
- File training-ready: `9769`.
- Chế độ training: `accepted_labels_only`.
- Label nguồn accepted: `cluster_review` và `model_pseudo`.

## 91. `split_counts` hiện tại của repo nói gì

- `train = 7814`
- `val = 975`
- `test = 980`
- `excluded = 230`

## 92. `label_counts` hiện tại của repo nói gì

- `folk = 464`
- `glass = 8776`
- `paper = 529`
- `unknown = 230`

## 93. `annotation_status_counts` hiện tại của repo nói gì

- `labeled = 9199`
- `pseudo_labeled = 570`
- `unlabeled = 230`

## 94. `review_status_counts` hiện tại của repo nói gì

- `cluster_accepted = 9199`
- `pseudo_accepted = 570`
- `pseudo_rejected = 220`
- `pending_review = 10`

## 95. Điều đó cho bạn biết gì

- Phần lớn dữ liệu đã được chấp nhận qua cluster review.
- Một phần nhỏ được mở rộng bằng pseudo-label.
- Một phần rất nhỏ vẫn còn chưa được chấp nhận.
- Tập dữ liệu hiện nghiêng mạnh về lớp `glass`.

## 96. Ý nghĩa của `pending_review = 10`

- Có một số sample chưa đi qua vòng review hoàn chỉnh.
- Chúng sẽ không được xem là accepted labels.

## 97. `workflow_state` trong Step 1 dùng để làm gì

- Biết Step 1 đã complete hay chưa.
- Biết embedding artifact đã tồn tại chưa.
- Biết cluster review đã sẵn sàng chưa.
- Biết bao nhiêu cluster đã được review.

## 98. Vị trí code của `workflow_state`

- `DatasetPipeline.workflow_state` ở `localagent/python/localagent/data/pipeline.py:582`
- `WorkflowState::from_config` ở `localagent/src/workflow.rs`

## 99. CLI flags chung của Step 1

- `--raw-dir`
- `--manifest-dir`
- `--report-dir`
- `--min-width`
- `--min-height`
- `--train-ratio`
- `--val-ratio`
- `--test-ratio`
- `--seed`
- `--num-clusters`
- `--no-filename-labels`
- `--no-progress`

## 100. Flag `--raw-dir`

- Mục đích: đổi thư mục ảnh đầu vào.
- Khi nào dùng: khi bạn không muốn dùng `dataset/` mặc định.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1367`.

## 101. Flag `--manifest-dir`

- Mục đích: đổi nơi ghi manifest.
- Khi nào dùng: khi bạn muốn tách artifact cho bộ dữ liệu khác.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1367`.

## 102. Flag `--report-dir`

- Mục đích: đổi nơi ghi report.
- Khi nào dùng: khi bạn muốn tách report ra chỗ khác.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1368`.

## 103. Flag `--min-width`

- Mục đích: đổi ngưỡng chiều rộng nhỏ nhất.
- Khi nào dùng: khi dataset của bạn có nhiều ảnh thumbnail.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1369`.

## 104. Flag `--min-height`

- Mục đích: đổi ngưỡng chiều cao nhỏ nhất.
- Khi nào dùng: khi dataset của bạn có nhiều ảnh thumbnail.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1370`.

## 105. Flag `--train-ratio`

- Mục đích: đổi tỉ lệ train.
- Khi nào dùng: khi cần nhiều dữ liệu train hơn.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1371`.

## 106. Flag `--val-ratio`

- Mục đích: đổi tỉ lệ validation.
- Khi nào dùng: khi cần theo dõi ổn định validation kỹ hơn.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1372`.

## 107. Flag `--test-ratio`

- Mục đích: đổi tỉ lệ test.
- Khi nào dùng: khi muốn dành nhiều mẫu hơn cho đánh giá cuối.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1373`.

## 108. Flag `--seed`

- Mục đích: đổi random seed cho split và clustering heuristics.
- Khi nào dùng: khi bạn muốn reproducibility hoặc muốn thử split khác.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1374`.

## 109. Flag `--num-clusters`

- Mục đích: chỉ định số cụm cho discovery.
- Khi nào dùng: khi heuristic không phù hợp với dataset của bạn.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1375`.

## 110. Flag `--no-filename-labels`

- Mục đích: tắt hoàn toàn suy nhãn từ tên file.
- Khi nào dùng: khi filename không phản ánh nhãn.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1376`.

## 111. Flag `--no-progress`

- Mục đích: tắt progress bar.
- Khi nào dùng: khi chạy trên môi trường log text hoặc CI.
- Chỗ parse: `localagent/python/localagent/data/pipeline.py:1377`.

## 112. Command `scan`

- Chỉ quét và ghi manifest ban đầu.
- Chưa gán split.
- Phù hợp khi bạn muốn inspect dữ liệu trước.

## 113. Command `split`

- Đọc manifest đã có.
- Gán lại split theo dữ liệu hiện tại.
- Hữu ích sau khi đổi nhãn hoặc đổi tỉ lệ split.

## 114. Command `report`

- Đọc manifest.
- Sinh lại `summary.json` và CSV report phụ.
- Hữu ích sau khi chỉnh tay manifest hoặc sau các bước discovery.

## 115. Command `run-all`

- Là command nên dùng đầu tiên cho người mới.
- Nó gom `scan + split + report`.

## 116. Command `export-labeling-template`

- Dùng khi bạn muốn con người gán nhãn ở mức sample.
- Tạo `labeling_template.csv`.

## 117. Command `import-labels`

- Dùng khi bạn đã sửa template hoặc có file gán nhãn riêng.
- Nó cập nhật manifest theo `sample_id`.

## 118. Command `validate-labels`

- Dùng để kiểm xem dữ liệu hiện tại đã đủ để train chưa.
- Cực kỳ hữu ích trước khi chạy `fit`.

## 119. Ví dụ workflow Step 1 tối thiểu

```powershell
cd localagent
uv run python -m localagent.data.pipeline run-all
uv run python -m localagent.data.pipeline validate-labels
```

## 120. Ví dụ workflow Step 1 khi filename không đáng tin

```powershell
cd localagent
uv run python -m localagent.data.pipeline run-all --no-filename-labels
uv run python -m localagent.data.pipeline export-labeling-template
uv run python -m localagent.data.pipeline import-labels --labels-file artifacts/manifests/labeling_template.csv
uv run python -m localagent.data.pipeline validate-labels
```

## 121. Ví dụ workflow Step 1 khi muốn đổi split

```powershell
cd localagent
uv run python -m localagent.data.pipeline split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
uv run python -m localagent.data.pipeline report
```

## 122. Các lỗi thường gặp ở Step 1

- Thư mục `dataset/` chưa tồn tại.
- Ảnh bị hỏng.
- Ảnh quá nhỏ.
- Ảnh trùng nội dung.
- Tên file không phản ánh nhãn.
- File label import có `sample_id` không khớp manifest.
- Số lớp sau khi import vẫn chỉ có một lớp duy nhất.

## 123. Dự án có test gì cho Step 1

- `test_scan_marks_invalid_small_and_duplicate_images`
- `test_run_all_writes_manifest_and_reports`
- `test_scan_infers_normalized_label_names`
- `test_pipeline_can_export_template_and_import_curated_labels`
- `test_validate_labels_warns_when_manifest_has_single_class`

## 124. Những test đó cho bạn biết điều gì

- Duplicate bị loại khỏi `is_valid`.
- Ảnh nhỏ bị loại khỏi `is_valid`.
- Report phải được ghi ra file.
- Label từ filename được normalize đúng.
- Import curated labels phải cập nhật manifest đúng.
- Validation phải cảnh báo nếu chỉ có một lớp.

## 125. Điều cần nhớ nhất của Step 1

- Step 1 tạo manifest chuẩn hóa.
- Step 1 không phải chỉ là “scan file”.
- Step 1 chính là nền móng dữ liệu cho toàn bộ workflow.
- Mọi bước về sau đều phụ thuộc vào manifest này.

## 126. Kết luận ngắn

- Nếu Step 1 sai, discovery sẽ sai.
- Nếu Step 1 sai, training sẽ sai.
- Nếu Step 1 thiếu report, UI sẽ khóa bước sau.
- Vì vậy đọc `pipeline.py` thật kỹ là khoản đầu tư tốt nhất khi mới vào dự án.

## 127. Phụ lục A: bản đồ hàm Step 1 trong code

- `DatasetPipeline.__init__()` nằm ở `localagent/python/localagent/data/pipeline.py:79`.
- Hàm khởi tạo nhận `DatasetPipelineConfig` và khóa các đường dẫn artifact cho toàn bộ Step 1.
- `scan()` nằm ở `localagent/python/localagent/data/pipeline.py:82`.
- `scan()` là wrapper ngắn gọi `run_scan()` rồi trả `polars.DataFrame`.
- `run_scan()` nằm ở `localagent/python/localagent/data/pipeline.py:100`.
- `run_scan()` lặp qua ảnh bằng `_iter_image_paths()` tại `pipeline.py:748`.
- `_inspect_image()` ở `pipeline.py:763` đọc metadata ảnh, chiều rộng, chiều cao, số kênh và lỗi decode.
- `_hash_file()` ở `pipeline.py:902` tạo hash để phát hiện trùng nội dung.
- `_mark_duplicates()` ở `pipeline.py:826` gắn cờ `is_duplicate` và `duplicate_of`.
- `_apply_quality_flags()` ở `pipeline.py:837` áp điều kiện ảnh quá nhỏ, ảnh lỗi, ảnh cần quarantine.
- `_infer_label()` ở `pipeline.py:878` cố suy luận nhãn từ tên file hoặc thư mục.
- `_normalize_label()` ở `pipeline.py:890` chuẩn hóa tên nhãn thành định dạng thống nhất.
- `_build_sample_id()` ở `pipeline.py:894` sinh `sample_id` ổn định từ đường dẫn tương đối.
- `_frame_from_records()` ở `pipeline.py:856` gom record thành manifest frame.
- `_ensure_manifest_columns()` ở `pipeline.py:680` đảm bảo schema luôn đủ cột.
- `assign_splits()` nằm ở `pipeline.py:105`.
- `assign_splits()` chia train/val/test theo ratio trong config và seed cố định.
- `_resolve_split()` ở `pipeline.py:1119` quyết định split cuối cùng cho từng record.
- `run_split()` nằm ở `pipeline.py:152`.
- `run_split()` ghi manifest đã có cột `split` về đĩa.
- `generate_reports()` nằm ở `pipeline.py:158`.
- `generate_reports()` tính summary, label summary, split summary, extension summary và quality summary.
- `run_report()` ở `pipeline.py:205` là wrapper để đọc manifest rồi gọi `generate_reports()`.
- `_write_count_table()` ở `pipeline.py:1170` ghi các bảng CSV tổng hợp.
- `_count_mapping()` ở `pipeline.py:1136` biến bảng đếm về dict gọn cho JSON.
- `_dimension_stats()` ở `pipeline.py:1150` tính min, max, median của kích thước ảnh.
- `write_manifest()` nằm ở `pipeline.py:653`.
- `write_manifest()` ghi cả `dataset_manifest.parquet` lẫn `dataset_manifest.csv`.
- `load_manifest()` ở `pipeline.py:646` là cửa vào chuẩn khi mọi bước sau cần manifest.
- `class_names()` ở `pipeline.py:659` đọc danh sách lớp train được.
- `train_label_counts()` ở `pipeline.py:668` đếm mẫu train theo lớp.
- `_training_ready_frame()` ở `pipeline.py:1174` lọc frame đủ điều kiện cho training.
- `export_labeling_template()` nằm ở `pipeline.py:397`.
- `export_labeling_template()` tạo CSV template cho gán nhãn thủ công.
- `_read_label_csv()` ở `pipeline.py:1338` đọc template CSV do người dùng cập nhật.
- `_read_label_json()` ở `pipeline.py:1342` hỗ trợ import từ JSON.
- `_read_label_jsonl()` ở `pipeline.py:1348` hỗ trợ import từ JSONL.
- `import_labels()` ở `pipeline.py:464` nhận file gán nhãn và cập nhật manifest.
- `_apply_label_update()` ở `pipeline.py:1082` áp từng cập nhật nhãn vào frame.
- `validate_labels()` ở `pipeline.py:535` kiểm tra dữ liệu nhãn có đủ điều kiện cho training hay chưa.
- `workflow_state()` ở `pipeline.py:582` tổng hợp trạng thái mở khóa workflow cho UI và server.
- `validate_discovery_command()` ở `pipeline.py:608` chặn Step 2 nếu Step 1 chưa đủ.
- `run_all()` ở `pipeline.py:639` chạy `scan -> split -> report` trong một lệnh.
- `build_parser()` ở `pipeline.py:1361` khai báo toàn bộ CLI của dataset pipeline.
- `build_config()` ở `pipeline.py:1400` ánh xạ CLI flags sang `DatasetPipelineConfig`.
- `main()` ở `pipeline.py:1429` là entry point của `python -m localagent.data.pipeline`.

## 128. Phụ lục B: cookbook lệnh Step 1 và khi nào dùng

- Dùng `scan` khi bạn vừa thay đổi dataset gốc và chưa cần chia split ngay.
- Dùng `split` khi manifest đã có đủ sample hợp lệ nhưng bạn muốn đổi ratio hoặc seed.
- Dùng `report` khi manifest đã tồn tại và bạn chỉ muốn refresh báo cáo.
- Dùng `run-all` khi đây là lần chuẩn bị dataset tiêu chuẩn nhất.
- Dùng `export-labeling-template` khi cần đưa danh sách ảnh chưa rõ nhãn cho người review.
- Dùng `import-labels` khi bạn đã nhận file nhãn từ bên ngoài.
- Dùng `validate-labels` trước khi mở Step 3 nếu nghi manifest bị lệch lớp.
- Dùng `embed` chỉ sau khi `workflow_state()` cho biết Step 1 đã xong.
- Dùng `cluster` chỉ sau khi embeddings đã có.
- Dùng `export-cluster-review` khi cần tạo file review theo cluster.
- Dùng `promote-cluster-labels` khi review cluster đã được chấp nhận.
- `--raw-dir` đổi thư mục ảnh nguồn, map vào `PipelineJobRequest.raw_dir` ở `localagent/src/jobs/types.rs:121`.
- `--manifest-dir` đổi nơi ghi manifest, map vào `PipelineJobRequest.manifest_dir` ở `jobs/types.rs:122`.
- `--report-dir` đổi nơi ghi report, map vào `PipelineJobRequest.report_dir` ở `jobs/types.rs:123`.
- `--min-width` đặt ngưỡng chiều rộng tối thiểu, map vào `PipelineJobRequest.min_width` ở `jobs/types.rs:124`.
- `--min-height` đặt ngưỡng chiều cao tối thiểu, map vào `PipelineJobRequest.min_height` ở `jobs/types.rs:125`.
- `--train-ratio` đặt tỷ lệ train.
- `--val-ratio` đặt tỷ lệ validation.
- `--test-ratio` đặt tỷ lệ test.
- `--seed` cố định tách split để tái lập.
- `--num-clusters` chỉ có ý nghĩa ở pha clustering nhưng vẫn đi cùng pipeline catalog.
- `--infer-filename-labels` bật suy luận nhãn từ file name nếu cấu trúc thư mục chưa sạch.
- `--labels-file` dùng khi import labels.
- `--review-file` dùng khi export hoặc promote cluster review.
- `--output` dùng cho các lệnh cần ghi file đích như template hoặc cluster review.
- `--no-progress` phù hợp khi chạy qua job server để log gọn hơn.
- Qua UI, `submitPipeline()` ở `interface/components/localagent/controller-actions.ts:297` sẽ đóng gói các field này.
- Phía server, `POST /jobs/pipeline` nằm ở `localagent/src/bin/server.rs:149`.
- Server chuyển payload thành argv ở `localagent/src/jobs/commands.rs`.
- Job runtime thực thi lệnh Python thông qua `uv` ở `localagent/src/jobs/runtime.rs`.

## 129. Phụ lục C: checklist kiểm tra manifest sau Step 1

- Kiểm tra `summary.json` có tồn tại trong `localagent/artifacts/reports/`.
- Kiểm tra `dataset_manifest.parquet` có tồn tại trong `localagent/artifacts/manifests/`.
- Kiểm tra `dataset_manifest.csv` có tồn tại để con người đọc nhanh.
- Kiểm tra `total_files` có gần đúng số ảnh nguồn mong đợi.
- Kiểm tra `valid_files` có thấp bất thường không.
- Kiểm tra `invalid_files` có tăng vọt sau lần bổ sung dữ liệu mới không.
- Kiểm tra `duplicate_files` có cao bất thường không.
- Kiểm tra `training_ready_files` có giảm mạnh không.
- Kiểm tra `label_counts` có chứa quá nhiều `unknown` không.
- Kiểm tra `trainable_label_counts` có đủ tối thiểu 2 lớp không.
- Kiểm tra `split_counts.train` không bằng 0.
- Kiểm tra `split_counts.val` không bằng 0.
- Kiểm tra `split_counts.test` không bằng 0.
- Kiểm tra `width_stats.min` có nhỏ hơn `min_width` mong đợi không.
- Kiểm tra `height_stats.min` có nhỏ hơn `min_height` mong đợi không.
- Kiểm tra `annotation_status_counts` có hợp lý với giai đoạn hiện tại không.
- Kiểm tra `review_status_counts` có quá nhiều `pending_review` không.
- Kiểm tra `label_source_counts` có nguồn nào chiếm ưu thế ngoài mong đợi không.
- Kiểm tra cột `decode_ok` trong manifest cho các ảnh lỗi.
- Kiểm tra cột `decode_error` để biết lỗi phát sinh từ decoder hay file hỏng.
- Kiểm tra cột `is_too_small` để biết lý do ảnh bị loại.
- Kiểm tra cột `is_valid` để xác định ảnh có thể đi tiếp hay không.
- Kiểm tra cột `quarantine_reason` để biết ảnh bị loại vì nguyên nhân nào.
- Kiểm tra cột `content_hash` để xác định duplicate theo nội dung thay vì tên file.
- Kiểm tra cột `relative_path` để UI còn render ảnh đúng qua `/dataset/image`.
- Kiểm tra cột `sample_id` có unique không.
- Kiểm tra cột `split` có giá trị ngoài `train`, `val`, `test`, `excluded` hay không.
- Kiểm tra cột `raw_label` nếu đang dùng infer từ thư mục.
- Kiểm tra cột `curated_label` nếu đã import nhãn tay.
- Kiểm tra cột `suggested_label` nếu đã chạy pseudo-label hoặc cluster review.
- Kiểm tra cột `label` là nhãn hiệu lực dùng cho training.
- Kiểm tra cột `label_source` có nằm trong nhóm accepted hay không.
- Kiểm tra cột `annotation_status` có chuyển sang `labeled` hoặc `pseudo_labeled` chưa.
- Kiểm tra cột `review_status` có phản ánh đúng tình trạng human review không.
- Kiểm tra cột `cluster_id` để biết ảnh đã được cluster hóa chưa.
- Kiểm tra cột `cluster_distance` khi nghi outlier.
- Kiểm tra cột `cluster_size` để tránh cluster quá nhỏ.
- Kiểm tra cột `is_cluster_outlier` để biết ảnh có đi vào pseudo-label hay review riêng không.
- Kiểm tra manifest bằng `validate-labels` trước mỗi lần `fit`.
- Kiểm tra `workflow_state()` nếu UI vẫn khóa Step 2 hoặc Step 3.
- Kiểm tra `localagent/src/workflow.rs` nếu bạn cần hiểu logic mở khóa ở server.

## 130. Phụ lục D: map giữa Step 1 và test

- `localagent/tests/test_pipeline.py:46` kiểm tra scan đánh dấu ảnh lỗi, ảnh quá nhỏ và duplicate.
- `localagent/tests/test_pipeline.py:82` kiểm tra `run_all()` ghi manifest và report đúng.
- `localagent/tests/test_pipeline.py:183` kiểm tra suy luận và chuẩn hóa label từ tên file.
- `localagent/tests/test_pipeline.py` còn bao phủ export/import template và validate labels.
- `localagent/tests/test_pipeline.py:272` kiểm tra nhánh embed, cluster và export cluster review.
- `localagent/tests/test_pipeline.py:387` kiểm tra promote cluster labels làm workflow chuyển sang accepted-label mode.
- `localagent/tests/test_pipeline.py:444` kiểm tra stale review row bị bỏ qua.
- `localagent/tests/test_settings.py` bảo vệ default config khỏi thay đổi âm thầm.
- Nếu sửa logic split, nên đọc cả `test_pipeline.py` lẫn `config/settings.py`.
- Nếu sửa schema manifest, phải cập nhật test pipeline trước khi chạm UI.

## 131. Phụ lục E: lỗi phổ biến khi vận hành Step 1

- Nếu UI không hiện ảnh review, kiểm tra `relative_path` trong manifest và route `GET /dataset/image` ở `localagent/src/bin/server.rs:448`.
- Nếu `run-all` xong nhưng Step 2 bị khóa, kiểm tra `workflow_state()` ở `pipeline.py:582`.
- Nếu import labels không có tác dụng, kiểm tra tên cột trong file đầu vào có khớp `sample_id` hoặc `relative_path` không.
- Nếu manifest có đúng nhãn nhưng vẫn không train được, kiểm tra `validate-labels`.
- Nếu nhiều ảnh bị `unknown`, xem lại cấu trúc tên file và regex `LABEL_PATTERN`.
- Nếu duplicate tăng đột biến, so lại nguồn ảnh xem có copy chồng thư mục hay không.
- Nếu report không cập nhật sau khi sửa dataset, chạy lại `report` hoặc `run-all`.
- Nếu `dataset_manifest.csv` và `dataset_manifest.parquet` lệch nhau, ưu tiên đọc lại quy trình ghi của `write_manifest()`.
- Nếu ảnh bị loại quá nhiều, giảm `min_width` hoặc `min_height` nhưng phải cân nhắc chất lượng nhận dạng.
- Nếu đổi `raw_dir`, luôn kiểm tra lại `summary.json.dataset_root`.
- Nếu thay đường dẫn report, đảm bảo server và UI vẫn trỏ đúng artifact location.
- Nếu Step 1 chạy qua server bị fail, đọc log job ở `GET /jobs/{job_id}/logs` tại `localagent/src/bin/server.rs:199`.
