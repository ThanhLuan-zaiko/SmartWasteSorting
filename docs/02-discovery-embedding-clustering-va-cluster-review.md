# Tài liệu 02: Discovery, embedding, clustering và cluster review

## 1. Discovery trong dự án này nghĩa là gì

- Discovery là bước trung gian giữa dataset pipeline và training.
- Discovery dùng biểu diễn ảnh để gom các ảnh giống nhau thành cụm.
- Sau đó con người review theo cụm thay vì review từng ảnh.
- Mục tiêu là giảm chi phí gán nhãn.
- Mục tiêu là tăng tốc chuyển từ weak labels sang accepted labels.

## 2. Vì sao dự án cần discovery

- Dataset ban đầu có thể chưa có nhãn chuẩn.
- Tên file có thể không đáng tin hoàn toàn.
- Review từng ảnh một rất tốn công.
- Gom cụm ảnh gần nhau về mặt trực quan giúp người dùng duyệt nhanh hơn.

## 3. Vị trí code chính của discovery

- `localagent/python/localagent/data/discovery.py`
- `localagent/python/localagent/data/pipeline.py`
- `localagent/src/cluster_review.rs`
- `localagent/src/workflow.rs`
- `interface/components/dashboard/discovery/`

## 4. Discovery có mấy pha

- Pha 1 là trích embedding.
- Pha 2 là clustering.
- Pha 3 là export cluster review.
- Pha 4 là user review cụm.
- Pha 5 là promote quyết định review vào manifest.

## 5. Subcommand tương ứng

- `embed`
- `cluster`
- `export-cluster-review`
- `promote-cluster-labels`

## 6. Điều rất quan trọng về “logic mờ”

- Code hiện tại không có fuzzy logic đúng nghĩa.
- Code hiện tại không có fuzzy c-means.
- Code hiện tại không có membership matrix.
- Code hiện tại không có luật suy diễn mờ.
- Code hiện tại không có defuzzification.
- Code hiện tại dùng CNN embedding cộng với spherical k-means.

## 7. Chỗ chứng minh điều đó trong code

- `localagent/python/localagent/data/discovery.py:112` là hàm `cluster_embeddings`.
- `localagent/python/localagent/data/discovery.py:266` là `_spherical_kmeans`.
- `localagent/python/localagent/data/discovery.py:301` là `_detect_cluster_outliers`.
- Không có file nào trong repo hiện tại tên kiểu `fuzzy.py`, `fcm.py` hoặc `logic_mo`.

## 8. Vì sao cần nói rõ chuyện này trong docs

- Vì yêu cầu bài toán ngoài đời có thể nhắc “logic mờ”.
- Nhưng tài liệu kỹ thuật phải trung thực với implementation hiện tại.
- Nếu docs ghi dự án đã dùng fuzzy logic trong khi code không có, đó là tài liệu sai.

## 9. Discovery bắt đầu ở đâu trong Python

- Từ `DatasetPipeline.embed_dataset` tại `localagent/python/localagent/data/pipeline.py:209`.
- Từ `DatasetPipeline.cluster_dataset` tại `localagent/python/localagent/data/pipeline.py:230`.

## 10. `embed_dataset` làm gì

- Đọc manifest.
- Lấy các record hợp lệ và decode được.
- Gọi `extract_embeddings`.
- Lưu artifact `.npz`.
- Ghi `embedding_summary.json`.
- Regenerate `summary.json`.

## 11. `cluster_dataset` làm gì

- Đọc manifest.
- Đọc embedding artifact.
- Kiểm tra embedding có stale với manifest không.
- Gọi `cluster_embeddings`.
- Gắn `cluster_id`, `cluster_distance`, `cluster_size`, `is_cluster_outlier` vào manifest.
- Ghi `cluster_summary.json`.
- Regenerate `summary.json`.

## 12. Điều kiện để `embed` được phép chạy

- Step 1 phải hoàn tất.
- Manifest và summary phải tồn tại.
- Logic khóa này có ở cả Python và Rust.

## 13. Điều kiện để `cluster` được phép chạy

- Step 1 phải hoàn tất.
- Embedding artifact phải tồn tại.

## 14. Điều kiện để `export-cluster-review` được phép chạy

- Cluster phải sẵn sàng.
- Tức là cluster summary tồn tại hoặc manifest đã có cluster assignment.

## 15. Điều kiện để `promote-cluster-labels` được phép chạy

- Cluster phải sẵn sàng.
- Cluster review phải có ít nhất một quyết định đã lưu.
- Nếu review stale, hệ thống sẽ chặn.

## 16. Vị trí code khóa mở workflow discovery trong Python

- `DatasetPipeline.validate_discovery_command` ở `localagent/python/localagent/data/pipeline.py:608`

## 17. Vị trí code khóa mở workflow discovery trong Rust

- `WorkflowState::from_config` ở `localagent/src/workflow.rs`
- `validate_pipeline_request` ở `localagent/src/jobs/commands.rs`

## 18. `extract_embeddings` làm gì

- Nhận list record hợp lệ.
- Cố gắng dùng extractor pretrained.
- Nếu extractor pretrained lỗi, fallback sang handcrafted feature.
- Chuẩn hóa vector theo chuẩn L2.
- Trả về `EmbeddingArtifact` và summary.

## 19. Vị trí code của `extract_embeddings`

- `localagent/python/localagent/data/discovery.py:72`

## 20. `EmbeddingArtifact` chứa gì

- `sample_ids`
- `relative_paths`
- `vectors`
- `extractor`
- `image_size`
- `fallback_reason`

## 21. Vị trí code của `EmbeddingArtifact`

- `localagent/python/localagent/data/discovery.py:16`

## 22. `EmbeddingArtifact.save` ghi gì

- Ghi `.npz` nén bằng `np.savez_compressed`.
- Ghi `sample_ids`.
- Ghi `relative_paths`.
- Ghi `vectors` kiểu `float32`.
- Ghi tên extractor.
- Ghi image size.
- Ghi fallback reason nếu có.

## 23. File embedding hiện tại trong repo

- `localagent/artifacts/manifests/dataset_embeddings.npz`

## 24. `embedding_summary.json` hiện tại của repo cho biết gì

- `num_samples = 9989`
- `vector_dim = 512`
- `extractor = resnet18_imagenet`
- `image_size = 224`
- `fallback_reason = null`

## 25. Điều đó nghĩa là gì

- Discovery hiện tại thật sự đang dùng feature từ ResNet18 pretrained.
- Không phải handcrafted fallback.
- Mỗi ảnh được biểu diễn bởi vector 512 chiều.

## 26. `_extract_pretrained_embeddings` làm gì

- Import `torch`.
- Import `resnet18` và `ResNet18_Weights.DEFAULT`.
- Dùng `model.fc = torch.nn.Identity()` để lấy feature trước classifier.
- Dùng transform training cùng preset normalization.
- Chạy theo mini-batch.
- Gom vector thành ma trận `N x D`.

## 27. Vị trí code của `_extract_pretrained_embeddings`

- `localagent/python/localagent/data/discovery.py:163`

## 28. Vì sao ResNet18 được dùng làm extractor cho discovery

- ResNet18 đủ phổ biến.
- Feature dimension `512` dễ dùng.
- Nhẹ hơn các backbone lớn hơn.
- Ổn định cho bài toán khám phá sơ bộ.

## 29. `_extract_handcrafted_embeddings` làm gì

- Dùng khi load pretrained extractor thất bại.
- Trích đặc trưng thủ công dựa trên thumbnail grayscale.
- Thêm histogram màu từng kênh.
- Thêm histogram edge.

## 30. Vị trí code của `_extract_handcrafted_embeddings`

- `localagent/python/localagent/data/discovery.py:206`

## 31. `_handcrafted_embedding` làm gì

- Resize ảnh RGB về `32x32`.
- Chuyển sang grayscale.
- Tạo thumbnail `16x16`.
- Flatten thumbnail và chuẩn hóa về `[0,1]`.
- Tính histogram 8 bins cho từng kênh RGB.
- Tính edge bằng Canny.
- Tính histogram edge.
- Concatenate tất cả feature.

## 32. Vị trí code của `_handcrafted_embedding`

- `localagent/python/localagent/data/discovery.py:225`

## 33. `_normalize_vectors` làm gì

- Tính chuẩn L2 của mỗi vector.
- Nếu norm quá nhỏ thì thay bằng `1.0`.
- Chia vector cho norm.

## 34. Vị trí code của `_normalize_vectors`

- `localagent/python/localagent/data/discovery.py:247`

## 35. Vì sao chuẩn hóa L2 quan trọng

- Vì spherical k-means dùng dot product giữa vector đã chuẩn hóa.
- Khi vector đã chuẩn hóa, dot product gần giống cosine similarity.

## 36. `cluster_embeddings` làm gì

- Nhận ma trận vector `N x D`.
- Chuẩn hóa lại vector.
- Suy số cụm nếu người dùng không truyền.
- Chạy spherical k-means.
- Tính kích thước cụm.
- Gắn cờ outlier trong từng cụm.
- Trả `ClusterArtifact` và summary.

## 37. Vị trí code của `cluster_embeddings`

- `localagent/python/localagent/data/discovery.py:112`

## 38. `ClusterArtifact` chứa gì

- `assignments`
- `distances`
- `cluster_sizes`
- `outliers`
- `cluster_count`

## 39. Vị trí code của `ClusterArtifact`

- `localagent/python/localagent/data/discovery.py:55`

## 40. `assignment_for` trong `ClusterArtifact` làm gì

- Lấy cluster id của sample theo index.
- Lấy distance tới centroid.
- Lấy size cụm tương ứng.
- Lấy cờ outlier.

## 41. Số cụm được chọn thế nào khi không truyền `--num-clusters`

- Dùng heuristic căn bậc hai của `num_samples / 3`.
- Sau đó chặn dưới và chặn trên.
- Tối thiểu là `3`.
- Tối đa heuristic là `32`.
- Nhưng luôn bị chặn không vượt quá số sample.

## 42. Vị trí code của heuristic số cụm

- `_resolve_cluster_count` ở `localagent/python/localagent/data/discovery.py:253`

## 43. Ý nghĩa practical của heuristic số cụm

- Nó là default an toàn.
- Nó không bảo đảm tối ưu cho mọi dataset.
- Nếu dataset của bạn có cấu trúc mạnh hoặc lệch lớn, hãy thử `--num-clusters`.

## 44. `spherical k-means` ở đây làm gì

- Khởi tạo centroid ngẫu nhiên bằng cách chọn ngẫu nhiên sample đã chuẩn hóa.
- Tính similarity giữa vector và centroid bằng tích vô hướng.
- Gán mỗi sample vào centroid có similarity lớn nhất.
- Cập nhật centroid bằng trung bình vector trong cụm rồi chuẩn hóa lại.
- Lặp cho tới khi assignment không đổi hoặc hết số iteration.

## 45. Vị trí code của `_spherical_kmeans`

- `localagent/python/localagent/data/discovery.py:266`

## 46. Tại sao gọi là spherical

- Vì các vector được chuẩn hóa lên hypersphere đơn vị.
- Similarity dựa trên hướng nhiều hơn độ lớn.

## 47. Distance trong code được tính thế nào

- Sau khi có centroid cuối, code lấy `best_similarity`.
- Distance được đặt là `1.0 - best_similarity`.
- Similarity càng lớn thì distance càng nhỏ.

## 48. Ý nghĩa của `cluster_distance`

- Gần `0` nghĩa là sample rất gần centroid của cụm.
- Lớn hơn nghĩa là sample kém điển hình hơn trong cụm.

## 49. Phát hiện outlier trong cụm làm thế nào

- Xét từng cụm riêng biệt.
- Lấy phân phối `cluster_distance` trong cụm đó.
- Nếu cụm có ít hơn hoặc bằng `3` phần tử thì không đánh dấu outlier.
- Tính median distance.
- Tính MAD là median của `abs(distance - median_distance)`.
- Nếu MAD đủ lớn thì threshold là `median + 2.5 * MAD`.
- Nếu MAD gần bằng `0`, fallback sang quantile `0.9`.
- Distance lớn hơn threshold bị đánh dấu outlier.

## 50. Vị trí code của `_detect_cluster_outliers`

- `localagent/python/localagent/data/discovery.py:301`

## 51. Ý nghĩa thực tiễn của outlier

- Outlier không nhất thiết là ảnh lỗi.
- Outlier chỉ là ảnh khác biệt tương đối so với tâm cụm.
- Khi promote cluster labels, code bỏ qua sample outlier.
- Điều này giảm rủi ro gán nhãn nhầm do review một cụm quá rộng.

## 52. `cluster_summary.json` hiện tại của repo cho biết gì

- `num_samples = 9989`
- `cluster_count = 32`
- `outlier_count = 790`
- `cluster_size_stats.min = 136`
- `cluster_size_stats.median = 296`
- `cluster_size_stats.max = 551`

## 53. Điều đó cho bạn biết gì

- Dataset hiện tại được chia thành 32 cụm.
- Cụm nhỏ nhất vẫn khá lớn.
- Cụm lớn nhất có 551 sample.
- Discovery ở đây đang vận hành ở quy mô hàng nghìn ảnh, không phải toy example.

## 54. `export_cluster_review` làm gì

- Nhóm manifest theo `cluster_id`.
- Chọn một số sample đại diện theo distance nhỏ nhất.
- Tính majority label hiện tại trong cụm.
- Sinh file CSV để người dùng review.
- Nếu file review cũ đã tồn tại và còn current, giữ lại quyết định cũ.
- Nếu row cũ stale, reset về trạng thái chưa review.

## 55. Vị trí code của `export_cluster_review`

- `localagent/python/localagent/data/pipeline.py:276`

## 56. `cluster_review.csv` có những cột gì

- `cluster_id`
- `cluster_size`
- `outlier_count`
- `representative_sample_ids`
- `representative_paths`
- `current_majority_label`
- `label`
- `status`
- `notes`

## 57. Ý nghĩa của `representative_sample_ids`

- Đây là danh sách sample id tiêu biểu trong cụm.
- Mặc định ghép bằng dấu `|`.
- Dùng để kiểm stale row khi cluster thay đổi.

## 58. Ý nghĩa của `representative_paths`

- Đây là danh sách path tương đối của ảnh đại diện trong cụm.
- UI dùng để hiển thị preview.

## 59. Ý nghĩa của `current_majority_label`

- Đây là nhãn chiếm đa số hiện tại trong cụm dựa trên manifest hiện thời.
- Nó chỉ là gợi ý.
- Nó không tự động được promote nếu người dùng không xác nhận.

## 60. Ý nghĩa của cột `label` trong cluster review CSV

- Đây là nhãn người dùng muốn gán cho cả cụm.
- Nếu `status = labeled`, label này sẽ được normalize và apply vào các sample không phải outlier.

## 61. Ý nghĩa của cột `status` trong cluster review CSV

- `labeled`
- `unlabeled`
- `excluded`

## 62. Ý nghĩa của cột `notes`

- Đây là chỗ người dùng ghi chú thủ công.
- Hệ thống giữ lại notes nếu row chưa stale.

## 63. Cách stale row được nhận biết

- So sánh fingerprint của row hiện tại với row cũ.
- Fingerprint gồm `cluster_id`, `cluster_size`, `outlier_count`, `representative_sample_ids`.
- Nếu khác, row cũ được coi là stale.

## 64. Vị trí code của fingerprint cluster review

- `_cluster_review_fingerprint` ở `localagent/python/localagent/data/pipeline.py:1315`
- `fingerprints_match` ở `localagent/src/cluster_review.rs`

## 65. Vì sao phải chống stale row

- Vì nếu bạn re-cluster hoặc dataset đổi, cluster cũ không còn mang nghĩa cũ.
- Nếu vẫn giữ nhãn review cũ, bạn có thể gán sai hàng loạt sample.

## 66. `promote_cluster_labels` làm gì

- Đọc `cluster_review.csv`.
- Kiểm current row hay stale row.
- Với mỗi sample trong manifest, nếu sample thuộc cụm đã có quyết định và không phải outlier, cập nhật nhãn.
- Nếu cluster bị `excluded`, sample trở về `unknown`.
- Re-assign split sau khi update.
- Ghi manifest và report mới.

## 67. Vị trí code của `promote_cluster_labels`

- `localagent/python/localagent/data/pipeline.py:318`

## 68. Cách `promote_cluster_labels` bỏ qua outlier

- Trong vòng lặp manifest, nếu `is_cluster_outlier` là `True`, sample được giữ nguyên.
- Điều này có chủ đích.

## 69. Tại sao bỏ qua outlier là hợp lý

- Vì outlier có thể không cùng lớp với phần lớn cụm.
- Review cụm là công cụ tăng tốc, không phải phép gán nhãn tuyệt đối cho mọi sample.

## 70. `ClusterReviewStore` trong Rust làm gì

- Tải state review từ manifest CSV và review CSV.
- Trả JSON đầy đủ cho UI.
- Lưu quyết định review khi người dùng sửa trên giao diện.
- Kiểm tra stale cluster id trước khi save.

## 71. Vị trí code của `ClusterReviewStore`

- `localagent/src/cluster_review.rs`

## 72. Vì sao server Rust giữ logic cluster review riêng

- UI cần đọc và ghi cluster review qua HTTP.
- Nếu mọi thứ đi qua Python CLI thì UX sẽ nặng hơn.
- Rust server có thể trả JSON trực tiếp, đồng thời kiểm stale ngay lúc save.

## 73. `GET /cluster-review` trả gì

- `review_file`
- `cluster_count`
- `reviewed_count`
- `stale_reset_count`
- `clusters`

## 74. `PUT /cluster-review` nhận gì

- `review_file` tùy chọn
- `clusters` là mảng các cụm đã sửa
- Mỗi cụm gồm `cluster_id`, `cluster_size`, `outlier_count`, `representative_sample_ids`, `representative_paths`, `label`, `status`, `notes`

## 75. Vì sao UI phải lưu review trước rồi mới promote

- Vì review là bước chỉnh tay của người dùng.
- Promote là bước batch update manifest.
- Tách hai bước giúp người dùng chỉnh đi chỉnh lại trước khi áp dụng hàng loạt.

## 76. `workflow/state` phản ánh discovery thế nào

- Cho biết embedding artifact có tồn tại không.
- Cho biết cluster ready hay không.
- Cho biết bao nhiêu cluster đã review.
- Cho biết có accepted cluster review label nào chưa.

## 77. Thống kê cluster preview trong `summary.json` dùng để làm gì

- Cho phép dashboard hiển thị preview cụm lớn.
- Cho phép người mới hiểu nhanh dataset đang có những nhóm ảnh nào.

## 78. Vị trí code build cluster preview

- `_build_cluster_preview_summary` ở `localagent/python/localagent/data/pipeline.py:1019`

## 79. Cluster preview hiện tại của repo cho thấy gì

- Cụm lớn nhất hiện tại là cluster `20` với `551` sample, majority label `paper`.
- Cụm lớn thứ hai là cluster `18` với `493` sample, majority label `folk`.
- Nhiều cụm lớn còn lại có majority label `glass`.

## 80. Điều đó gợi ý gì về dataset

- Lớp `glass` chiếm ưu thế.
- `paper` và `folk` vẫn có cụm rõ.
- Clustering đang tách ra những nhóm tương đối có nghĩa về mặt nhãn.

## 81. Ví dụ workflow discovery tiêu chuẩn

```powershell
cd localagent
uv run python -m localagent.data.pipeline embed
uv run python -m localagent.data.pipeline cluster
uv run python -m localagent.data.pipeline export-cluster-review
```

## 82. Bước tiếp theo sau khi export cluster review

- Mở `artifacts/manifests/cluster_review.csv`.
- Hoặc dùng UI cluster review nếu server và interface đang chạy.
- Điền `label` và `status` cho cụm.
- Lưu file.
- Chạy promote.

## 83. Command promote

```powershell
cd localagent
uv run python -m localagent.data.pipeline promote-cluster-labels --review-file artifacts/manifests/cluster_review.csv
```

## 84. Nếu muốn thử số cụm khác

```powershell
cd localagent
uv run python -m localagent.data.pipeline cluster --num-clusters 16
```

## 85. Khi nào nên giảm số cụm

- Khi một lớp bị chia quá nhỏ thành quá nhiều cluster.
- Khi review cụm trở nên quá vụn.

## 86. Khi nào nên tăng số cụm

- Khi mỗi cluster quá hỗn tạp.
- Khi majority label trong cluster không đủ rõ.

## 87. Cảnh báo khi đổi số cụm

- Review cũ rất có thể trở thành stale.
- Bạn phải re-export cluster review.

## 88. Discovery có dùng threshold xác suất của model không

- Không.
- Discovery dùng feature extractor và clustering.
- Threshold xác suất thuộc về pseudo-label ở Step 3.

## 89. Discovery có tự gán accepted label không

- Không.
- Discovery chỉ đề xuất cấu trúc cụm.
- Accepted label chỉ xuất hiện sau khi người dùng review hoặc sau pseudo-label gate.

## 90. Discovery có sửa ảnh gốc không

- Không.
- Nó chỉ ghi artifact `.npz`, `.csv`, `.json` và update manifest.

## 91. Discovery có dùng cache ảnh training không

- Không trực tiếp.
- Cache ảnh training là concern của trainer.

## 92. Discovery có phụ thuộc training không

- Không.
- Bạn có thể chạy discovery trước khi train.
- Trên thực tế nên chạy discovery trước để có accepted labels tốt hơn.

## 93. Discovery có phụ thuộc nhãn đã curate chưa

- Không bắt buộc.
- Nhưng nếu đã có một số nhãn curate, majority label và review cụm sẽ dễ hơn.

## 94. Discovery có cho phép all-unknown dataset không

- Có.
- Bạn vẫn có thể embed và cluster dataset không có nhãn.
- Nhưng để vào training, cuối cùng bạn vẫn cần accepted labels đủ tốt.

## 95. Nếu muốn cắm fuzzy c-means thật vào dự án, nên sửa ở đâu

- Sửa `cluster_embeddings` trong `localagent/python/localagent/data/discovery.py`.
- Thêm artifact chứa membership score nếu cần.
- Quyết định cách map membership sang `cluster_id` cứng cho UI hiện tại.
- Mở rộng manifest nếu muốn lưu membership entropy hoặc top-2 membership.
- Mở rộng `cluster_summary.json`.
- Mở rộng `cluster_review.csv` nếu muốn hiển thị độ mờ của cụm.

## 96. Nếu muốn cắm logic mờ kiểu luật suy diễn, nên sửa ở đâu

- Sau bước embedding nhưng trước bước propose label.
- Hoặc trong bước promote/pseudo-label như một bộ lọc quyết định.
- Tuy nhiên đây là tính năng mới.
- Code hiện tại không có scaffold cho fuzzy rules.

## 97. Nếu muốn giữ nguyên UI hiện tại mà đổi thuật toán cụm

- Tốt nhất là vẫn xuất ra các trường sau:
- `cluster_id`
- `cluster_distance`
- `cluster_size`
- `is_cluster_outlier`
- Nếu giữ được bốn trường đó, phần lớn UI và manifest logic sẽ ít phải sửa.

## 98. Các test quan trọng cho discovery

- `test_embed_cluster_and_export_cluster_review`
- `test_export_cluster_review_preserves_matching_saved_rows`
- `test_export_cluster_review_resets_stale_rows`
- `test_promote_cluster_labels_switches_manifest_to_accepted_labels_only_mode`
- `test_promote_cluster_labels_skips_stale_review_rows`
- `test_discovery_cli_requires_completed_step_one`

## 99. Những test đó xác nhận điều gì

- Embedding và clustering phải sinh artifact thật.
- Export cluster review phải giữ row cũ nếu fingerprint vẫn hợp lệ.
- Export cluster review phải reset row stale.
- Promote phải chuyển training mode sang `accepted_labels_only` khi có accepted labels.
- Promote phải bỏ qua stale review row.

## 100. Chốt file này

- Discovery của dự án hiện tại là discovery bằng embedding CNN cộng với spherical k-means.
- Human-in-the-loop là phần quyết định biến cluster thành accepted labels.
- Fuzzy logic chưa có trong code.
- Nếu muốn thêm fuzzy logic, cần mở rộng `data/discovery.py`, manifest và có thể cả UI/server.

## 101. Phụ lục A: flow discovery chi tiết theo từng bước

- Bước đầu tiên của discovery là gọi `embed_dataset()` ở `localagent/python/localagent/data/pipeline.py:209`.
- `embed_dataset()` nạp manifest hiện tại và chỉ lấy các record đủ điều kiện discovery.
- Logic lọc record đủ điều kiện nằm ở `_discovery_ready_records()` tại `pipeline.py:924`.
- Sau khi lọc xong, pipeline gọi `extract_embeddings()` ở `localagent/python/localagent/data/discovery.py:72`.
- `extract_embeddings()` trả về `EmbeddingArtifact`.
- `EmbeddingArtifact` được khai báo ở `discovery.py:16`.
- Artifact này chứa ma trận vector, danh sách sample id, relative path, metadata về extractor.
- Nếu trích xuất CNN thành công, `extractor` trong artifact sẽ là `resnet18_imagenet`.
- Nếu trích xuất CNN thất bại, pipeline có thể rơi về handcrafted embedding.
- Nhánh fallback nằm ở `_extract_handcrafted_embeddings()` tại `discovery.py:206`.
- Hàm `_handcrafted_embedding()` ở `discovery.py:225` sinh vector thủ công từ histogram và thumbnail.
- Trước khi cluster, vector được chuẩn hóa bằng `_normalize_vectors()` tại `discovery.py:247`.
- Sau bước embed, pipeline lưu file `dataset_embeddings.npz`.
- Metadata mô tả file embedding được ghi vào `embedding_summary.json`.
- Bước tiếp theo là gọi `cluster_dataset()` ở `pipeline.py:230`.
- `cluster_dataset()` nạp artifact embeddings rồi gọi `cluster_embeddings()` ở `discovery.py:112`.
- `cluster_embeddings()` quyết định số cluster bằng `_resolve_cluster_count()` tại `discovery.py:253`.
- Nếu không truyền `--num-clusters`, code chọn số cluster theo heuristic nội bộ.
- Sau đó thuật toán chạy `_spherical_kmeans()` tại `discovery.py:266`.
- Kết quả cluster gồm `cluster_ids`, `distances`, `cluster_sizes`.
- Sau khi có cluster, code chạy `_detect_cluster_outliers()` tại `discovery.py:301`.
- Hàm này đánh dấu ảnh nằm xa centroid bất thường trong từng cluster.
- Từ đó pipeline cập nhật manifest bằng `_apply_cluster_updates()` tại `pipeline.py:931`.
- `cluster_id` được gán về manifest.
- `cluster_distance` được gán về manifest.
- `cluster_size` được gán về manifest.
- `is_cluster_outlier` được gán về manifest.
- Sau đó `write_manifest()` ghi manifest mới xuống đĩa.
- Bước review cho con người bắt đầu ở `export_cluster_review()` tại `pipeline.py:276`.
- File cluster review chứa mỗi cluster một dòng tổng hợp.
- Mỗi dòng có fingerprint để chống stale update.
- Logic fingerprint nằm ở `_cluster_review_fingerprint()` tại `pipeline.py:1315`.
- Khi người dùng cập nhật CSV rồi submit, server nhận qua `PUT /cluster-review` ở `localagent/src/bin/server.rs:535`.
- Module xử lý lưu file review là `localagent/src/cluster_review.rs`.
- Khi promote review, pipeline gọi `promote_cluster_labels()` tại `pipeline.py:318`.
- Hàm này chỉ áp row còn current, không áp stale row.
- Check stale row nằm ở `_cluster_review_row_is_current()` tại `pipeline.py:1306`.
- Cuối cùng `workflow_state()` bật Step 3 khi đã có accepted labels từ cluster review.

## 102. Phụ lục B: embedding CNN trong repo này thực sự làm gì

- CNN ở pha discovery không trực tiếp dự đoán nhãn cuối.
- CNN ở đây đóng vai trò bộ trích xuất đặc trưng.
- Cụ thể, `_extract_pretrained_embeddings()` ở `discovery.py:163` nạp `torchvision.models.resnet18`.
- Sau khi nạp model, code thay `model.fc` bằng `torch.nn.Identity()`.
- Việc bỏ classifier head làm output còn lại là vector đặc trưng trước tầng phân loại.
- Mỗi ảnh sau khi qua backbone cho ra vector chiều 512.
- Giá trị này khớp với `embedding_summary.json.vector_dim = 512`.
- Ảnh đầu vào được resize theo cấu hình extractor.
- Các phép normalize ảnh cho CNN dùng chuẩn ImageNet.
- Điều này thống nhất với chuẩn training trong `vision/transforms.py`.
- Vì extractor được pretrained trên ImageNet, vector có tính khái quát tốt hơn feature thủ công.
- Tuy nhiên vector không phải là nhãn.
- Nó chỉ biểu diễn ảnh trong không gian đặc trưng.
- Hai ảnh giống nhau về hình thức thường ở gần nhau trong không gian đó.
- Nhờ vậy, clustering mới có cơ sở gom nhóm.
- Nếu extractor thất bại, fallback handcrafted làm chất lượng cluster có thể giảm.
- `fallback_reason` trong `embedding_summary.json` là nơi bạn nên nhìn đầu tiên.
- Nếu `fallback_reason = null`, pipeline đang dùng đúng CNN embedding.
- Nếu fallback kích hoạt, discovery vẫn chạy nhưng bạn không nên kỳ vọng cluster đẹp như CNN.
- Kích thước ảnh embedding hiện tại thường là `224`.
- Đây là thỏa hiệp hợp lý giữa tốc độ CPU và chất lượng đặc trưng.
- Việc dùng `resnet18` thay vì model lớn hơn giúp pha discovery nhẹ hơn training.
- Vì cluster review là bước có người duyệt, embedding chỉ cần đủ tốt để gom nhóm gần đúng.
- Đây là thiết kế practical, không phải fully automatic labeling.

## 103. Phụ lục C: spherical k-means và cách repo áp dụng

- Thuật toán cluster chính hiện tại là spherical k-means.
- Cốt lõi của spherical k-means là chuẩn hóa vector về độ dài 1.
- Sau khi chuẩn hóa, tích vô hướng giữa hai vector phản ánh cosine similarity.
- K-means chuẩn tối thiểu hóa Euclidean distance.
- Spherical k-means tối đa hóa độ tương đồng cosine giữa vector và centroid.
- Vì embedding CNN thường được so bằng cosine, lựa chọn này khá hợp lý.
- `_normalize_vectors()` ở `discovery.py:247` chính là bước chuẩn hóa L2.
- `_spherical_kmeans()` ở `discovery.py:266` lặp các bước gán cụm và cập nhật centroid.
- Bước gán cụm chọn centroid có cosine similarity lớn nhất.
- Vì vector đã chuẩn hóa, cosine similarity và dot product là tương đương.
- Sau khi gán cluster, centroid mới được tính bằng trung bình các vector trong cluster.
- Centroid tiếp tục được chuẩn hóa về unit norm.
- Quá trình lặp đến khi hội tụ hoặc đạt số vòng lặp tối đa.
- Trong tài liệu toán học, nếu `x_i` và `μ_k` đã chuẩn hóa thì:
- `cluster(i) = argmax_k x_i^T μ_k`.
- Sau đó cập nhật:
- `μ_k = normalize(Σ_{i: cluster(i)=k} x_i)`.
- Khoảng cách lưu trong manifest là khoảng cách theo logic của code sau khi chọn centroid.
- Bạn nên hiểu `cluster_distance` như thước đo “xa tâm cụm” để phục vụ review.
- Không nên hiểu nó như xác suất.
- Không nên hiểu nó như membership mềm.
- Không nên hiểu nó như fuzzy degree.
- Đây là điểm khác rất lớn so với fuzzy c-means.
- Trong fuzzy c-means, mỗi điểm thuộc nhiều cụm với các mức membership khác nhau.
- Trong repo hiện tại, mỗi ảnh chỉ nhận đúng một `cluster_id`.
- Vì vậy cluster review file hiện tại cũng thiết kế theo kiểu one-cluster-one-label.
- Nếu muốn hỗ trợ multi-membership, bạn phải đổi schema review lẫn manifest.

## 104. Phụ lục D: outlier detection bằng MAD

- Sau khi cluster hóa, code không tin mọi điểm trong cluster đều “đẹp”.
- Vì vậy `_detect_cluster_outliers()` ở `discovery.py:301` chạy thêm bộ lọc outlier.
- Bộ lọc này làm việc trong từng cluster.
- Với mỗi cluster, code lấy phân phối khoảng cách tới centroid.
- Code tính median của khoảng cách.
- Code tính MAD là median absolute deviation.
- Ký hiệu:
- `d_i` là khoảng cách của mẫu `i` tới centroid cụm.
- `m = median(d_i)`.
- `MAD = median(|d_i - m|)`.
- Nếu `MAD` đủ lớn, ngưỡng outlier được đặt theo `m + 2.5 * MAD`.
- Nếu `MAD` gần như bằng 0, code dùng fallback quantile `0.9`.
- Điều này ngăn ngưỡng bị co lại quá mức khi cluster rất chặt.
- Mục đích là đưa các điểm “khó tin” sang human review hoặc pseudo-label.
- Đây là quyết định thực dụng.
- Nó làm cluster review sạch hơn.
- Nó cũng tránh trường hợp một vài ảnh kỳ dị làm nhiễu label cả cluster.
- `cluster_summary.json.outlier_count` giúp bạn biết số ảnh bị đánh dấu.
- Trong snapshot hiện tại, `outlier_count = 790`.
- Đây không phải lỗi.
- Đây là số ảnh cần thận trọng hơn.
- Tập này rất phù hợp cho bước pseudo-label có ngưỡng confidence chặt.
- Bạn có thể hiểu MAD như phiên bản robust của độ lệch chuẩn.
- Khác với standard deviation, MAD ít nhạy với một vài điểm quá xa.
- Vì vậy nó phù hợp cho discovery thô trên dữ liệu thực tế dễ bẩn.
- Nếu thay MAD bằng z-score chuẩn, cluster có thể bị kéo lệch bởi outlier mạnh.
- Nếu thay MAD bằng IQR, bạn phải đánh đổi khác về sensitivity.
- Hiện code chọn MAD là phương án cân bằng.

## 105. Phụ lục E: cấu trúc và ý nghĩa của `cluster_review.csv`

- File review cluster mặc định nằm ở `localagent/artifacts/manifests/cluster_review.csv`.
- Đường dẫn này cũng được UI dùng làm default form.
- `pipelineForm.review_file` được khai báo ở `interface/components/use-localagent-controller.ts:75`.
- File review được sinh bởi `export_cluster_review()` tại `pipeline.py:276`.
- Dòng review đại diện cho cluster, không phải cho từng ảnh.
- Mỗi dòng chứa `cluster_id`.
- Mỗi dòng chứa fingerprint để xác định row có còn khớp cluster hiện tại hay không.
- Mỗi dòng chứa preview summary để người review xem cluster trông giống cái gì.
- Mỗi dòng chứa kích thước cluster.
- Mỗi dòng có cờ outlier summary.
- Mỗi dòng có trường nhãn đề xuất hoặc nhãn người review chấp nhận.
- Khi UI tải review, route `GET /cluster-review` ở `server.rs:523` trả dữ liệu JSON.
- `saveClusterReview()` ở `interface/components/localagent/controller-actions.ts:273` gửi nội dung đã sửa trở lại server.
- Server từ chối lưu nếu đang có dataset pipeline job active.
- Lý do là tránh stale write khi cluster vừa được tính lại.
- Nếu cluster review stale, `promote_cluster_labels()` sẽ bỏ qua.
- Điều đó được test bởi `localagent/tests/test_pipeline.py:444`.
- Nếu bạn thấy review đã nhập mà không promote được, kiểm tra fingerprint trước.
- Nếu số cluster thay đổi do đổi `--num-clusters`, file review cũ gần như chắc chắn stale.
- Không nên chỉnh `cluster_id` bằng tay trong CSV.
- Không nên xóa fingerprint.
- Không nên dùng review file cũ sau khi đã chạy lại `embed` hoặc `cluster`.
- Đúng workflow là export review mới rồi review lại.
- Các hàm liên quan trực tiếp:
- `_build_cluster_review_rows()` ở `pipeline.py:950`.
- `_cluster_groups()` ở `pipeline.py:1007`.
- `_build_cluster_preview_summary()` ở `pipeline.py:1019`.
- `_load_cluster_review_assignments()` ở `pipeline.py:1239`.
- `_load_existing_cluster_review_rows()` ở `pipeline.py:1283`.
- `_reviewed_cluster_count()` ở `pipeline.py:1293`.
- `_cluster_review_row_is_current()` ở `pipeline.py:1306`.
- `_cluster_review_fingerprint()` ở `pipeline.py:1315`.
- `_normalize_annotation_status()` ở `pipeline.py:1323`.

## 106. Phụ lục F: hiện trạng “logic mờ” và nếu muốn thêm thì phải sửa đâu

- Câu trả lời ngắn gọn là hiện tại repo chưa có logic mờ.
- Không có fuzzy rules engine trong Python.
- Không có fuzzy membership matrix trong manifest.
- Không có fuzzy c-means trong `data/discovery.py`.
- Không có fuzzy score trong `cluster_review.csv`.
- Không có UI nào cho phép một cluster mang nhiều nhãn với membership khác nhau.
- Không có API field nào trả về membership degree.
- Điều này nghĩa là khi tài liệu nói về logic mờ, nó phải phân biệt rõ giữa hiện trạng và hướng mở rộng.
- Nếu bạn muốn thêm fuzzy c-means, vị trí sửa đầu tiên là `cluster_embeddings()` ở `discovery.py:112`.
- Bạn cần trả về không chỉ `cluster_id` cứng mà còn membership matrix `u_{ik}`.
- Sau đó manifest phải có cột mới như `cluster_membership_top1`, `cluster_membership_top2`.
- Có thể thêm `cluster_entropy` để đo độ mơ hồ của ảnh.
- `export_cluster_review()` cũng cần sửa để hiển thị mức chắc chắn của cluster.
- UI ở `interface/components/dashboard/discovery/` cũng phải hỗ trợ xem cluster mơ hồ.
- Rust API phải serialize thêm field review mới trong `cluster_review.rs`.
- `workflow_state()` có thể cần logic mới để xác định khi nào cluster “đủ chắc” để promote tự động.
- Nếu thêm fuzzy rules cho labeling, nơi hợp lý là sau cluster và trước promote.
- Ví dụ một rule có thể là:
- Nếu cluster có top label chiếm ưu thế và entropy thấp thì auto-suggest label.
- Nếu entropy cao thì bắt buộc human review.
- Nhưng hiện tại đó chỉ là thiết kế đề xuất.
- Không có code thực thi sẵn trong repo.
- Vì vậy khi vận hành repo bây giờ, hãy nghĩ theo sơ đồ:
- `CNN embedding -> spherical k-means -> MAD outlier -> cluster review -> accepted labels`.
- Đừng nghĩ theo sơ đồ:
- `CNN -> fuzzy inference -> fuzzy membership -> auto labeling`.

## 107. Phụ lục G: command thực chiến và chỉ dấu thành công

- Sau `run-all`, dấu hiệu thành công là có `summary.json`, `dataset_manifest.csv`, `dataset_manifest.parquet`.
- Sau `embed`, dấu hiệu thành công là có `dataset_embeddings.npz` và `embedding_summary.json`.
- Sau `cluster`, dấu hiệu thành công là manifest có `cluster_id` và `cluster_summary.json`.
- Sau `export-cluster-review`, dấu hiệu thành công là có `cluster_review.csv`.
- Sau review thủ công, dấu hiệu đúng là file review vẫn current và không stale.
- Sau `promote-cluster-labels`, dấu hiệu thành công là `label_source_counts.cluster_review` tăng.
- `summary.json.effective_training_mode` nên chuyển sang `accepted_labels_only`.
- Điều này đã được test ở `localagent/tests/test_pipeline.py:387`.
- Nếu `clustered_files` nhỏ hơn `valid_files`, kiểm tra pipeline có bỏ record nào khỏi discovery hay không.
- Nếu `embedding_artifact_exists` là `false`, đừng chạy tiếp `cluster`.
- Nếu `cluster_summary_exists` là `false`, UI discovery sẽ thiếu dữ liệu.
- Nếu `cluster_preview_total` bằng 0, cluster review gần như không usable.
- Nếu `pending_review` còn nhiều, Step 3 có thể vẫn khóa.
- Nếu `cluster_outlier_files` quá lớn, xem lại số cluster và chất lượng dữ liệu nguồn.

## 108. Phụ lục H: map giữa discovery và test

- `localagent/tests/test_pipeline.py:272` là test tốt nhất để đọc flow embed -> cluster -> export review.
- Test này giúp bạn biết file nào phải được sinh ra.
- `localagent/tests/test_pipeline.py:387` xác nhận promote cluster labels chuyển workflow sang chế độ accepted labels.
- `localagent/tests/test_pipeline.py:444` xác nhận stale row không được promote.
- `localagent/tests/test_pipeline.py` còn có test chặn discovery khi Step 1 chưa xong.
- Nếu sửa thuật toán cluster, bạn nên bổ sung test deterministic hoặc ít nhất test shape của artifact.
- Nếu sửa schema review, bạn phải sửa cả test pipeline và UI type trong `interface/lib/localagent.ts`.

## 109. Phụ lục I: câu hỏi tự kiểm tra khi đọc discovery code

- Bạn đã xác định đúng extractor đang dùng chưa.
- Bạn đã mở `embedding_summary.json` để kiểm tra `extractor` chưa.
- Bạn đã kiểm tra `fallback_reason` có phải `null` không.
- Bạn đã phân biệt rõ embedding với nhãn cuối chưa.
- Bạn đã phân biệt rõ cluster cứng với membership mờ chưa.
- Bạn đã biết `cluster_id` hiện tại là gán cứng một-nhãn-một-cụm chưa.
- Bạn đã biết outlier detection chạy sau clustering chưa.
- Bạn đã biết manifest lưu khoảng cách cụm ở cột nào chưa.
- Bạn đã biết file review là theo cụm chứ không theo ảnh chưa.
- Bạn đã biết stale fingerprint được dùng để chặn review cũ chưa.
- Bạn đã biết route nào cấp dữ liệu review cho UI chưa.
- Bạn đã biết route nào nhận save review từ UI chưa.
- Bạn đã biết `workflow_state()` dùng reviewed cluster count để mở Step 3 chưa.
- Bạn đã biết test nào bảo vệ stale-row behavior chưa.
- Bạn đã biết test nào bảo vệ whole flow embed -> cluster -> review chưa.
- Bạn đã biết hiện tại chưa có fuzzy c-means trong repo chưa.
- Bạn đã biết nếu thêm fuzzy logic thì phải sửa cả Python, Rust và UI chưa.
- Bạn đã biết tại sao spherical k-means phù hợp hơn k-means thường cho embedding đã normalize chưa.
- Bạn đã biết tại sao MAD được chọn thay vì standard deviation chưa.
- Bạn đã biết tại sao cluster review vẫn cần con người dù embedding đã khá tốt chưa.
- Bạn đã biết khi nào nên tăng `num_clusters` chưa.
- Bạn đã biết khi nào không nên tăng `num_clusters` chưa.
- Bạn đã biết outlier nhiều có thể là dấu hiệu dữ liệu bẩn chứ không chỉ là lỗi thuật toán chưa.
- Bạn đã biết phải export review mới sau mỗi lần cluster lại chưa.
- Bạn đã biết file nào cần đọc trước nếu muốn thay thuật toán cluster chưa.
- Tệp đầu tiên là `localagent/python/localagent/data/discovery.py`.
- Tệp thứ hai là `localagent/python/localagent/data/pipeline.py`.
- Tệp thứ ba là `localagent/src/cluster_review.rs`.
- Tệp thứ tư là `interface/lib/localagent.ts`.
- Tệp thứ năm là `interface/components/localagent/controller-actions.ts`.
- Tệp thứ sáu là `localagent/tests/test_pipeline.py`.

## 110. Phụ lục J: lộ trình đọc code discovery trong 30 phút đầu

- Phút 1 đến phút 5 đọc `extract_embeddings()` ở `discovery.py:72`.
- Phút 6 đến phút 10 đọc `_extract_pretrained_embeddings()` ở `discovery.py:163`.
- Phút 11 đến phút 13 đọc `_normalize_vectors()` ở `discovery.py:247`.
- Phút 14 đến phút 18 đọc `_spherical_kmeans()` ở `discovery.py:266`.
- Phút 19 đến phút 22 đọc `_detect_cluster_outliers()` ở `discovery.py:301`.
- Phút 23 đến phút 25 đọc `cluster_dataset()` ở `pipeline.py:230`.
- Phút 26 đến phút 27 đọc `export_cluster_review()` ở `pipeline.py:276`.
- Phút 28 đến phút 29 đọc `promote_cluster_labels()` ở `pipeline.py:318`.
- Phút 30 đọc test `test_embed_cluster_and_export_cluster_review()` ở `localagent/tests/test_pipeline.py:272`.
- Nếu chỉ có đúng 10 phút, hãy đọc `cluster_embeddings()` và `promote_cluster_labels()` trước.
- Nếu chỉ có đúng 5 phút, hãy đọc `cluster_summary.json`, `embedding_summary.json` và `cluster_review.csv` trước.

## 111. Phụ lục K: dấu hiệu discovery đang vận hành đúng

- `embedding_summary.json.extractor` là `resnet18_imagenet`.
- `embedding_summary.json.vector_dim` là `512`.
- `embedding_summary.json.fallback_reason` là `null`.
- `cluster_summary.json.cluster_count` lớn hơn `1`.
- `cluster_summary.json.outlier_count` không âm.
- Manifest có `cluster_id` khác rỗng cho phần lớn ảnh hợp lệ.
- `summary.json.clustered_files` gần với `valid_files`.
- `summary.json.cluster_preview_total` lớn hơn `0`.
- `cluster_review.csv` có nhiều dòng ứng với nhiều cụm.
- `workflow_state()` báo Step 2 completed khi embed và cluster đã xong.
- UI tải được cluster review qua `GET /cluster-review`.
- `test_embed_cluster_and_export_cluster_review` vẫn pass sau khi bạn sửa code.
