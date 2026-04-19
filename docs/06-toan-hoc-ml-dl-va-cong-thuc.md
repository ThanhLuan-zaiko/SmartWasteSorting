# Tài liệu 06: Toán học, machine learning, deep learning và công thức đang dùng

## 1. File này dùng để làm gì

- File này giải thích nền tảng toán học của pipeline.
- File này nói rõ công thức nào đang thực sự xuất hiện trong code.
- File này cũng nói công thức nào là kiến thức nền để hiểu nhưng không nhất thiết được code trực tiếp dưới dạng biểu thức.

## 2. Machine learning là gì trong bối cảnh dự án này

- Machine learning là cách để mô hình học quy luật từ dữ liệu thay vì viết tay mọi luật phân loại.
- Trong dự án này, phần machine learning rõ nhất là classifier học từ ảnh đã gán nhãn.
- Ngoài classifier, clustering cũng là một bài toán machine learning không giám sát.

## 3. Deep learning là gì trong bối cảnh dự án này

- Deep learning là nhánh của machine learning dùng mạng nơ-ron nhiều tầng.
- Trong dự án này, CNN backbone là thành phần deep learning chính.
- Các backbone như MobileNet, ResNet, EfficientNet đều là deep neural network.

## 4. Supervised learning là gì ở repo này

- Mô hình nhận ảnh đầu vào.
- Mô hình nhận nhãn đích từ manifest.
- Loss được tính giữa dự đoán và nhãn thật.
- Model cập nhật tham số để giảm loss đó.

## 5. Semi-supervised learning là gì ở repo này

- Dùng một tập nhãn chấp nhận ban đầu để train baseline.
- Dùng model baseline để gợi ý nhãn cho dữ liệu chưa chấp nhận.
- Chỉ chấp nhận pseudo-label khi đủ confident.
- Dùng cluster review để tăng tốc annotation ở mức cụm.

## 6. Weak supervision là gì ở repo này

- Nhãn suy từ filename là weak label.
- Nó hữu ích để khởi động pipeline.
- Nhưng sau khi có accepted labels, filename label không còn là ground truth nữa.

## 7. CNN là gì

- CNN là viết tắt của Convolutional Neural Network.
- Đây là loại mạng rất phù hợp cho ảnh.
- Ý tưởng cơ bản là dùng bộ lọc tích chập để phát hiện pattern cục bộ rồi dần dần gom chúng thành pattern mức cao hơn.

## 8. Vì sao CNN hợp với ảnh rác

- Ảnh rác có texture, shape, edge và pattern màu.
- CNN học được các đặc trưng như viền, vùng sáng tối, hình dáng cục bộ.

## 9. Transfer learning là gì

- Là lấy backbone đã học trước trên tập lớn như ImageNet.
- Sau đó fine-tune hoặc chỉ thay head cho bài toán mới.
- Repo hiện tại dùng transfer learning.

## 10. Chứng cứ code dùng transfer learning

- `build_model_stub` trong `localagent/python/localagent/training/trainer.py` cố load `Weights.DEFAULT`.

## 11. Freeze backbone về mặt toán học có nghĩa gì

- Gradient chỉ cập nhật head cuối.
- Tham số backbone được coi là hằng trong bước tối ưu.

## 12. Hàm mục tiêu của supervised classifier

- Mục tiêu là tìm tham số `theta` sao cho loss trung bình trên tập train nhỏ nhất.
- Viết ngắn gọn:
- `theta* = argmin_theta (1/N) * sum_i L(f_theta(x_i), y_i)`

## 13. Trong repo này `f_theta(x)` là gì

- Là CNN backbone cộng classifier head.
- Output của `f_theta` là vector logits cho các lớp.

## 14. Logit là gì

- Là điểm số chưa chuẩn hóa cho từng lớp.
- Logit không phải xác suất.
- Phải qua softmax mới thành xác suất.

## 15. Công thức softmax

- Với logit `z_k` của lớp `k`:
- `p_k = exp(z_k) / sum_j exp(z_j)`

## 16. Vị trí code softmax trong repo

- Trong pseudo-label Python, code dùng `torch.softmax`.
- Trong Rust inference, có hàm `softmax` riêng ở `localagent/src/inference.rs`.

## 17. Cross entropy loss là gì

- Nếu lớp thật là `y`, cross entropy cho một sample là:
- `L = -log(p_y)`
- Nếu dùng one-hot label:
- `L = -sum_k y_k log(p_k)`

## 18. Vì sao cross entropy hợp với classifier nhiều lớp

- Nó phạt mạnh khi mô hình đặt xác suất thấp cho lớp đúng.
- Nó kết hợp tự nhiên với softmax.

## 19. Trong repo này cross entropy được dùng ở đâu

- `criterion = nn.CrossEntropyLoss(...)` trong `WasteTrainer.fit`.
- `criterion = nn.CrossEntropyLoss(...)` trong `WasteTrainer.evaluate`.

## 20. Gradient descent là gì

- Là ý tưởng cập nhật tham số theo hướng ngược gradient của loss.
- Mục tiêu là giảm loss.

## 21. Adam optimizer là gì

- Là optimizer kết hợp momentum và adaptive learning rate.
- Repo hiện tại dùng Adam.

## 22. Công thức Adam ở mức khái niệm

- `m_t = beta1 * m_{t-1} + (1 - beta1) * g_t`
- `v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2`
- `theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)`

## 23. Vì sao Adam hợp với repo hiện tại

- Dễ dùng.
- Ổn định cho fine-tuning nhỏ và vừa.
- Không cần tuning quá phức tạp cho baseline CPU.

## 24. Class imbalance là gì

- Một số lớp có quá nhiều sample.
- Một số lớp có quá ít sample.
- Repo hiện tại bị lệch mạnh về `glass`.

## 25. Tác hại của class imbalance

- Model có xu hướng thiên về lớp đông.
- Accuracy tổng thể có thể nhìn cao nhưng lớp nhỏ bị học rất kém.

## 26. Class weighting trong repo có công thức gì

- Nếu lớp `c` có `n_c > 0`.
- Tổng sample của các lớp hiện diện là `N`.
- Số lớp hiện diện là `K`.
- Weight:
- `w_c = N / (K * n_c)`

## 27. Ý nghĩa của công thức đó

- Lớp càng ít thì `n_c` càng nhỏ.
- Lớp càng ít thì `w_c` càng lớn.

## 28. Weighted sampler là gì

- Là cách lấy sample train với xác suất thiên về lớp hiếm.
- Repo dùng `WeightedRandomSampler` khi `class_bias_strategy` là `sampler` hoặc `both`.

## 29. Accuracy là gì

- `accuracy = so_mau_du_doan_dung / tong_so_mau`

## 30. Precision là gì

- Với một lớp:
- `precision = TP / (TP + FP)`

## 31. Recall là gì

- Với một lớp:
- `recall = TP / (TP + FN)`

## 32. F1 là gì

- `F1 = 2 * precision * recall / (precision + recall)`

## 33. Macro F1 là gì

- Tính F1 cho từng lớp.
- Lấy trung bình đều giữa các lớp.
- Mỗi lớp có trọng số như nhau.

## 34. Weighted F1 là gì

- Tính F1 cho từng lớp.
- Lấy trung bình có trọng số support của từng lớp.

## 35. Vì sao repo báo cả macro F1 và weighted F1

- Macro F1 phản ánh công bằng giữa các lớp.
- Weighted F1 phản ánh hiệu năng chung có tính đến số lượng mẫu.
- Khi data lệch nặng, hai số này có thể chênh đáng kể.

## 36. Confusion matrix là gì

- Ma trận mà hàng là lớp thật.
- Cột là lớp dự đoán.
- Phần tử `(i,j)` là số mẫu lớp thật `i` bị dự đoán thành `j`.

## 37. Repo tính classification report ở đâu

- `_build_classification_report` ở `localagent/python/localagent/training/trainer.py:678`
- Hoặc qua Rust bridge nếu bridge sẵn sàng.

## 38. Precision, recall, F1 được tính trong code thế nào

- `tp` là phần tử chéo của confusion matrix.
- `predicted_count` là tổng cột.
- `support` là tổng hàng.
- `fp = predicted_count - tp`
- `fn = support - tp`

## 39. Embedding là gì

- Embedding là vector biểu diễn ảnh trong không gian số.
- Hai ảnh giống nhau về nội dung trực quan thường có embedding gần nhau hơn.

## 40. Discovery đang dùng embedding nào

- Feature 512 chiều từ ResNet18 pretrained.

## 41. Chuẩn hóa L2 của vector là gì

- Với vector `v`, norm là:
- `||v||_2 = sqrt(sum_i v_i^2)`
- Vector chuẩn hóa là:
- `v_hat = v / ||v||_2`

## 42. Vì sao phải chuẩn hóa vector trước clustering

- Để so sánh theo hướng thay vì độ lớn.
- Điều này phù hợp với cosine-like similarity.

## 43. Spherical k-means là gì

- Là biến thể của k-means trên vector đã chuẩn hóa.
- Nó tối đa hóa similarity góc thay vì tối thiểu hóa khoảng cách Euclid đơn thuần.

## 44. Assignment trong spherical k-means

- Với centroid `c_j`, sample `x_i` được gán vào cụm:
- `argmax_j x_i^T c_j`

## 45. Update centroid trong spherical k-means

- Với tập thành viên cụm `C_j`:
- `mu_j = (1 / |C_j|) * sum_{x_i in C_j} x_i`
- Sau đó chuẩn hóa:
- `c_j = mu_j / ||mu_j||_2`

## 46. Repo dùng số iteration tối đa bao nhiêu

- `max_iterations = 20`

## 47. Distance trong clustering được định nghĩa ra sao

- Nếu similarity tốt nhất là `s`.
- Distance là:
- `d = 1 - s`

## 48. Median là gì

- Là giá trị ở giữa sau khi sắp xếp.
- Dùng tốt khi dữ liệu có outlier.

## 49. MAD là gì

- MAD là Median Absolute Deviation.
- Nếu median distance là `m`.
- `MAD = median(|d_i - m|)`

## 50. Threshold outlier trong repo

- Nếu `MAD > 1e-8`:
- `threshold = median_distance + 2.5 * MAD`
- Nếu `MAD` gần 0:
- `threshold = quantile_0.9`

## 51. Vì sao MAD hợp lý cho outlier

- Ít nhạy với outlier hơn variance hoặc standard deviation.
- Hợp cho cụm có phân phối không chuẩn.

## 52. Pseudo-label về mặt toán học là gì

- Model tạo phân phối xác suất `p(y|x)` cho sample chưa nhãn.
- Nếu phân phối đủ chắc, lấy `argmax_y p(y|x)` làm nhãn tạm.
- Repo chỉ accept pseudo-label nếu confidence và margin vượt ngưỡng.

## 53. Confidence trong repo là gì

- `confidence = max_k p_k`

## 54. Margin trong repo là gì

- `margin = p_top1 - p_top2`

## 55. Vì sao confidence cao nhưng margin thấp vẫn nguy hiểm

- Vì model có thể hơi thiên về một lớp nhưng vẫn phân vân mạnh với lớp thứ hai.
- Accept các sample đó dễ lan truyền nhãn sai.

## 56. Normalization ảnh kiểu ImageNet là gì

- Ảnh RGB sau khi chia `255` được chuẩn hóa:
- `x'_c = (x_c - mean_c) / std_c`
- Với mean:
- `(0.485, 0.456, 0.406)`
- Với std:
- `(0.229, 0.224, 0.225)`

## 57. Vị trí code normalization stats

- `localagent/python/localagent/vision/transforms.py`

## 58. Vì sao normalization quan trọng

- Backbone pretrained mong chờ phân phối đầu vào gần giống lúc pretrain.
- Nếu normalization sai, inference và training đều lệch.

## 59. Resize ảnh trong repo là gì

- Repo hiện dùng resize về hình vuông `image_size x image_size`.
- Không giữ nguyên aspect ratio trong transform training mặc định.

## 60. Ý nghĩa practical của resize vuông

- Đơn giản.
- Dễ export ONNX.
- Phù hợp với backbone tiêu chuẩn.
- Đổi lại có thể méo tỷ lệ vật thể.

## 61. Overfitting là gì

- Model học quá sát train data.
- Validation hoặc test không cải thiện tương xứng.

## 62. Early stopping giúp gì về mặt học máy

- Nó là một dạng regularization thực dụng.
- Nó chặn việc train tiếp khi validation không còn tốt lên.

## 63. Checkpoint best là gì về mặt tối ưu

- Là tham số có metric theo dõi tốt nhất trong quá trình train.
- Repo dùng loss nhỏ nhất làm tiêu chí chính.

## 64. Benchmark trong repo đo gì

- Duration từng stage.
- Accuracy.
- Macro F1.
- Weighted F1.
- Best loss.
- Best epoch.
- Tổng thời gian.

## 65. ONNX là gì

- ONNX là định dạng trao đổi model.
- Nó giúp model rời khỏi runtime PyTorch để chạy ở engine khác.

## 66. Vì sao repo export ONNX

- Để inference bằng Rust ONNX Runtime.
- Để có deployment artifact tương đối độc lập với PyTorch train loop.

## 67. Verification ONNX trong repo là gì

- Chạy cùng dummy input qua PyTorch và ONNX.
- So sánh logits.
- Nếu `max_abs_diff` quá lớn, export bị xem là không đạt.

## 68. Công thức `max_abs_diff`

- `max_abs_diff = max(|onnx_logits_i - pytorch_logits_i|)`

## 69. Calibration là gì và repo có làm không

- Calibration là làm xác suất đầu ra phản ánh xác suất thật tốt hơn.
- Repo hiện tại chưa có bước calibration riêng như temperature scaling.

## 70. Data split là gì

- Tập train dùng để học.
- Tập val dùng để chọn checkpoint và early stopping.
- Tập test dùng để báo hiệu năng cuối.

## 71. Vì sao split phải theo lớp

- Để mỗi split còn giữ đại diện của các lớp.
- Nếu không, validation/test có thể thiếu một lớp.

## 72. Semi-supervised learning của repo gồm những mảnh nào

- Weak label từ filename.
- Cluster review theo embedding.
- Accepted labels do người dùng xác nhận.
- Pseudo-label do model accept theo cổng confidence và margin.

## 73. Human-in-the-loop là gì

- Con người vẫn tham gia xác nhận nhãn.
- Hệ thống chỉ giúp gom cụm, gợi ý và tăng tốc.

## 74. Fuzzy logic khác gì so với clustering đang dùng

- Fuzzy logic hoặc fuzzy c-means cho phép một sample thuộc nhiều cụm với mức độ membership khác nhau.
- Repo hiện tại gán cứng mỗi sample vào đúng một cụm.

## 75. Nếu dùng fuzzy c-means thì sẽ có gì thêm về toán

- Membership `u_ij` cho sample `i` với cluster `j`.
- Ràng buộc `sum_j u_ij = 1`.
- Objective thường là:
- `J_m = sum_i sum_j (u_ij^m) ||x_i - c_j||^2`
- Repo hiện tại không triển khai objective này.

## 76. Tại sao spherical k-means đủ thực dụng cho repo hiện tại

- Dễ cài đặt.
- Nhanh.
- Phù hợp với embedding đã chuẩn hóa.
- Dễ map sang UI review theo cụm cứng.

## 77. ROC-AUC có trong repo không

- Không thấy được tính trong code hiện tại.
- Repo tập trung vào accuracy, macro F1, weighted F1 và confusion matrix.

## 78. Loss curve có ở đâu

- Trong `history` của training report.
- `ArtifactStore.training_overview` cũng trích chuỗi epoch để vẽ chart.

## 79. `history` lưu những gì

- `epoch`
- `train_loss`
- `train_accuracy`
- `val_loss`
- `val_accuracy`

## 80. Vì sao chart loss và accuracy hữu ích

- Chúng cho thấy model đang học thật hay không.
- Chúng giúp phát hiện overfitting sớm.

## 81. Kết quả evaluation hiện tại cho ta bài học gì

- Accuracy cao khoảng `92.96%`.
- Nhưng macro F1 chỉ khoảng `0.8074`.
- Điều này cho thấy lớp nhỏ vẫn khó hơn và accuracy tổng chưa nói hết câu chuyện.

## 82. Tại sao macro F1 thấp hơn nhiều weighted F1

- Vì lớp lớn `glass` chiếm đa số và làm weighted F1 đẹp hơn.
- Hai lớp nhỏ `folk` và `paper` có precision thấp hơn.

## 83. Precision của lớp `paper` hiện tại cho thấy gì

- Theo evaluation hiện tại, precision `paper` chỉ khoảng `0.538`.
- Nghĩa là nhiều mẫu bị dự đoán là `paper` thực ra không phải paper.

## 84. Recall của lớp `paper` hiện tại cho thấy gì

- Recall `paper` khoảng `0.907`.
- Nghĩa là model tìm được phần lớn paper thật, nhưng tradeoff bằng khá nhiều false positive.

## 85. Đây là bài toán tradeoff kiểu gì

- Tradeoff giữa sensitivity và specificity ở lớp nhỏ.
- Dữ liệu lệch lớp làm vấn đề rõ hơn.

## 86. Tại sao mô hình vẫn có thể hữu ích dù precision lớp nhỏ chưa cao

- Vì pipeline không dừng ở accuracy.
- Bạn vẫn có thể dùng threshold, pseudo-label gate hoặc review con người cho tình huống nhạy cảm.

## 87. Entropy của phân phối dự đoán có dùng trong repo không

- Không thấy dùng trực tiếp.
- Repo dùng top1 confidence và margin.

## 88. Tại sao margin đơn giản hơn entropy

- Dễ hiểu.
- Dễ tính.
- Dễ giải thích cho người vận hành pipeline.

## 89. Ý nghĩa của seed trong toán học thực nghiệm

- Seed giữ ổn định quá trình random shuffle và khởi tạo một số bước.
- Nó giúp run có thể tái hiện gần hơn.

## 90. Giới hạn của reproducibility ở repo

- Dù có seed, một số thành phần phụ thuộc thư viện, device và môi trường.
- Do đó reproducibility không phải tuyệt đối ở mọi cấu hình.

## 91. Kiến thức cần nhớ nhất

- Repo hiện tại là một pipeline deep learning thực dụng.
- Nó kết hợp supervised learning, semi-supervised labeling và clustering không giám sát.
- Công thức đang dùng khá chuẩn, không dị biệt.
- Phần “logic mờ” hiện tại là khoảng trống của implementation, không phải thành phần đã có.

## 92. Phụ lục A: bảng ký hiệu sẽ dùng trong phần còn lại

- `x` là một ảnh đầu vào sau khi đã được biến đổi.
- `x_i` là mẫu thứ `i`.
- `y` là nhãn thật của ảnh.
- `y_i` là nhãn thật của mẫu thứ `i`.
- `ŷ` là nhãn dự đoán.
- `p(y=c|x)` là xác suất mô hình gán ảnh `x` vào lớp `c`.
- `z` là vector logits trước softmax.
- `z_c` là logit của lớp `c`.
- `K` là số lớp.
- `N` là số mẫu.
- `f_θ(x)` là hàm mạng nơ-ron với tham số `θ`.
- `h(x)` là embedding đặc trưng của ảnh.
- `μ_k` là centroid của cluster `k`.
- `d_i` là khoảng cách từ mẫu `i` tới centroid của cluster được gán.
- `n_c` là số mẫu thuộc lớp `c`.
- `w_c` là trọng số loss của lớp `c`.
- `η` là learning rate.
- `L` là loss tổng.
- `L_i` là loss của một mẫu.
- `u_{ik}` thường dùng cho membership mờ của fuzzy c-means.
- Repo hiện tại không lưu `u_{ik}`.
- `m` trong ngữ cảnh MAD là median khoảng cách.
- `MAD` là median absolute deviation.
- `TP` là true positive.
- `FP` là false positive.
- `FN` là false negative.
- `precision = TP / (TP + FP)`.
- `recall = TP / (TP + FN)`.
- `F1 = 2 * precision * recall / (precision + recall)`.
- `support` là số mẫu thật thuộc lớp đang xét.
- `||v||_2` là chuẩn L2 của vector `v`.
- `normalize(v) = v / ||v||_2` khi `||v||_2 > 0`.

## 93. Phụ lục B: softmax và cross-entropy trong classifier

- Classifier CNN của repo cho ra logits.
- Mỗi logit là một giá trị thực chưa chuẩn hóa.
- Để biến logits thành xác suất, dùng softmax.
- Công thức softmax:
- `p_c = exp(z_c) / Σ_j exp(z_j)`.
- Tổng các `p_c` bằng 1.
- Lớp có xác suất cao nhất là dự đoán top-1.
- Trong code, PyTorch xử lý phần này ẩn bên trong `CrossEntropyLoss`.
- `CrossEntropyLoss` được tạo ở `localagent/python/localagent/training/trainer.py:1294`.
- `CrossEntropyLoss` cũng xuất hiện ở nhánh evaluate tại `trainer.py:908`.
- Nếu nhãn thật của ảnh là lớp `t`, cross-entropy cho mẫu đó là:
- `L_i = -log(p_t)`.
- Nếu mô hình gán xác suất 0.9 cho lớp đúng, loss khoảng `0.1053`.
- Nếu mô hình gán xác suất 0.5 cho lớp đúng, loss khoảng `0.6931`.
- Nếu mô hình gán xác suất 0.1 cho lớp đúng, loss khoảng `2.3026`.
- Vì vậy cross-entropy phạt rất mạnh những dự đoán tự tin nhưng sai.
- Đây là lý do nó phù hợp cho classification.
- Trong dataset lệch lớp, loss này có thể bị lớp đông lấn át.
- Vì vậy repo thêm class weights.
- Nếu có class weights, loss tổng quát cho mẫu lớp `t` là:
- `L_i = -w_t log(p_t)`.
- `w_t` càng lớn, lỗi ở lớp hiếm càng bị phạt mạnh.
- Nếu mô hình nhầm `paper` nhiều mà `paper` hiếm, tăng `w_paper` là hợp lý.
- Nếu bạn chỉ nhìn accuracy mà không nhìn cross-entropy, bạn có thể bỏ sót việc model quá tự tin khi sai.
- History epoch trong training report phản ánh chính quá trình tối ưu cross-entropy này.

## 94. Phụ lục C: công thức class weighting đang dùng

- Repo có chiến lược class bias ở mức loss và sampler.
- Với loss weighting, code tính `w_c` dựa trên số mẫu của từng lớp.
- Logic nằm ở `_loss_class_weights()` tại `trainer.py:2069`.
- Công thức thực tế trong repo tương ứng:
- `w_c = N / (K * n_c)`.
- `N` là tổng số mẫu của các lớp hiện diện.
- `K` là số lớp hiện diện.
- `n_c` là số mẫu của lớp `c`.
- Nếu lớp `glass` có rất nhiều mẫu thì `w_glass` nhỏ.
- Nếu lớp `folk` có ít mẫu thì `w_folk` lớn.
- Đây là inverse-frequency weighting đã được scale để trung bình tương đối ổn.
- Với snapshot hiện tại:
- `w_folk ≈ 7.0207`.
- `w_glass ≈ 0.3710`.
- `w_paper ≈ 6.1576`.
- Từ đây ta thấy loss của lỗi ở `folk` bị nhân lớn hơn nhiều so với `glass`.
- Không có gì thần kỳ ở công thức này.
- Nó chỉ là cách cân bằng ảnh hưởng của lớp hiếm.
- Nếu class weights quá lớn, model có thể tăng recall lớp hiếm nhưng giảm precision.
- Vì vậy phải đọc cùng confusion matrix.
- Nếu class weights quá thấp, model dễ thiên về lớp đông.
- Không có một công thức duy nhất đúng cho mọi dataset.
- Nhưng công thức hiện tại là lựa chọn đơn giản, dễ giải thích và đã được test.

## 95. Phụ lục D: sampler bias khác gì loss weighting

- Loss weighting can thiệp vào công thức loss.
- Sampler bias can thiệp vào xác suất lấy mẫu trong mini-batch.
- Repo hỗ trợ cả hai qua `class_bias = sampler` hoặc `both`.
- Sampler logic nằm ở `_build_train_sampler()` tại `trainer.py:2048`.
- Khi dùng sampler, mini-batch sẽ thấy lớp hiếm thường xuyên hơn.
- Khi dùng loss weighting, mini-batch có thể vẫn lệch nhưng lỗi lớp hiếm bị nhân lớn hơn.
- Khi dùng `both`, bạn vừa lấy lớp hiếm nhiều hơn vừa phạt sai lớp hiếm nặng hơn.
- Cách này mạnh tay hơn.
- Cách này cũng có thể làm train kém ổn định hơn nếu dataset cực nhỏ.
- Vì vậy repo mặc định các preset đang dùng `loss`.
- Đây là lựa chọn khá bảo thủ.
- Nó giảm độ lệch mà không làm sampler quá “ảo”.

## 96. Phụ lục E: tối ưu bằng Adam trong repo

- Optimizer chính được tạo ở `trainer.py:1297`.
- Repo dùng `torch.optim.Adam`.
- Adam kết hợp momentum bậc một và bậc hai.
- Ký hiệu thường dùng:
- `g_t` là gradient ở bước `t`.
- `m_t = β_1 m_{t-1} + (1-β_1) g_t`.
- `v_t = β_2 v_{t-1} + (1-β_2) g_t^2`.
- Sau hiệu chỉnh bias:
- `m̂_t = m_t / (1-β_1^t)`.
- `v̂_t = v_t / (1-β_2^t)`.
- Cập nhật tham số:
- `θ_t = θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε)`.
- Ý nghĩa thực dụng là learning rate của từng tham số được thích nghi theo lịch sử gradient.
- Điều này giúp train CNN ổn hơn so với SGD thuần trong nhiều bài toán nhỏ và vừa.
- Repo cũng dùng `weight_decay`.
- Weight decay đóng vai trò regularization.
- Nó giúp tránh tham số phình quá lớn.
- Thông số mặc định được ghi vào experiment spec.

## 97. Phụ lục F: embedding và không gian đặc trưng

- Khi dùng CNN làm extractor, ảnh đi qua nhiều tầng tích chập.
- Tầng cuối trước classifier tạo ra embedding.
- Embedding là biểu diễn nén của nội dung ảnh.
- Hai ảnh cùng kiểu vật thể thường có embedding gần nhau hơn ảnh khác loại.
- Trong repo, embedding discovery có dimension `512`.
- Vector này được chuẩn hóa L2 trước khi cluster.
- Công thức:
- `x'_i = x_i / ||x_i||_2`.
- Nếu `||x_i||_2 = 0`, code phải xử lý để tránh chia cho 0.
- Sau chuẩn hóa, dot product và cosine similarity gần như cùng ý nghĩa.
- Đây là lý do spherical k-means hợp lý hơn Euclidean k-means thô trên embedding đã normalize.

## 98. Phụ lục G: spherical k-means viết thành công thức

- Giả sử đã có embedding chuẩn hóa `x_1, x_2, ..., x_N`.
- Mục tiêu của spherical k-means là tìm phân cụm và centroid chuẩn hóa.
- Bước gán cụm:
- `c_i = argmax_k x_i^T μ_k`.
- Bước cập nhật centroid:
- `μ_k = normalize(Σ_{i: c_i = k} x_i)`.
- Hai bước này lặp lại cho tới hội tụ.
- Nếu centroid không đổi nhiều nữa, thuật toán dừng.
- Nếu số vòng lặp đạt giới hạn, thuật toán dừng.
- Vì dùng cosine-like assignment, cluster phản ánh hướng vector nhiều hơn độ lớn.
- Điều này hợp với embedding CNN đã được normalize.
- Trong manifest, mỗi mẫu chỉ có một `cluster_id`.
- Điều đó nghĩa là lời giải là hard assignment.
- Không có membership mềm.
- Không có xác suất một ảnh thuộc 30% cụm A, 70% cụm B.
- Nói cách khác, về mặt toán học repo đang ở họ bài toán partition cứng.

## 99. Phụ lục H: MAD outlier viết thành công thức

- Trong một cluster, giả sử có khoảng cách `d_1, d_2, ..., d_m`.
- Median khoảng cách:
- `med = median(d_1, ..., d_m)`.
- Độ lệch tuyệt đối tới median:
- `a_i = |d_i - med|`.
- MAD:
- `MAD = median(a_1, ..., a_m)`.
- Ngưỡng outlier chính trong repo:
- `threshold = med + 2.5 * MAD`.
- Nếu `MAD` quá nhỏ, repo fallback sang quantile 0.9 của phân phối khoảng cách.
- Đây là heuristic robust.
- Nó không phải định lý tối ưu duy nhất.
- Nhưng nó thường ổn hơn dùng mean + std khi cluster có đuôi dài hoặc có vài điểm rất lạ.
- Ảnh vượt ngưỡng bị đánh dấu `is_cluster_outlier = true`.
- Từ đó chúng đi theo nhánh thận trọng hơn của workflow.

## 100. Phụ lục I: công thức pseudo-label đang dùng

- Với một mẫu chưa nhãn, model cho phân phối xác suất `p_1, ..., p_K`.
- Gọi `p_(1)` là xác suất lớn nhất.
- Gọi `p_(2)` là xác suất lớn thứ hai.
- Repo chấp nhận pseudo-label nếu:
- `p_(1) >= threshold_confidence`.
- Đồng thời:
- `p_(1) - p_(2) >= threshold_margin`.
- Với snapshot hiện tại:
- `threshold_confidence = 0.95`.
- `threshold_margin = 0.25`.
- Điều kiện margin giúp loại mẫu mà model còn phân vân giữa hai lớp.
- Nếu chỉ dùng confidence, có thể nhận nhầm các mẫu top-1 cao nhưng top-2 cũng rất sát.
- Pseudo-label đúng bản chất là self-training có kiểm soát.
- Nó không thay thế human review hoàn toàn.
- Nó chỉ mở rộng tập nhãn từ những dự đoán mà model rất chắc.

## 101. Phụ lục J: accuracy, macro F1 và weighted F1 khác nhau thế nào

- Accuracy là tỷ lệ đoán đúng toàn bộ.
- Nếu dataset lệch lớp, accuracy có thể cao dù lớp hiếm rất tệ.
- Macro F1 tính F1 từng lớp rồi lấy trung bình không trọng số.
- Vì vậy macro F1 nhạy với chất lượng lớp hiếm hơn.
- Weighted F1 tính trung bình có trọng số theo support.
- Nếu lớp `glass` áp đảo, weighted F1 sẽ nghiêng về chất lượng của `glass`.
- Snapshot hiện tại có accuracy khoảng `0.9296`.
- Snapshot hiện tại có macro F1 khoảng `0.8074`.
- Snapshot hiện tại có weighted F1 khoảng `0.9366`.
- Việc weighted F1 cao hơn macro F1 là dấu hiệu dataset lệch lớp.
- Khi chọn mô hình cho production, bạn không nên nhìn một metric duy nhất.

## 102. Phụ lục K: confusion matrix đọc như thế nào

- Confusion matrix của repo được lưu theo thứ tự `labels`.
- Hàng là nhãn thật.
- Cột là nhãn dự đoán.
- Nếu phần tử đường chéo lớn, model đoán đúng nhiều.
- Nếu phần tử ngoài đường chéo lớn, có nhầm lẫn đáng kể.
- Snapshot hiện tại cho lớp `glass` có nhầm sang `paper` nhiều hơn sang `folk`.
- Snapshot hiện tại cho lớp `paper` vẫn có một phần bị đoán thành `glass`.
- Điều này hợp lý vì lớp đông thường “hút” mẫu từ lớp hiếm khi decision boundary chưa tốt.
- Khi bạn thấy recall lớp hiếm cao nhưng precision thấp, thường confusion matrix sẽ cho thấy nhiều false positive đi vào lớp đó.

## 103. Phụ lục L: fuzzy c-means sẽ khác repo hiện tại ở đâu

- Fuzzy c-means không gán cứng mỗi điểm cho đúng một cluster.
- Nó học membership `u_{ik}` cho mọi điểm `i` và mọi cụm `k`.
- Ràng buộc:
- `0 <= u_{ik} <= 1`.
- `Σ_k u_{ik} = 1`.
- Hàm mục tiêu điển hình:
- `J_m = Σ_i Σ_k u_{ik}^m ||x_i - μ_k||^2`.
- `m > 1` là hệ số mờ.
- Cập nhật centroid:
- `μ_k = (Σ_i u_{ik}^m x_i) / (Σ_i u_{ik}^m)`.
- Cập nhật membership:
- `u_{ik} = 1 / Σ_j (||x_i - μ_k|| / ||x_i - μ_j||)^{2/(m-1)}`.
- Repo hiện tại không có bất kỳ bước nào như vậy.
- Repo không tối ưu `J_m`.
- Repo không lưu `u_{ik}`.
- Repo không có hyperparameter fuzzifier `m`.
- Vì vậy nếu ai nói dự án hiện tại đã “dùng logic mờ để phân cụm”, phát biểu đó không khớp code hiện hành.
- Điều đúng phải là:
- Dự án hiện tại dùng CNN embeddings, spherical k-means và human review.
- Nếu muốn fuzzy logic, đó là hướng mở rộng tương lai chứ chưa phải hiện trạng.

## 104. Phụ lục M: chỗ công thức đi vào code

- Softmax và cross-entropy đi vào `CrossEntropyLoss` ở `trainer.py:908` và `trainer.py:1294`.
- Adam đi vào `torch.optim.Adam` ở `trainer.py:1297`.
- Class weights đi vào `_loss_class_weights()` ở `trainer.py:2069`.
- Sampler bias đi vào `_build_train_sampler()` ở `trainer.py:2048`.
- Embedding CNN đi vào `_extract_pretrained_embeddings()` ở `discovery.py:163`.
- L2 normalize đi vào `_normalize_vectors()` ở `discovery.py:247`.
- Spherical k-means đi vào `_spherical_kmeans()` ở `discovery.py:266`.
- MAD outlier đi vào `_detect_cluster_outliers()` ở `discovery.py:301`.
- Pseudo-label thresholds đi vào `pseudo_label()` ở `trainer.py:1126`.
- Metric evaluation được build ở `_build_classification_report()` tại `trainer.py:678`.

## 105. Phụ lục N: test giúp kiểm chứng các ý toán học ở mức workflow

- `localagent/tests/test_training_pseudo_label.py:14` kiểm tra confidence gate.
- `localagent/tests/test_training_artifacts.py:13` kiểm tra fit rồi ghi evaluation.
- `localagent/tests/test_training_artifacts.py:62` kiểm tra evaluate riêng từ checkpoint.
- `localagent/tests/test_training_artifacts.py:136` kiểm tra export ONNX và manifest.
- `localagent/tests/test_pipeline.py:272` kiểm tra embedding, clustering và review artifact.
- Các test này không chứng minh toán học theo nghĩa học thuật.
- Nhưng chúng xác minh rằng công thức và heuristic đã được đóng thành workflow chạy được.

## 106. Phụ lục O: ví dụ số học cho cross-entropy

- Giả sử có ba lớp `folk`, `glass`, `paper`.
- Giả sử ảnh thật thuộc `paper`.
- Mô hình cho xác suất `[0.05, 0.05, 0.90]`.
- Khi đó loss không trọng số là `-log(0.90) ≈ 0.1053`.
- Nếu mô hình cho xác suất `[0.20, 0.50, 0.30]` cho cùng ảnh đó.
- Loss là `-log(0.30) ≈ 1.2040`.
- Nếu mô hình cho xác suất `[0.70, 0.20, 0.10]`.
- Loss là `-log(0.10) ≈ 2.3026`.
- Ta thấy càng tự tin sai, loss càng lớn.
- Điều này giải thích tại sao cross-entropy tạo áp lực mạnh để sửa dự đoán sai.
- Nếu lớp `paper` có trọng số `w_paper = 6.1576`.
- Với dự đoán đúng `0.90`, weighted loss xấp xỉ `6.1576 * 0.1053 ≈ 0.6484`.
- Với dự đoán `0.30`, weighted loss xấp xỉ `6.1576 * 1.2040 ≈ 7.414`.
- Với dự đoán `0.10`, weighted loss xấp xỉ `6.1576 * 2.3026 ≈ 14.178`.
- Đây là cách loss weighting làm lỗi ở lớp hiếm được chú ý hơn.

## 107. Phụ lục P: ví dụ số học cho class weights

- Giả sử có ba lớp.
- Giả sử số mẫu lần lượt là `folk = 464`, `glass = 8776`, `paper = 529`.
- Tổng số mẫu trainable là `9769`.
- Số lớp hiện diện là `3`.
- Trọng số `folk` là `9769 / (3 * 464) ≈ 7.0207`.
- Trọng số `glass` là `9769 / (3 * 8776) ≈ 0.3710`.
- Trọng số `paper` là `9769 / (3 * 529) ≈ 6.1576`.
- Kết quả này khớp training artifact hiện tại.
- Nếu một lớp chỉ có 100 mẫu trong cùng tổng thể, trọng số của nó sẽ còn lớn hơn.
- Nếu các lớp cân bằng hoàn toàn, các trọng số sẽ xấp xỉ nhau.
- Vì vậy công thức này phản ánh trực tiếp lệch lớp.

## 108. Phụ lục Q: ví dụ số học cho pseudo-label

- Giả sử threshold confidence là `0.95`.
- Giả sử threshold margin là `0.25`.
- Trường hợp A: xác suất top1 là `0.98`, top2 là `0.01`.
- Margin là `0.97`.
- Mẫu A được chấp nhận.
- Trường hợp B: top1 là `0.96`, top2 là `0.80`.
- Margin là `0.16`.
- Mẫu B bị từ chối dù top1 vượt `0.95`.
- Trường hợp C: top1 là `0.94`, top2 là `0.10`.
- Margin là `0.84`.
- Mẫu C vẫn bị từ chối vì confidence chưa đủ.
- Điều này cho thấy repo dùng cổng hai lớp để kiểm soát chất lượng pseudo-label.

## 109. Phụ lục R: ví dụ số học cho macro và weighted F1

- Giả sử có hai lớp để đơn giản.
- Lớp A có F1 là `0.95` với support `900`.
- Lớp B có F1 là `0.50` với support `100`.
- Macro F1 là `(0.95 + 0.50) / 2 = 0.725`.
- Weighted F1 là `(0.95 * 900 + 0.50 * 100) / 1000 = 0.905`.
- Chênh lệch lớn giữa hai chỉ số cho thấy lớp nhỏ đang kém hơn nhiều.
- Đây chính là lý do trong repo accuracy và weighted F1 có thể cao hơn macro F1 khá rõ.

## 110. Phụ lục S: ví dụ số học cho spherical k-means

- Giả sử có hai vector đã chuẩn hóa `x_1 = [1, 0]` và `x_2 = [0.8, 0.2]`.
- Giả sử có hai centroid `μ_1 = [1, 0]` và `μ_2 = [0, 1]`.
- Tích vô hướng `x_1^T μ_1 = 1`.
- Tích vô hướng `x_1^T μ_2 = 0`.
- Vì vậy `x_1` vào cluster 1.
- Tích vô hướng `x_2^T μ_1 = 0.8`.
- Tích vô hướng `x_2^T μ_2 = 0.2`.
- Vì vậy `x_2` cũng vào cluster 1.
- Centroid mới của cluster 1 trước chuẩn hóa là `[1.8, 0.2]`.
- Sau chuẩn hóa, centroid này vẫn hướng gần trục `x`.
- Ví dụ này cho thấy assignment phụ thuộc hướng vector nhiều hơn.

## 111. Phụ lục T: ví dụ số học cho MAD outlier

- Giả sử khoảng cách trong một cluster là `[0.10, 0.12, 0.11, 0.13, 0.50]`.
- Median là `0.12`.
- Độ lệch tuyệt đối là `[0.02, 0.00, 0.01, 0.01, 0.38]`.
- MAD là median của dãy trên, tức `0.01`.
- Ngưỡng `median + 2.5 * MAD = 0.145`.
- Mẫu có khoảng cách `0.50` vượt ngưỡng và bị xem là outlier.
- Ví dụ này cho thấy chỉ một điểm rất xa là đủ để bị tách ra mà không làm median biến dạng nhiều.

## 112. Phụ lục U: câu hỏi tự kiểm tra khi đọc công thức

- Công thức này có thật sự xuất hiện trong code hay chỉ là kiến thức nền.
- Công thức này đi vào hàm nào.
- Tham số trong công thức map sang field config nào.
- Artifact nào phản ánh kết quả của công thức đó.
- Có test nào chạm tới hành vi từ công thức này không.
- Nếu không trả lời được, quay lại các mục `104` và `105` của file này.

## 113. Phụ lục V: ví dụ số học cho confusion matrix

- Giả sử lớp `paper` có `support = 54`.
- Nếu model đoán đúng `49` mẫu `paper`, recall của `paper` là `49 / 54 ≈ 0.9074`.
- Nếu tổng số dự đoán `paper` là `91` và đúng `49`, precision của `paper` là `49 / 91 ≈ 0.5385`.
- Từ precision và recall đó, F1 của `paper` xấp xỉ `0.6759`.
- Đây chính là cách report per-class trong repo được diễn giải.
- Một lớp có recall cao nhưng precision thấp nghĩa là model bắt được nhiều mẫu thật của lớp đó.
- Nhưng đồng thời nó cũng kéo nhiều mẫu của lớp khác vào nhầm lớp đó.
- Đó là lý do phải nhìn cả confusion matrix chứ không chỉ một scalar metric.

## 114. Phụ lục W: quy tắc diễn giải số liệu cho người vận hành

- Accuracy cao chưa chắc tốt nếu dataset lệch lớp.
- Macro F1 thấp thường là tín hiệu lớp hiếm đang đau.
- Weighted F1 cao hơn macro F1 nhiều là tín hiệu lệch lớp rõ.
- Confidence pseudo-label cao chưa đủ nếu margin thấp.
- Cluster đẹp bằng mắt chưa đủ nếu review vẫn nhiều stale row.
- ONNX export thành công chưa đủ nếu verify không pass.
- Model tốt trên test chưa đủ nếu preprocess inference lệch manifest.
- Vì vậy mọi con số đều phải đọc trong ngữ cảnh artifact liên quan.

## 115. Phụ lục X: ví dụ số học cho verify ONNX

- Giả sử logits PyTorch là `[1.20, -0.30, 0.10]`.
- Giả sử logits ONNX là `[1.200001, -0.300000, 0.099999]`.
- Hiệu tuyệt đối theo từng phần tử là `[0.000001, 0.000000, 0.000001]`.
- `max_abs_diff` khi đó là `0.000001`.
- Một sai số cỡ này thường chấp nhận được cho verify số học.
- Đây là tinh thần của field `max_abs_diff` trong export report.

## 116. Phụ lục Y: câu hỏi chốt khi đọc số liệu

- Số này đo cái gì.
- Số này sinh ở bước nào của workflow.
- Số này bị ảnh hưởng bởi config nào.
- Số này phản ánh dữ liệu hay phản ánh model.
- Số này có artifact nguồn để kiểm chứng không.
- Số này có bị metric khác phản biện không.
- Nếu không trả lời được, đừng dùng số đó để kết luận mạnh.

## 117. Phụ lục Z: câu hỏi tự kiểm cuối file này

- Bạn đã phân biệt supervised learning với clustering chưa.
- Bạn đã phân biệt hard cluster với fuzzy membership chưa.
- Bạn đã biết loss weighting và sampler bias khác nhau ở đâu chưa.
- Bạn đã biết pseudo-label cần cả confidence và margin chưa.
- Bạn đã biết accuracy không đủ để kết luận trên dataset lệch chưa.
- Bạn đã biết ONNX verify không phải thủ tục hình thức chưa.
- Bạn đã biết công thức nào là hiện trạng code, công thức nào là hướng mở rộng chưa.
- Nếu đã biết, file này đã hoàn thành nhiệm vụ.

## 118. Dòng chốt

- Toán học trong repo này phục vụ workflow thực dụng chứ không phải trình diễn công thức.
- Muốn đọc đúng, luôn nối công thức với config, code và artifact tương ứng.
- Muốn mở rộng đúng, luôn tách rõ hiện trạng và phần dự định thêm mới.

## 119. Chốt phụ cuối

- Công thức tốt nhưng không đi vào code thì chưa tạo ra hệ thống.
- Code chạy được nhưng không hiểu metric thì rất dễ tối ưu sai hướng.
- Artifact đầy đủ nhưng không biết đọc số thì rất khó ra quyết định đúng.
- Vì vậy ba lớp toán học, code và artifact phải luôn đi cùng nhau.

## 120. Chốt phụ cuối nữa

- Khi số liệu đẹp, hãy hỏi nó đẹp cho lớp nào.
- Khi cluster trông đẹp, hãy hỏi review có chấp nhận được không.
- Khi pseudo-label nhận nhiều mẫu, hãy hỏi margin có đủ cao không.
- Khi ONNX chạy được, hãy hỏi verify có thật sự pass không.

## 121. Dòng chốt cuối cùng

- Toán học chỉ thật sự hữu ích khi nó giúp bạn chọn đúng hành động trong workflow này.
- Nếu không giúp ra quyết định, nó mới chỉ là kiến thức nền.
