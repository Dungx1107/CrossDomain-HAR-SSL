# Nhận xét thí nghiệm Cross-Fraction

## 1. Phạm vi và cách đọc kết quả

Báo cáo này tổng hợp thí nghiệm **cross-domain với các fraction nhãn 1%, 5% và 10%**. Kết quả được lấy từ:

- [`aggregated_fraction_detailed.csv`](../outputs_evaluation/aggregated_fraction_detailed.csv): F1 và độ lệch chuẩn theo từng cấu hình.
- [`pivot_fraction_f1_report.csv`](../outputs_evaluation/pivot_fraction_f1_report.csv): bảng pivot đã tổng hợp theo fraction và protocol.
- Các file `metrics.json` và ma trận nhầm lẫn trong [`outputs_evaluation/cross_fraction/`](../outputs_evaluation/cross_fraction/).

Có 2 hướng chuyển miền (`UCI HAR → MotionSense` và `MotionSense → UCI HAR`), 2 phương pháp (`contrastive`, `prototype`), 4 backbone (`CNN-Transformer`, `Standard`, `TSTCC`, `ViT-1D`), 3 mức fraction, 2 protocol và 5 seed cho mỗi cấu hình. Vì vậy, các con số trong bảng tổng hợp nên được hiểu là **macro-F1 trung bình**, còn giá trị `±` là độ lệch chuẩn giữa các seed/domain pair theo cách tổng hợp của pipeline.

> Lưu ý: macro-F1 được báo cáo theo thang phần trăm. Do cross-domain là bài toán khó và hai domain có phân phối khác nhau, nên cần ưu tiên so sánh cùng protocol, cùng fraction và cùng hướng chuyển miền thay vì chỉ nhìn một kết quả đơn lẻ.

## 2. Kết quả tổng quan

### 2.1. Full fine-tuning và linear probing

| Fraction | Full fine-tuning | Linear probing | Chênh lệch |
|---|---:|---:|---:|
| 1% | 72.11 | 55.83 | +16.28 |
| 5% | 85.09 | 69.27 | +15.82 |
| 10% | 86.95 | 72.02 | +14.93 |

Đây là kết luận mạnh nhất của thí nghiệm: **fine-tuning toàn bộ encoder luôn tốt hơn linear probing**. Khi chỉ huấn luyện classifier trên representation cố định, đặc trưng SSL chưa đủ thích nghi với domain đích; khoảng cách khoảng 15–16 điểm F1 cho thấy phần thích nghi representation là rất quan trọng trong cross-domain HAR.

Khi tăng nhãn từ 1% lên 5%, F1 tăng mạnh:

- Full fine-tuning: `+12.98` điểm.
- Linear probing: `+13.44` điểm.

Từ 5% lên 10%, mức tăng nhỏ hơn:

- Full fine-tuning: `+1.86` điểm.
- Linear probing: `+2.75` điểm.

Điều này gợi ý **5% nhãn đã cung cấp phần lớn tín hiệu cần thiết cho việc thích nghi domain**; thêm nhãn lên 10% vẫn có ích nhưng lợi ích biên giảm. Tuy nhiên, ở mức 1% kết quả vẫn có độ dao động cao, nên không nên diễn giải một seed tốt là khả năng tổng quát ổn định.

### 2.2. So sánh hai phương pháp SSL

| Method | F1 trung bình trên các cấu hình |
|---|---:|
| Prototype | 75.93 |
| Contrastive | 71.47 |

`Prototype` tốt hơn `contrastive` khoảng **4.46 điểm F1** khi gộp các backbone, fraction và protocol. Ưu thế này rõ nhất ở các cấu hình fine-tuning và trong vùng ít nhãn. Tuy vậy, phương pháp không phải yếu tố duy nhất quyết định kết quả: một backbone phù hợp có thể làm thay đổi thứ hạng giữa các cấu hình.

### 2.3. So sánh backbone

| Backbone | F1 trung bình với prototype | F1 trung bình với contrastive |
|---|---:|---:|
| Standard | 80.78 | 80.09 |
| CNN-Transformer | 77.27 | 71.47 |
| TSTCC | 75.93 | 76.77 |
| ViT-1D | 65.72 | 60.33 |

Nhận xét chính:

- **Standard** là backbone ổn định và mạnh nhất khi xét trung bình toàn bộ cấu hình.
- **CNN-Transformer** đứng sau Standard; với contrastive, kết quả thấp hơn đáng kể so với prototype.
- **TSTCC** có giá trị tốt nhất trong cấu hình cụ thể: `prototype + TSTCC + 10% + full fine-tuning` đạt **91.25 F1** trung bình. Đây là cấu hình tốt nhất trong bảng tổng hợp.
- **ViT-1D** là backbone yếu nhất. Khoảng cách so với Standard rất lớn, đặc biệt ở linear probing và 1% nhãn. Điều này cho thấy ViT-1D khó học representation hữu ích khi dữ liệu nhãn ít và domain shift lớn; không nên kết luận rằng kiến trúc Transformer nói chung kém, mà cụ thể là cấu hình ViT-1D hiện tại chưa phù hợp hoặc chưa được tối ưu đủ cho bài toán này.

## 3. Các cấu hình tốt và chưa tốt

### Cấu hình tốt

Các cấu hình có F1 trung bình cao nhất:

| Method | Backbone | Fraction | Protocol | F1 |
|---|---|---:|---|---:|
| Prototype | TSTCC | 10% | Full fine-tuning | **91.25 ± 1.06** |
| Contrastive | TSTCC | 10% | Full fine-tuning | **90.76 ± 1.66** |
| Prototype | Standard | 5% | Full fine-tuning | **90.40 ± 0.74** |
| Prototype | Standard | 10% | Full fine-tuning | **90.35 ± 0.96** |
| Prototype | TSTCC | 5% | Full fine-tuning | **89.25 ± 1.67** |

Điểm đáng chú ý là `prototype + Standard + 5%` gần như đã đạt kết quả của `prototype + Standard + 10%`. Đây là tín hiệu tích cực về hiệu quả nhãn. Với 10% nhãn, `TSTCC` có lợi thế rõ hơn và cho kết quả tốt nhất.

### Cấu hình chưa tốt

Nhóm kết quả thấp nhất chủ yếu là **linear probing với ViT-1D**:

- `contrastive + ViT-1D + 1% + linear probing`: **40.05 ± 3.02**
- `prototype + ViT-1D + 1% + linear probing`: **47.48 ± 5.80**
- `contrastive + ViT-1D + 5% + linear probing`: **47.52 ± 1.77**
- `contrastive + ViT-1D + 10% + linear probing`: **52.53 ± 1.35**

Độ lệch chuẩn cũng lớn hơn ở nhiều cấu hình ViT-1D, đặc biệt `prototype + ViT-1D` ở 5% và 10% linear probing. Vì vậy, đây không chỉ là vấn đề F1 thấp mà còn là vấn đề **độ ổn định giữa các seed**.

## 4. Ảnh hưởng của hướng chuyển miền

Khi gộp các method, backbone, fraction, protocol và seed:

| Hướng | F1 trung bình |
|---|---:|
| UCI HAR → MotionSense | **76.90 ± 11.34** |
| MotionSense → UCI HAR | **70.19 ± 16.69** |

Hướng `UCI HAR → MotionSense` tốt hơn khoảng **6.71 điểm F1** và ổn định hơn. Như vậy domain shift có tính **bất đối xứng**: representation học từ MotionSense chuyển sang UCI HAR khó hơn chiều ngược lại. Đây là lý do nên báo cáo riêng từng hướng thay vì chỉ lấy trung bình hai chiều.

## 5. Phân tích ma trận nhầm lẫn

Các ma trận trong [`outputs_evaluation/cross_fraction/`](../outputs_evaluation/cross_fraction/) đều chuẩn hóa theo hàng: hàng là nhãn thật, cột là nhãn dự đoán. Đường chéo càng cao thì khả năng nhận diện lớp đó càng tốt.

### 5.1. Các hoạt động dễ nhầm với nhau

1. **Walking, Upstairs và Downstairs**

   Đây là nhóm chuyển động có tín hiệu gia tốc và nhịp chuyển động tương tự nhau. Ở các cấu hình tốt, lỗi còn lại tập trung chủ yếu ở:

   - Walking → Downstairs hoặc Upstairs.
   - Upstairs → Walking/Standing.
   - Downstairs → Walking/Upstairs.

   Ví dụ, ma trận của một cấu hình `contrastive + CNN-Transformer + UCI HAR → MotionSense + 10% + full fine-tuning` cho thấy Walking nhận đúng 89.1%, nhưng nhầm sang Downstairs 10.3%; Upstairs nhầm sang Downstairs 10.0%. Điều này phù hợp với việc ba lớp đều là hoạt động di chuyển và khó tách hoàn toàn khi sensor/domain thay đổi.

2. **Sitting và Standing**

   Hai lớp tĩnh có xu hướng bị nhầm lẫn theo tư thế, vị trí đặt sensor và cách cắt cửa sổ tín hiệu. Ở một số ma trận, Sitting → Standing và Standing → Sitting là lỗi lớn nhất trong nhóm lớp tĩnh. Khi representation tốt, cả hai lớp này có thể đạt gần 100% trên đường chéo; vì vậy đây là nhóm phụ thuộc mạnh vào chất lượng representation và domain alignment.

3. **Suy biến về một lớp, đặc biệt là Standing, ở cấu hình ViT-1D ít nhãn**

   Ma trận đại diện của `prototype + ViT-1D + UCI HAR → MotionSense + 1% + linear probing` có F1 chỉ **42.09%**. Mô hình dự đoán đúng Walking 24.8%, Upstairs 36.3%, Downstairs 12.6%, trong khi nhiều mẫu bị dồn sang Standing: Walking → Standing 54.6%, Upstairs → Standing 31.6%, Downstairs → Standing 50.2%. Đây là dấu hiệu classifier trên representation cố định chưa tách được các hoạt động động, chứ không đơn thuần là một vài lỗi ngẫu nhiên.

### 5.2. Ma trận của cấu hình tốt

Ma trận của `prototype + TSTCC + UCI HAR → MotionSense + 10% + full fine-tuning` (seed 100) đạt macro-F1 **91.40%**. Đường chéo của Walking/Upstairs/Downstairs lần lượt khoảng 86.4%/85.1%/88.3%, còn Sitting và Standing đạt 100%. Lỗi đáng kể nhất là:

- Walking → Downstairs: 9.7%.
- Upstairs → Standing: 12.6%.
- Downstairs → Standing: 6.4%.

Mô hình đã tách tốt hai lớp tĩnh và phần lớn các lớp chuyển động; phần khó còn lại chủ yếu là ranh giới giữa chuyển động đi bộ/lên xuống cầu thang và ảnh hưởng của domain shift.

## 6. Kết luận thực nghiệm

1. **Full fine-tuning là lựa chọn nên dùng** cho cross-domain fraction. Linear probing phù hợp để đo chất lượng representation thuần túy, nhưng kết quả thấp hơn rõ rệt và dễ suy biến khi chỉ có 1% nhãn.
2. **Prototype đang là phương pháp có kết quả tổng thể tốt hơn contrastive** trong thí nghiệm này.
3. **TSTCC là backbone nổi bật ở cấu hình tốt nhất**, còn Standard là backbone có trung bình toàn cục tốt và ổn định hơn.
4. **ViT-1D là điểm yếu chính**, nhất là khi ít nhãn và linear probing; cần xem lại thiết kế đầu vào, kích thước model, regularization và chiến lược fine-tuning trước khi dùng làm backbone chính.
5. **5% nhãn là điểm cân bằng tốt** giữa chi phí gán nhãn và chất lượng; 10% vẫn cải thiện kết quả, đặc biệt cho TSTCC, nhưng mức tăng so với 5% nhỏ hơn nhiều so với bước 1% → 5%.
6. **Các lỗi cần ưu tiên xử lý là Walking/Upstairs/Downstairs và Sitting/Standing**. Các hướng tiếp theo nên tập trung vào domain-invariant temporal features, class-aware augmentation và phân tích riêng từng hướng source–target.

## 7. Khuyến nghị cho các thí nghiệm tiếp theo

- Báo cáo đồng thời macro-F1, per-class F1 và độ lệch chuẩn theo seed; không chỉ báo cáo một seed tốt nhất.
- Giữ `prototype + TSTCC + full fine-tuning` làm baseline mạnh, đồng thời giữ `prototype + Standard + 5%` làm baseline hiệu quả nhãn.
- Kiểm tra calibration và phân bố confidence cho các mẫu bị dồn sang Standing.
- Thử loss có trọng số hoặc margin/metric learning cho ba lớp chuyển động gần nhau.
- Phân tích domain shift theo từng cặp source–target và cân nhắc adaptation không giám sát ở domain đích.

## 8. Nhận xét thí nghiệm Cross-Domain K-shot

### 8.1. Thiết lập thí nghiệm

Phần K-shot đánh giá khả năng thích nghi khi domain đích chỉ có một số lượng mẫu gán nhãn cố định cho mỗi lớp, với các mức **10-shot, 20-shot, 50-shot và 100-shot**. So với phần fraction, thí nghiệm này mở rộng lên 4 dataset (`HHAR Phone`, `HHAR Watch`, `MotionSense`, `UCI HAR`) và 12 hướng source → target khác nhau.

Kết quả được lấy từ [`aggregated_k_shot_detailed.csv`](../outputs_evaluation/aggregated_k_shot_detailed.csv) và [`pivot_k_shot_f1_report.csv`](../outputs_evaluation/pivot_k_shot_f1_report.csv). Các hình đã plot nằm trong [`outputs_evaluation/figures/cross_k_shot/`](../outputs_evaluation/figures/cross_k_shot/).

> **Lưu ý về phạm vi so sánh:** trong file tổng hợp, các mức 10/20/50-shot chủ yếu có kết quả cho phương pháp `prototype`, còn `contrastive` xuất hiện đầy đủ ở 100-shot. Vì vậy, đường cong theo số shot nên được dùng để đánh giá xu hướng của prototype; không nên diễn giải rằng contrastive kém hơn ở 10/20/50 shot khi chưa có cùng cấu hình để so sánh.

### 8.2. Xu hướng theo số lượng shot

| Số shot | Full fine-tuning | Linear probing |
|---:|---:|---:|
| 10 | 55.10 ± 20.15 | 47.34 ± 17.25 |
| 20 | 64.67 ± 17.88 | 51.74 ± 18.72 |
| 50 | 68.93 ± 18.35 | 58.08 ± 16.93 |
| 100 | 70.46 ± 19.77 | 58.40 ± 18.23 |

Xu hướng tổng thể là tăng số shot giúp cải thiện macro-F1, nhưng mức tăng giảm dần:

- Từ 10 lên 20 shot: full fine-tuning tăng **9.57 điểm**, linear probing tăng **4.40 điểm**.
- Từ 20 lên 50 shot: full fine-tuning tăng **4.26 điểm**, linear probing tăng **6.34 điểm**.
- Từ 50 lên 100 shot: full fine-tuning chỉ tăng **1.53 điểm**, linear probing gần như bão hòa, tăng **0.32 điểm**.

Kết quả cho thấy **10 shot là vùng thiếu dữ liệu rõ rệt**, còn khoảng 50–100 shot bắt đầu đi vào vùng lợi ích biên thấp. Full fine-tuning vẫn tốt hơn linear probing ở mọi mức shot, với khoảng cách lần lượt là **7.76, 12.93, 10.85 và 12.06 điểm F1** ở 10, 20, 50 và 100 shot. Điều này củng cố kết luận ở phần fraction: representation cần được cập nhật để thích nghi với domain đích.

Tuy nhiên, độ lệch chuẩn khá lớn, khoảng 16–20 điểm khi gộp các cặp domain. Đây là dấu hiệu cho thấy **độ khó của cặp source–target ảnh hưởng mạnh hơn việc chỉ tăng số shot**. Một cặp dễ có thể đạt F1 cao với ít shot, trong khi một cặp khó vẫn thấp dù dùng 100 shot.

### 8.3. Vị trí nên chèn các hình

Nên đặt hình **ngay sau đoạn mô tả xu hướng theo shot và trước mục 8.3**, theo thứ tự sau:

#### Hình 1 — Đường cong học theo số shot

Đặt sau bảng tổng hợp ở mục 8.2:

```markdown
![Đường cong macro-F1 theo số shot](../outputs_evaluation/figures/cross_k_shot/learning_curve_f1.png)

**Hình X.** Macro-F1 theo số lượng shot cho full fine-tuning và linear probing.
```

Đây nên là **hình chính** của phần K-shot vì nó trả lời trực tiếp câu hỏi: thêm nhãn có cải thiện không, và cải thiện đến mức nào. Trong phần nhận xét, nên nhấn mạnh độ dốc lớn từ 10 → 20 shot và hiện tượng bão hòa sau 50 shot.

#### Hình 2 — So sánh method/backbone ở 100-shot

Đặt sau mục 8.4, nơi so sánh các method và backbone:

```markdown
![So sánh method và backbone ở 100-shot](../outputs_evaluation/figures/cross_k_shot/bar_chart_100shot.png)

**Hình X.** So sánh macro-F1 giữa prototype/contrastive và Standard/CNN-Transformer ở 100-shot.
```

Hình này phù hợp để minh họa rằng ở 100-shot, full fine-tuning đạt khoảng 69–71% tùy cấu hình, trong khi linear probing thấp hơn. Cần chú thích rằng error bar thể hiện độ phân tán giữa các kết quả, không phải khoảng tin cậy thống kê nếu pipeline không tính confidence interval.

#### Hình 3 — So sánh theo hướng source → target

Đặt sau mục 8.5:

```markdown
![Kết quả theo hướng source target ở 100-shot](../outputs_evaluation/figures/cross_k_shot/source_grid_2x2_100_shot.png)

**Hình X.** Macro-F1 theo từng hướng chuyển miền ở 100-shot.
```

Đây là hình quan trọng để giải thích độ lệch chuẩn lớn trong bảng tổng hợp. Nếu hình bị quá nhỏ khi đưa vào báo cáo, nên đặt ở chế độ full-width hoặc tách thành các hình 10/20/50/100-shot; không nên gom cả bốn mức vào một hình nhỏ.

#### Hình 4 — Hybrid/source grid

Các hình [`hybrid_grid_2x2_20_shot.png`](../outputs_evaluation/figures/cross_k_shot/hybrid_grid_2x2_20_shot.png) và [`hybrid_grid_2x2_100_shot.png`](../outputs_evaluation/figures/cross_k_shot/hybrid_grid_2x2_100_shot.png) nên để ở phần phụ lục hoặc subsection phân tích chi tiết, không nên đặt trước đường cong học. Chúng nhiều thông tin hơn nhưng khó đọc nếu người đọc chưa biết xu hướng tổng quan.

### 8.4. So sánh method và backbone ở 100-shot

Ở 100-shot, kết quả trung bình theo method/backbone là:

| Method | Backbone | Full fine-tuning | Linear probing |
|---|---|---:|---:|
| Prototype | Standard | 71.28 | 61.52 |
| Prototype | CNN-Transformer | 70.57 | 60.81 |
| Contrastive | Standard | 71.02 | 63.60 |
| Contrastive | CNN-Transformer | 68.96 | 47.66 |

Một số nhận xét:

- `Prototype + Standard` là cấu hình tốt nhất khi full fine-tuning, đạt **71.28 F1**.
- `Contrastive + Standard` có kết quả linear probing cao nhất, **63.60 F1**, cho thấy representation của cấu hình này tương đối hữu ích ngay cả khi encoder bị đóng băng.
- `Contrastive + CNN-Transformer` có sự suy giảm rất lớn khi chuyển từ full fine-tuning sang linear probing: từ **68.96 xuống 47.66 F1**. Đây là dấu hiệu representation cần được thích nghi mạnh với domain đích.
- Hai backbone Standard và CNN-Transformer khá gần nhau khi full fine-tuning trong prototype, nhưng khác biệt rõ hơn với contrastive và linear probing. Vì vậy, không nên xếp hạng backbone chỉ dựa trên một protocol.

### 8.5. Độ khó của từng hướng chuyển miền

Khi gộp các shot, method, backbone và protocol, các hướng có kết quả cao nhất là:

| Source → Target | F1 trung bình |
|---|---:|
| MotionSense → UCI HAR | 78.95 ± 8.13 |
| HHAR Phone → UCI HAR | 78.23 ± 9.15 |
| HHAR Watch → MotionSense | 74.79 ± 7.84 |
| UCI HAR → MotionSense | 74.64 ± 7.32 |
| HHAR Watch → UCI HAR | 74.53 ± 10.45 |
| HHAR Phone → MotionSense | 71.82 ± 8.63 |
| HHAR Watch → HHAR Phone | 60.03 ± 9.23 |
| MotionSense → HHAR Phone | 58.17 ± 10.89 |
| UCI HAR → HHAR Phone | 55.62 ± 12.10 |
| HHAR Phone → HHAR Watch | 38.61 ± 8.29 |
| UCI HAR → HHAR Watch | 31.09 ± 7.57 |
| MotionSense → HHAR Watch | **27.81 ± 8.93** |

Có sự phân tầng rất rõ:

1. **Đích UCI HAR hoặc MotionSense thường dễ hơn**, với F1 khoảng 71–79. Hai dataset này có số lớp và semantics gần nhau hơn trong thí nghiệm hiện tại.
2. **Các hướng vào HHAR Phone khó hơn**, đặc biệt khi source là UCI HAR hoặc MotionSense.
3. **HHAR Watch là target khó nhất**: ba hướng thấp nhất đều có đích là HHAR Watch, trong đó `MotionSense → HHAR Watch` chỉ đạt 27.81 F1. Khả năng cao domain shift của thiết bị đeo tay, vị trí sensor và cách ghi nhận tín hiệu làm representation khó chuyển trực tiếp.
4. Độ lệch chuẩn cao ở `UCI HAR → HHAR Phone` và `MotionSense → HHAR Phone` cho thấy kết quả nhạy với cách chọn các mẫu few-shot.

Vì vậy, với K-shot nên báo cáo **ma trận source–target**, không chỉ một đường cong trung bình. Đường cong trung bình có thể che khuất việc một số cặp domain đã bão hòa ở mức tốt, trong khi các cặp vào HHAR Watch vẫn là nút thắt chính.

### 8.6. Kết luận cho phần K-shot

- Tăng shot từ 10 lên 20 đem lại lợi ích lớn nhất cho full fine-tuning; sau 50 shot, lợi ích biên giảm rõ rệt.
- Full fine-tuning đáng tin cậy hơn linear probing trong toàn bộ các mức shot.
- Ở 100-shot, Standard là backbone cân bằng tốt; `prototype + Standard` tốt nhất với full fine-tuning, còn `contrastive + Standard` tốt nhất với linear probing.
- Độ khó source–target chi phối kết quả mạnh. `MotionSense/UCI HAR → HHAR Watch` cần được xem là nhóm cần ưu tiên cải thiện.
- Hình nên trình bày theo thứ tự: **learning curve → bar chart 100-shot → source-target grid → hybrid grid/phụ lục**. Thứ tự này đi từ xu hướng tổng quát đến giải thích chi tiết.

Nếu viết thành báo cáo khoa học, phần K-shot nên dùng `learning_curve_f1.png` làm hình chính, `bar_chart_100shot.png` làm hình so sánh kiến trúc, và `source_grid_2x2_100_shot.png` làm hình phân tích domain shift. Các hình 10/20/50-shot và hybrid grid nên đưa vào phụ lục để tránh làm phần kết quả chính quá dày.
