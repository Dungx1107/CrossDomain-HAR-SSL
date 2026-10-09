# Báo cáo phân tích Cross-Domain HAR theo K-shot

## 1. Tóm tắt điều hành

Bài toán được khảo sát là nhận dạng hoạt động người (HAR) khi mô hình phải chuyển từ một tập dữ liệu cảm biến sang một tập dữ liệu khác, trong điều kiện chỉ có một số lượng nhỏ mẫu gán nhãn ở miền đích. Đây là vấn đề quan trọng vì dữ liệu cảm biến thường thay đổi đồng thời về vị trí đeo, thiết bị, tần số lấy mẫu và thói quen vận động; một bộ phân loại tốt trên miền nguồn vì vậy chưa chắc giữ được khả năng phân biệt ở miền đích. Thực nghiệm đã đối chiếu ba phương pháp học biểu diễn (Prototype, Contrastive và CrossHAR), hai backbone (standard 1D-CNN và CNN-Transformer), hai protocol (full fine-tuning và linear probing), cùng các mức shot khả dụng trong cây kết quả.

Phát hiện thứ nhất là backbone standard nhìn chung ổn định và đạt điểm cao hơn CNN-Transformer trong các cấu hình pocket-level, đặc biệt khi biểu diễn bị đóng băng. Phát hiện thứ hai là Prototype giữ được chất lượng tốt trong cả full fine-tuning và linear probing, trong khi Contrastive phụ thuộc rõ hơn vào việc cập nhật toàn bộ mạng. Phát hiện thứ ba là thêm shot giúp tăng F1 mạnh từ 10 lên 20 và 50 shot, nhưng lợi ích biên giảm khi tiến tới 100 shot, phù hợp với hiện tượng bão hòa do giới hạn domain shift. Phát hiện thứ tư là target `hhar_watch` tạo ra suy giảm lớn cho mọi phương pháp, cho thấy nút thắt chính nằm ở động học vật lý của cảm biến cổ tay chứ không chỉ ở lựa chọn loss. Phát hiện thứ năm là lỗi lớn nhất tập trung ở các cặp Walking–Upstairs và Downstairs–Walking.

Cấu hình có accuracy cao nhất trong các bản ghi pocket-level là Prototype + standard + 100-shot + full fine-tuning trên MotionSense → UCI HAR. Insight tổng quát là inductive bias phù hợp và representation chuyển giao quan trọng hơn việc tăng độ phức tạp; LP cho thấy Prototype và CrossHAR đã tạo feature hữu ích trước adaptation.

## 2. Bối cảnh và cơ chế phương pháp

### 2.1. Bài toán Cross-Domain HAR

Mỗi mẫu là một cửa sổ chuỗi gia tốc hoặc tín hiệu quán tính với nhãn Walking, Upstairs, Downstairs, Sitting hoặc Standing. Trong cross-domain, phân phối nguồn và đích thay đổi theo vị trí cảm biến, người dùng, thiết bị hoặc tiền xử lý. Mục tiêu là học không gian đặc trưng vẫn tách lớp sau thay đổi miền; macro F1 được dùng để cân bằng đóng góp của các lớp.

Phân tích đã tách riêng 76 bản ghi có target chứa `hhar_watch` khỏi 236 bản ghi pocket-level, vì target cổ tay là một cấu hình đo khác về mặt cơ học. Các phép so sánh nhiều phương pháp chỉ dùng giao các pair có đủ bản ghi; dữ liệu thiếu không được thay bằng không hoặc nội suy.

### 2.2. Prototype

Prototype học một vector đại diện, hay class centroid, cho từng hoạt động trong không gian embedding. Loss làm khoảng cách tới prototype của lớp đúng nhỏ hơn khoảng cách tới lớp khác, nên quyết định phân loại dựa trên hình học khoảng cách. Trong few-shot, centroid có thể ước lượng từ ít mẫu; fine-tuning di chuyển embedding theo miền đích, còn linear probing kiểm tra xem cấu trúc đó đã có sẵn hay chưa.

### 2.3. Contrastive

Contrastive learning tạo các cặp hoặc nhóm positive–negative: hai view của cùng mẫu được kéo gần, còn mẫu khác lớp bị đẩy xa. Cơ chế này tăng tính bất biến với nhiễu, nhưng phụ thuộc vào augmentation, nhiệt độ loss và độ đầy đủ của negative samples. Contrastive CNN-Transformer giảm mạnh ở linear probing, phù hợp với diễn giải rằng representation này cần thích nghi thêm.

### 2.4. CrossHAR

CrossHAR hướng trực tiếp vào chuyển miền: representation giữ thông tin hoạt động đồng thời giảm khác biệt nguồn–đích. Khác với Prototype nhấn mạnh centroid và Contrastive nhấn mạnh cặp mẫu, CrossHAR kết hợp tín hiệu phân biệt lớp với mục tiêu làm các miền gần nhau trong embedding. Tuy nhiên, alignment không thể loại bỏ nút thắt khi hai cảm biến ghi nhận động học khác bản chất; CrossHAR chỉ đạt LP tốt hơn trên standard, không khắc phục được target cổ tay.

### 2.5. Hai backbone

Backbone standard được diễn giải là 1D-CNN. Kernel cục bộ có inductive bias về tính gần kề theo thời gian, nên dễ phát hiện chu kỳ bước chân, đỉnh gia tốc và thay đổi năng lượng trong cửa sổ ngắn. Bias này giảm số mẫu cần học pattern cơ bản và phù hợp với tín hiệu thời gian đều.

CNN-Transformer dùng tích chập tạo đặc trưng cục bộ rồi dùng attention để mô hình hóa quan hệ xa. Thiết kế này hữu ích khi hoạt động được nhận biết bởi thứ tự nhiều pha, nhưng attention có nhiều bậc tự do và cần nhiều mẫu để phân biệt pattern hữu ích với nhiễu. Khi LP đóng băng các tầng này, classifier không sửa được các quan hệ đã học.

### 2.6. Hai protocol

Full fine-tuning cập nhật backbone và head bằng dữ liệu miền đích, cho phép điều chỉnh scale, orientation và pattern thời gian nhưng có rủi ro overfitting khi ít shot. Linear probing đóng băng backbone và chỉ huấn luyện head, nên đo trực tiếp tính transferable của feature với ít tham số hơn. FT vượt LP lớn gợi ý representation cần adaptation; LP vượt FT cần được tái lập vì có thể do seed.

## 3. Phân tích kết quả

### 3.1. So sánh phương pháp

Bảng 1 trình bày so sánh tại 100-shot, protocol linear probing, trên backbone standard và pocket-level để trả lời câu hỏi phương pháp nào tạo feature đóng băng tốt hơn. Các giá trị được tính trên các bản ghi hiện có; CrossHAR có 7 pair, còn Prototype và Contrastive có 9 pair.

| Method | Macro F1 (%) | Pair khả dụng |
|---|---:|---:|
| CrossHAR | 73.68 | 7 |
| Prototype | 70.72 | 9 |
| Contrastive | 72.20 | 9 |

Ở backbone standard, CrossHAR đạt 73.68%, cao hơn Prototype 2.96 điểm và Contrastive 1.48 điểm; cơ chế alignment liên miền có thể đã giữ được các đặc trưng hình thái chung khi backbone có bias cục bộ. Ngược lại, ở CNN-Transformer, Prototype đạt 69.09%, cao hơn CrossHAR 7.30 điểm và Contrastive 14.48 điểm, cho thấy prototype ít phụ thuộc vào attention đã được huấn luyện hoàn hảo hơn. Vì vậy, thứ hạng method phụ thuộc vào backbone chứ không chỉ vào objective.

Khi chuyển từ LP sang FT, lợi ích lớn nhất xuất hiện ở Contrastive + CNN-Transformer: F1 tăng từ 54.61% lên 79.65%, tức 25.04 điểm phần trăm. Mức tăng này gợi ý feature contrastive của backbone này chưa tự căn chỉnh với miền đích, nhưng đủ thông tin để fine-tuning khai thác lại; Contrastive + standard chỉ tăng 9.85 điểm, trong khi CrossHAR + standard tăng 8.43 điểm. Prototype + standard đạt 82.32% FT và chỉ giảm xuống 70.72% LP, nên representation của nó có tính chuyển giao tốt hơn trong cùng backbone.

**Nhận xét:** Prototype là phương pháp cân bằng nhất giữa chất lượng feature và khả năng thích nghi, nhưng không thắng tuyệt đối ở mọi backbone; CrossHAR dẫn đầu LP trên standard, còn Prototype dẫn đầu CNN-Transformer. **Giải thích:** centroid cung cấp hình học lớp rõ ràng, còn alignment của CrossHAR chỉ phát huy khi backbone đã biểu diễn được pattern cục bộ ổn định; Contrastive cần fine-tuning mạnh hơn vì quan hệ positive–negative chưa bảo đảm cùng hình học ở domain mới.

**Kết luận mục 3.1:** Không có bằng chứng rằng một loss duy nhất thắng trong mọi cấu hình. Nếu ưu tiên frozen feature, standard + CrossHAR là lựa chọn mạnh tại 100-shot; nếu cần một cấu hình FT tổng quát, Prototype + standard cho điểm cao nhất và khoảng cách FT–LP nhỏ hơn các cấu hình CNN-Transformer.

### 3.2. So sánh kiến trúc

Bảng 2 dùng trung bình pocket-level theo backbone, protocol và shot; mỗi dòng có 9 bản ghi ở 10/20 shot và 16 hoặc 25 bản ghi ở 50/100 shot do coverage giữa phương pháp không đồng nhất.

| Backbone | Protocol | 10-shot | 100-shot |
|---|---|---:|---:|
| standard 1D-CNN | FT | 65.24 | 82.16 |
| standard 1D-CNN | LP | 56.27 | 72.08 |
| CNN-Transformer | FT | 65.63 | 80.37 |
| CNN-Transformer | LP | 55.13 | 61.83 |

Tại 100-shot, standard thắng CNN-Transformer ở cả FT và LP, lần lượt 1.79 và 10.25 điểm phần trăm. Chênh lệch LP lớn hơn nhiều so với FT vì fine-tuning cho phép CNN-Transformer sửa một phần biểu diễn không phù hợp, còn LP buộc classifier làm việc với feature cố định. Kết quả này cho thấy độ phức tạp attention không tự động chuyển thành khả năng chuyển miền.

Tại 10-shot, CNN-Transformer FT cao hơn standard 0.39 điểm, nhưng LP thấp hơn 1.14 điểm. Sự mong manh rõ hơn ở đường LP: CNN-Transformer chỉ tăng 6.70 điểm từ 10 đến 100 shot, trong khi standard tăng 15.81 điểm. Attention cần nhiều mẫu để học tương quan xa; CNN đã mã hóa trước motif ngắn như nhịp bước và đỉnh gia tốc.

Khi chuyển từ FT sang LP ở 100-shot, standard sụt 10.08 điểm, còn CNN-Transformer sụt 18.54 điểm. Sụt giảm nhỏ hơn của standard cho thấy feature 1D-CNN transferable hơn ở pocket-level; các chênh lệch gần 1 điểm vẫn cần đọc thận trọng vì 55/312 bản ghi có std trên 5 điểm.

**Nhận xét:** Standard thắng ở 100-shot và có độ suy giảm LP nhỏ hơn, trong khi CNN-Transformer chỉ nhỉnh nhẹ ở 10-shot FT. **Giải thích:** inductive bias cục bộ của CNN giảm phương sai khi dữ liệu ít; attention có năng lực biểu diễn cao nhưng cần nhiều quan sát để ước lượng pattern xa và vẫn dễ bị đóng băng sai khi LP.

**Kết luận mục 3.2:** Với dữ liệu hiện có, standard là backbone có tính transferable và ổn định hơn. CNN-Transformer chỉ nên được ưu tiên khi có đủ dữ liệu hoặc khi mục tiêu nghiên cứu yêu cầu mô hình hóa phụ thuộc dài hạn và chấp nhận chi phí thích nghi cao hơn.

### 3.3. So sánh protocol

Bảng 3 chọn các cấu hình có đủ cả FT và LP và báo cáo trung bình pocket-level. Khoảng cách được tính là FT trừ LP, do đó phản ánh lượng hiệu năng cần nhờ adaptation.

| Method | Backbone | Shot | FT (%) | LP (%) | FT−LP |
|---|---|---:|---:|---:|---:|
| Prototype | standard | 100 | 82.32 | 70.72 | +11.60 |
| Contrastive | standard | 100 | 82.05 | 72.20 | +9.85 |
| CrossHAR | standard | 100 | 82.11 | 73.68 | +8.43 |
| Prototype | CNN-Transformer | 100 | 81.13 | 69.09 | +12.04 |
| Contrastive | CNN-Transformer | 100 | 79.65 | 54.61 | +25.04 |
| CrossHAR | CNN-Transformer | 100 | 80.33 | 61.79 | +18.54 |

Ở 100-shot, FT cao hơn LP từ 8.43 đến 25.04 điểm. Khoảng cách nhỏ nhất thuộc CrossHAR + standard, cho thấy feature đã có mức căn chỉnh liên miền tương đối tốt; khoảng cách lớn nhất thuộc Contrastive + CNN-Transformer, cho thấy classifier tuyến tính không thể tự sửa các sai lệch của embedding attention. Đây là bằng chứng định lượng để xem LP như phép đo chất lượng representation, thay vì chỉ xem LP là một cách huấn luyện rẻ hơn.

Khi tăng shot, khoảng cách không giảm đồng đều: Prototype + standard là 8.97 điểm ở 10-shot và 11.60 điểm ở 100-shot, còn CrossHAR + standard là 9.65 và 8.43 điểm ở 50/100-shot. Coverage thay đổi nên chưa thể kết luận feature luôn tốt lên theo shot.

Có 8 trường hợp LP vượt FT trong 156 cặp FT–LP matched. Đây là cảnh báo về overfitting hoặc dao động seed, không phải bằng chứng chắc chắn rằng đóng băng luôn tốt hơn cập nhật toàn mạng. Ví dụ, một số cấu hình Prototype CNN-Transformer có LP cao hơn FT ở 50-shot; các độ lệch chuẩn cao trong cùng nhóm cho thấy cần thêm seed và validation độc lập trước khi chọn protocol.

**Nhận xét:** FT thường thắng LP, đặc biệt khi backbone là CNN-Transformer. **Giải thích:** FT sửa được scale và hướng của feature theo miền đích, nhưng cũng mở thêm nhiều bậc tự do dẫn đến overfitting; LP ổn định hơn khi representation gốc đã có bias phù hợp.

**Kết luận mục 3.3:** FT là protocol mặc định để tối đa hóa điểm số trong dữ liệu hiện có, còn LP là kiểm định quan trọng về tính chuyển giao. Các trường hợp LP > FT cần được tái lập trước khi dùng làm cơ sở thiết kế hệ thống.

### 3.4. Phân tích data efficiency theo shot

Bảng 4 cho thấy learning curve của Prototype, phương pháp có coverage đầy đủ ở bốn mức shot, sau khi gộp hai backbone ở pocket-level. Chọn một phương pháp đầy đủ giúp tránh biến missingness của Contrastive và CrossHAR thành kết luận sai về saturation.

| Shot | FT standard | LP standard | FT CNN-Transformer | LP CNN-Transformer |
|---:|---:|---:|---:|---:|
| 10 | 65.24 | 56.27 | 65.63 | 55.13 |
| 20 | 74.42 | 61.07 | 73.37 | 60.94 |
| 50 | 79.79 | 65.96 | 77.59 | 65.64 |
| 100 | 82.32 | 70.72 | 81.13 | 69.09 |

Từ 10 lên 100 shot, Prototype standard FT tăng 17.08 điểm, còn LP tăng 14.45 điểm; CNN-Transformer FT tăng 15.50 điểm và LP tăng 13.96 điểm. Lợi ích lớn nhất xảy ra từ 10 lên 20 shot: standard FT tăng 9.18 điểm và CNN-Transformer FT tăng 7.74 điểm, vì vài mẫu đầu tiên đã giúp centroid và classifier bao phủ các mode hoạt động cơ bản. Từ 50 lên 100 shot, lợi ích giảm còn 2.53 điểm ở standard FT và 3.54 điểm ở CNN-Transformer FT, là dấu hiệu bão hòa tương đối.

Điểm bắt đầu giảm lợi nhuận nằm quanh 50 shot; tăng lên 100 shot vẫn có ích nhưng không còn tỷ lệ thuận với lượng dữ liệu. Capacity của backbone đã mô tả phần lớn pattern cục bộ, trong khi sai số còn lại đến từ domain shift giữa thiết bị và người dùng, vốn không thể loại bỏ chỉ bằng thêm mẫu cùng protocol. Với LP, saturation xuất hiện sớm hơn về mặt tuyệt đối vì head bị giới hạn bởi feature cố định.

Contrastive và CrossHAR chủ yếu có 50/100-shot, nên không đủ dữ liệu để kết luận khoảng cách ở 10/20 shot. Ở 100-shot standard LP, CrossHAR cao hơn Prototype 2.96 điểm; ở FT, Prototype cao hơn 0.21 điểm, cho thấy thứ hạng phụ thuộc protocol.

**Nhận xét:** Dữ liệu tăng từ 10 lên 20 shot đem lại lợi ích lớn nhất, còn 50 lên 100 shot cho lợi ích biên nhỏ hơn. **Giải thích:** giai đoạn đầu giúp ước lượng lớp và biến thiên người dùng, nhưng sau khi centroid đã ổn định, phần sai số còn lại bị chi phối bởi mismatch vật lý và các lớp có động học gần nhau.

**Kết luận mục 3.4:** 50-shot là điểm cân bằng thực dụng cho Prototype, còn 100-shot phù hợp khi ưu tiên điểm tối đa. Không thể đánh giá đầy đủ saturation của Contrastive và CrossHAR ở shot thấp do thiếu bản ghi matched.

## 4. Nút thắt vật lý: target `hhar_watch`

Target `hhar_watch` là phần quan trọng nhất để diễn giải giới hạn của thuật toán. Trung bình macro F1 của 76 bản ghi watch-level là 33.70%, so với 70.38% của 236 bản ghi pocket-level, tức giảm 36.68 điểm phần trăm. Mức chênh này lớn hơn nhiều so với khác biệt giữa các method hoặc backbone trong cùng một cụm, nên không nên quy nó đơn giản cho loss chưa tối ưu.

| Nhóm method | Pocket-level (%) | Target hhar_watch (%) | Suy giảm |
|---|---:|---:|---:|
| Prototype | 69.02 | 31.92 | −37.10 |
| Contrastive | 72.13 | 34.86 | −37.27 |
| CrossHAR | 72.75 | 38.20 | −34.55 |
| Tất cả bản ghi | 70.38 | 33.70 | −36.68 |

Bảng 5 cho thấy mọi method đều suy giảm trong khoảng 34.55–37.27 điểm. Vì độ suy giảm tương tự giữa Prototype, Contrastive và CrossHAR, việc thay loss không thể tự khôi phục thông tin động học mà cảm biến không ghi nhận giống nhau. CrossHAR có watch-level trung bình cao hơn, nhưng biên 3.34 điểm không đủ để phủ nhận bottleneck chung.

Cảm biến cổ tay đo bàn tay và cẳng tay, nên biên độ, tần số và nhiễu từ các động tác như vẫy tay có thể lớn. Cảm biến ở túi quần hoặc hông đo thân người với biên độ nhỏ hơn và nhịp thấp hơn, phản ánh trực tiếp hơn chu kỳ đi, đứng hoặc ngồi. Vì vậy Walking ở cổ tay có thể chứa dao động cánh tay, còn Walking ở hông chủ yếu thể hiện dao động thân và bước chân.

Hệ quả là embedding không đồng nhất giữa source và target dù nhãn giống nhau. Fine-tuning không thể suy ra đầy đủ chuyển động thân từ tín hiệu cổ tay bị trộn với hành động tay; cả ba method đều bị giới hạn bởi thiếu tương ứng vật lý.

Kiểm chứng định lượng củng cố giả thuyết physical bottleneck: mức sụt giảm của ba method gần nhau hơn nhiều so với khoảng cách FT–LP. Một số LP watch-level có std trên 10 điểm, cho thấy few-shot cổ tay vừa thấp vừa kém ổn định; vì vậy cần báo cáo cluster này độc lập.

**Nhận xét:** Sự suy giảm gần 35–40 điểm xuất hiện đồng thời ở mọi method, nên không có phương pháp nào loại bỏ được khác biệt cảm biến. **Giải thích:** đây là physical domain shift: vị trí đo thay đổi hệ quy chiếu, biên độ và thành phần chuyển động, khiến ánh xạ từ tín hiệu sang hoạt động không còn là cùng một bài toán quan sát.

**Kết luận mục 4:** `hhar_watch` là bottleneck vật lý chứ không chỉ là thất bại của thuật toán. Hướng phù hợp là thu thập thêm dữ liệu cổ tay có đa dạng hành động tay, hoặc dùng domain adaptation chuyên biệt có mô hình hóa sensor geometry và temporal alignment, thay vì chỉ tăng độ sâu backbone.

## 5. Phân tích ma trận nhầm lẫn

Phân tích dựa trên 312 ảnh `aggregated_confusion_matrix.png` và các ma trận JSON companion. Ba hướng nhầm lớn nhất là Walking → Upstairs, Upstairs → Walking và Downstairs → Walking; cả ba lớp đều chứa chuỗi bước lặp lại và chỉ khác độ dốc, cường độ cùng biến thiên theo người.

Trong cửa sổ ngắn, ba hoạt động có thể có cùng chu kỳ thấp và đỉnh do tiếp xúc bàn chân. Khác biệt về năng lượng hoặc trục trọng lực bị scale hoặc xoay khi đổi vị trí cảm biến, nên classifier dễ chọn nhãn “đi bộ” thay vì hướng cầu thang.

Lỗi bất đối xứng cho thấy mô hình ưu tiên lớp đi bộ khi bằng chứng về độ dốc yếu. Vì vậy, feature extraction nên bổ sung năng lượng theo trục, phổ tần số cục bộ và độ nghiêng tương đối, đồng thời theo dõi macro F1 theo lớp.

**Nhận xét:** Ma trận nhầm lẫn tập trung ở các hoạt động có cùng nhịp bước, không phải ở các lớp tĩnh hoàn toàn khác nhau. **Giải thích:** các lớp này chia sẻ tần số thấp và hình thái chu kỳ, trong khi khác biệt quyết định nằm ở biến thiên biên độ và thành phần trọng lực tinh tế.

**Kết luận mục 5:** Nút thắt phân loại lớp nằm cạnh nhau đòi hỏi đặc trưng động học tinh hơn và đánh giá per-class. Một loss liên miền chỉ giải quyết phân phối tổng thể nếu nó vẫn bảo toàn được các dấu hiệu nhỏ phân biệt Upstairs, Downstairs và Walking.

## 6. Thảo luận

### 6.1. Các insight tổng hợp

Thứ nhất, inductive bias của backbone có vai trò tương đương với objective: standard đạt 82.16% FT và 72.08% LP ở 100-shot, còn CNN-Transformer đạt 80.37% và 61.83%. Thứ hai, LP là phép đo representation chứ không chỉ là baseline rẻ; khoảng cách 25.04 điểm của Contrastive CNN-Transformer cho thấy accuracy FT có thể che giấu feature kém transferable.

Thứ ba, shot efficiency có diminishing returns: Prototype standard tăng 9.18 điểm từ 10 lên 20 shot nhưng chỉ tăng 2.53 điểm từ 50 lên 100 shot. Thứ tư, domain shift vật lý thống trị ở target cổ tay vì mọi method đều mất khoảng 35–40 điểm. Thứ năm, phương pháp tốt nhất phụ thuộc protocol: CrossHAR mạnh hơn ở LP standard, còn Prototype nhỉnh hơn ở FT standard.

### 6.2. Đối chiếu với literature

Mức pocket-level khoảng 70–82% và watch-level khoảng 30–40% phù hợp với xu hướng thường thấy trong literature HAR cross-domain: cảm biến gần thân người thường cho F1 cao hơn wrist-level. So sánh chỉ mang tính định hướng vì paper khác có số lớp và split khác nhau; xu hướng nhất quán là pocket dễ hơn wrist.

### 6.3. Hạn chế

Hạn chế là coverage không đồng đều: Contrastive và CrossHAR thiếu mức 10/20-shot; 55/312 bản ghi có std lớn hơn 5 điểm, chủ yếu ở watch-level và shot thấp. Mean ± std chỉ mô tả ổn định, chưa thay thế kiểm định thống kê.

Hạn chế thứ tư là chưa có metadata compute cho FLOPs, memory, latency hoặc thời gian huấn luyện, nên chưa thể khẳng định cấu hình accuracy cao nhất tối ưu triển khai.

### 6.4. Hướng phát triển

Một là hoàn thiện factorial design cho mọi method × backbone × shot × protocol, đặc biệt Contrastive và CrossHAR ở 10/20 shot. Hai là tăng seed và kiểm định paired trên cùng pair. Ba là xây dựng adaptation nhận thức cảm biến với alignment theo trục và frequency-aware normalization. Bốn là ghi nhận FLOPs, memory, latency và thời gian fine-tuning.

**Kết luận mục 6:** Thực nghiệm cho thấy độ phù hợp của inductive bias, chất lượng feature frozen và vị trí cảm biến quan trọng hơn việc tăng capacity thuần túy. Các kết luận về ranking cần được củng cố bằng coverage cân bằng, nhiều seed hơn và metadata compute.

## 7. Kết luận và đề xuất

### 7.1. Best Configuration


### 7.2. Best trade-off

Nếu chỉ xét accuracy trên các aggregate hiện có, Prototype + standard + 50-shot + full fine-tuning đạt 79.79%, thấp hơn 2.53 điểm so với 100-shot nhưng dùng một nửa số shot. Khoảng cách nhỏ ở giai đoạn này cho thấy 50-shot là lựa chọn hợp lý khi chi phí gán nhãn quan trọng hơn vài điểm F1. Tuy nhiên, compute cost chưa được ghi nhận, nên báo cáo không thể kết luận cấu hình này tối ưu về FLOPs hoặc latency; đây chỉ là trade-off accuracy/shot.

### 7.3. Đề xuất

Nên dùng standard + Prototype + FT làm baseline triển khai pocket-level, dùng LP như phép kiểm tra trước khi quyết định fine-tuning toàn mạng, và báo cáo riêng mọi target `hhar_watch`. Với watch-level, ưu tiên thêm dữ liệu đa dạng về chuyển động tay trước khi tăng model size. Cuối cùng, cần chuẩn hóa ma trận thí nghiệm, bổ sung compute profiling và đánh giá subject-wise để chuyển các quan sát hiện tại thành kết luận có ý nghĩa thống kê.

**Kết luận mục 7:** Prototype + standard + 100-shot + FT là cấu hình dẫn đầu về accuracy, trong khi 50-shot là điểm cân bằng dữ liệu thực dụng. Không nên gọi đây là cấu hình tối ưu triển khai cho đến khi có FLOPs, latency và memory.

## 8. Gợi ý visualization và mã Python

Code giả định DataFrame `df` được tạo từ các `summary_*.json` theo schema `method`, `backbone`, `source`, `target`, `shot`, `protocol`, `f1`, `std`, `seed_f1`.

### Hình 1 — Grouped Bar Chart: so sánh 100-shot

Loại biểu đồ: grouped bar chart. Trục X là method, trục Y là macro F1 (%), hue là backbone, error bar là std. Hình truyền tải chênh lệch giữa standard và CNN-Transformer ở FT/LP 100-shot; đặt ngay sau Mục 3.1.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plot = df[(~df.target.str.contains("hhar_watch")) &
          (df.shot == "100_shot")].copy()
plot["config"] = plot.method + " / " + plot.protocol
summary = (plot.groupby(["method", "backbone", "protocol"], as_index=False)
                .agg(f1=("f1", "mean"), std=("f1", "std")))
ax = sns.barplot(data=summary, x="method", y="f1", hue="backbone",
                 errorbar=None, palette="deep")
for i, row in summary.reset_index(drop=True).iterrows():
    x = i // 2 + (-0.2 if row.backbone == "standard" else 0.2)
    ax.errorbar(x, row.f1, yerr=row.std, fmt="none", color="black", capsize=3)
ax.set(xlabel="Method", ylabel="Macro F1 (%)",
       title="100-shot comparison with seed/domain variability")
plt.tight_layout()
plt.savefig("fig1_grouped_100shot.png", dpi=300)
```

### Hình 2 — Learning Curve

Loại biểu đồ: line chart. Trục X là shot, trục Y là macro F1 (%), mỗi đường là một method–backbone–protocol configuration; error band là độ lệch chuẩn. Hình cho thấy lợi ích biên giảm sau 50-shot và đặt tại Mục 3.4.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plot = df[~df.target.str.contains("hhar_watch")].copy()
plot["config"] = (plot.method + " / " + plot.backbone + " / " +
                  plot.protocol)
order = ["10_shot", "20_shot", "50_shot", "100_shot"]
plot = plot[plot.shot.isin(order)]
sns.lineplot(data=plot, x="shot", y="f1", hue="config",
             errorbar="sd", marker="o", sort=False)
plt.xticks(range(4), order)
plt.xlabel("Shot"); plt.ylabel("Macro F1 (%)")
plt.title("Data efficiency across available shots")
plt.tight_layout()
plt.savefig("fig2_learning_curve.png", dpi=300)
```

### Hình 3 — Transfer Heatmap

Loại biểu đồ: heatmap source × target. Hàng là source, cột là target, màu và annotation là macro F1; dùng best configuration có đủ pair. Hình làm nổi bật bất đối xứng transfer và đặt tại cuối Mục 3.4 hoặc đầu Mục 4; target watch phải được gắn nhãn riêng.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plot = df[~df.target.str.contains("hhar_watch")].copy()
best_key = (plot.groupby(["method", "backbone", "shot", "protocol"])
                 .f1.mean().idxmax())
best = plot[(plot.method == best_key[0]) &
            (plot.backbone == best_key[1]) &
            (plot.shot == best_key[2]) &
            (plot.protocol == best_key[3])]
pivot = best.pivot_table(index="source", columns="target",
                         values="f1", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis",
            vmin=0, vmax=100, linewidths=.5)
plt.xlabel("Target"); plt.ylabel("Source")
plt.title("Best-method pocket-level transfer")
plt.tight_layout()
plt.savefig("fig3_transfer_heatmap.png", dpi=300)
```

### Hình 4 — Boxplot seed stability

Loại biểu đồ: boxplot theo seed. Trục X là method, trục Y là F1 (%), hue là shot; đặt tại Mục 6.3 để kiểm tra trực quan các cấu hình có std cao. Nếu `seed_f1` thiếu, không thay bằng một giá trị tổng hợp; cần bỏ record đó và ghi rõ trong caption.

```python
import matplotlib.pyplot as plt
import seaborn as sns

seed = df.explode("seed_f1").copy()
seed = seed[~seed.target.str.contains("hhar_watch")]
seed = seed.dropna(subset=["seed_f1"])
sns.boxplot(data=seed, x="method", y="seed_f1", hue="shot",
            showfliers=True)
plt.xlabel("Method"); plt.ylabel("Seed macro F1 (%)")
plt.title("Seed-level stability on pocket-level targets")
plt.tight_layout()
plt.savefig("fig4_seed_boxplot.png", dpi=300)
```

### Hình 5 — Full Fine-tuning versus Linear Probing

Loại biểu đồ: grouped bar chart. Trục X là method–backbone, trục Y là F1 (%), hue là protocol và error bar là std; đặt ở Mục 3.3 để trực quan hóa feature quality. Biểu đồ cần giữ cùng source–target intersection khi tính hai protocol, nếu không khoảng cách FT–LP sẽ bị lệch bởi coverage.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plot = df[~df.target.str.contains("hhar_watch")].copy()
plot["config"] = plot.method + " / " + plot.backbone
summary = (plot.groupby(["config", "protocol"], as_index=False)
                .agg(f1=("f1", "mean"), std=("f1", "std")))
ax = sns.barplot(data=summary, x="config", y="f1", hue="protocol",
                 errorbar=None, capsize=.1)
for i, row in summary.iterrows():
    offset = -.2 if row.protocol == "full_finetuning" else .2
    ax.errorbar(i // 2 + offset, row.f1, yerr=row.std,
                fmt="none", color="black", capsize=3)
plt.xticks(rotation=25, ha="right")
plt.xlabel("Method / backbone"); plt.ylabel("Macro F1 (%)")
plt.title("Adaptation gain from full fine-tuning")
plt.tight_layout()
plt.savefig("fig5_ft_vs_lp.png", dpi=300)
```

**Ghi chú dữ liệu:** `N/A` là record thiếu và đã bị loại khỏi phép trung bình; không có bảng nào dùng toàn `N/A`.
Cấu hình có accuracy cao nhất là Prototype + standard + MotionSense → UCI HAR + 100-shot + full fine-tuning, với accuracy 91.30% và macro F1 91.40%. Cấu hình này thắng nhờ hình học centroid ổn định, bias cục bộ của 1D-CNN và khả năng fine-tuning theo miền đích; đây là kết quả trên một pair cụ thể.
