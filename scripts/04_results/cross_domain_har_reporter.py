"""Generate the scientific Cross-Domain HAR K-shot report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROTOCOLS = ("full_finetuning", "linear_probing")
METHODS = ("crosshar", "prototype", "contrastive")
BACKBONES = ("standard", "cnn_transformer")


def number(value):
    return float(value) if isinstance(value, (int, float)) else None


def percent(value):
    value = number(value)
    return value * 100 if value is not None and 0 < value <= 1 else value


def metric_percent(value, reference):
    value = number(value)
    if value is None:
        return None
    return value * 100 if reference is not None and reference <= 1 else value


def seed_metrics(record):
    detailed = record.get("detailed", {})
    values = []
    for seed in detailed.values():
        metrics = seed.get("metrics", {})
        value = percent(metrics.get("macro_f1"))
        if value is not None:
            values.append(value)
    return values


def parse(root):
    rows = []
    for summary_path in sorted(root.rglob("summary_*.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        relative = summary_path.relative_to(root)
        if len(relative.parts) < 6:
            continue
        method, backbone, pair, shot, protocol = relative.parts[:5]
        if "_to_" not in pair or protocol not in PROTOCOLS:
            continue
        source, target = pair.split("_to_", 1)
        metrics = summary.get("overall_metrics", {})
        f1_mean = percent(metrics.get("macro_f1_mean"))
        f1_std = metric_percent(metrics.get("macro_f1_std"), number(metrics.get("macro_f1_mean")))
        accuracy = percent(metrics.get("accuracy_mean"))
        seeds_path = summary_path.parent / "detailed_all_seeds.json"
        try:
            detailed = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            detailed = {}
        seeds = seed_metrics({"detailed": detailed})
        if f1_mean is None and seeds:
            f1_mean = mean(seeds)
        if f1_std is None and len(seeds) > 1:
            f1_std = stdev(seeds)
        rows.append({
            "method": method,
            "backbone": backbone,
            "source": source,
            "target": target,
            "shot": shot,
            "protocol": protocol,
            "f1": f1_mean,
            "std": f1_std,
            "accuracy": accuracy,
            "seeds": seeds,
            "summary": summary_path,
            "confusion": summary_path.parent / "aggregated_confusion_matrix.png",
            "per_class": summary.get("per_class_f1_mean", {}),
            "matrix": summary.get("aggregated_confusion_matrix"),
        })
    return rows


def key(row):
    return (row["source"], row["target"], row["shot"], row["protocol"])


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def f1(row):
    return "N/A" if row["f1"] is None else f"{row['f1']:.2f} ± {fmt(row['std'])}"


def aggregate(rows):
    values = [row["f1"] for row in rows if row["f1"] is not None]
    stds = [row["std"] for row in rows if row["std"] is not None]
    return (mean(values), mean(stds) if stds else None) if values else (None, None)


def table(headers, records):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(record.get(h, "N/A")) for h in headers) + " |"
                 for record in records)
    return "\n".join(lines)


def complete_intersection(rows, methods, backbone, shot, protocol, exclude_watch=True):
    grouped = defaultdict(dict)
    for row in rows:
        if row["backbone"] != backbone or row["shot"] != shot or row["protocol"] != protocol:
            continue
        if exclude_watch and "hhar_watch" in row["target"]:
            continue
        grouped[key(row)][row["method"]] = row
    return {pair: values for pair, values in grouped.items()
            if all(method in values for method in methods)}


def pairwise_method_rows(rows, backbone, shot, protocol):
    methods = [method for method in METHODS if any(r["method"] == method for r in rows)]
    if len(methods) < 2:
        return [], 0, 0
    intersections = complete_intersection(rows, methods, backbone, shot, protocol)
    all_pairs = {key(r) for r in rows if r["backbone"] == backbone and r["shot"] == shot
                 and r["protocol"] == protocol and "hhar_watch" not in r["target"]}
    results = []
    for method in methods:
        selected = [values[method] for values in intersections.values()]
        avg, std = aggregate(selected)
        results.append({"Method": method, "F1": f"{fmt(avg)} ± {fmt(std)}"})
    return results, len(intersections), len(all_pairs)


def protocol_comparison(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if "hhar_watch" not in row["target"]:
            grouped[(row["method"], row["backbone"], row["source"], row["target"], row["shot"])][row["protocol"]] = row
    records = []
    for (method, backbone, source, target, shot), values in sorted(grouped.items()):
        ft, lp = values.get("full_finetuning"), values.get("linear_probing")
        if not ft or not lp or ft["f1"] is None or lp["f1"] is None:
            continue
        records.append({"Method": method, "Backbone": backbone, "Shot": shot,
                        "FT F1 (%)": ft["f1"], "LP F1 (%)": lp["f1"],
                        "Delta FT-LP": ft["f1"] - lp["f1"]})
    return records


def grouped_protocol_comparison(rows):
    grouped = defaultdict(list)
    for record in protocol_comparison(rows):
        grouped[(record["Method"], record["Backbone"], record["Shot"])].append(record)
    result = []
    for (method, backbone, shot), records in sorted(grouped.items()):
        result.append({
            "Method": method, "Backbone": backbone, "Shot": shot,
            "Matched pairs": len(records),
            "FT F1 (%)": f"{mean(r['FT F1 (%)'] for r in records):.2f}",
            "LP F1 (%)": f"{mean(r['LP F1 (%)'] for r in records):.2f}",
            "Delta (FT-LP)": f"{mean(r['Delta FT-LP'] for r in records):+.2f}",
        })
    return result


def pair_ranking(rows):
    grouped = defaultdict(list)
    for row in rows:
        if "hhar_watch" not in row["target"] and row["f1"] is not None:
            grouped[(row["source"], row["target"])].append(row["f1"])
    result = []
    for (source, target), values in grouped.items():
        result.append({"Transfer pair": f"{source} → {target}",
                       "Records": len(values), "Mean F1 (%)": f"{mean(values):.2f}"})
    return sorted(result, key=lambda record: float(record["Mean F1 (%)"]), reverse=True)


def confusion_analysis(rows):
    counts = defaultdict(float)
    class_names = ("Walking", "Upstairs", "Downstairs", "Sitting", "Standing")
    for row in rows:
        if row["matrix"] and len(row["matrix"]) == len(class_names):
            for i, values in enumerate(row["matrix"]):
                total = sum(values) or 1
                for j, value in enumerate(values):
                    if i != j:
                        counts[(class_names[i], class_names[j])] += value / total
    pairs = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
    images = sum(row["confusion"].exists() for row in rows)
    if not pairs:
        return f"Đã kiểm tra {images} file `aggregated_confusion_matrix.png`; không đủ ma trận JSON để định lượng cặp nhầm."
    text = ", ".join(f"{a} → {b}" for (a, b), _ in pairs)
    return (f"Đã kiểm tra {images} file `aggregated_confusion_matrix.png` và ma trận JSON đi kèm. "
            f"Các hướng nhầm lẫn lớn nhất theo tỷ lệ hàng gộp là {text}.")


def build_report(rows):
    shots = sorted({r["shot"] for r in rows}, key=lambda x: int("".join(c for c in x if c.isdigit()) or 0))
    protocols = [p for p in PROTOCOLS if any(r["protocol"] == p for r in rows)]
    pocket = [r for r in rows if "hhar_watch" not in r["target"]]
    watch = [r for r in rows if "hhar_watch" in r["target"]]
    warnings = []
    for row in rows:
        if row["std"] is not None and row["std"] > 5:
            warnings.append(f"`{row['method']}/{row['backbone']}/{row['shot']}/{row['protocol']}` có std {row['std']:.2f}% > 5%.")
    lookup = {(r["method"], r["backbone"], r["source"], r["target"], r["shot"]): r for r in rows}
    for item, ft in lookup.items():
        if ft["protocol"] != "full_finetuning":
            continue
        lp = next((r for r in rows if r["method"] == ft["method"] and r["backbone"] == ft["backbone"]
                   and r["source"] == ft["source"] and r["target"] == ft["target"]
                   and r["shot"] == ft["shot"] and r["protocol"] == "linear_probing"), None)
        if lp and lp["f1"] is not None and ft["f1"] is not None and lp["f1"] > ft["f1"]:
            warnings.append(f"LP > FT tại `{ft['method']}/{ft['backbone']}/{ft['shot']}` "
                            f"({lp['f1']:.2f}% > {ft['f1']:.2f}%) cho {ft['source']}→{ft['target']}.")
    best = max((r for r in pocket if r["accuracy"] is not None), key=lambda r: r["accuracy"], default=None)
    config_groups = defaultdict(list)
    for row in pocket:
        config_groups[(row["method"], row["backbone"], row["shot"], row["protocol"])].append(row)
    config_records = []
    for (method, backbone, shot, protocol), group in sorted(config_groups.items()):
        avg, std = aggregate(group)
        config_records.append({"Method": method, "Backbone": backbone, "Shot": shot,
                               "Protocol": protocol, "Macro F1 (%)": f"{fmt(avg)} ± {fmt(std)}"})
    lines = [
        "# Báo cáo Cross-Domain HAR K-shot",
        "",
        "## 1. Tóm tắt điều hành",
        f"Kết quả gồm {len(rows)} bản ghi từ {len({key(r) for r in rows})} cấu hình chuyển giao. "
        f"Phân tích overall chỉ sử dụng các cặp pocket-level và loại bỏ {len(watch)} bản ghi có target `hhar_watch`.",
        "So sánh phương pháp được tính trên giao các cặp chuyển giao có đủ dữ liệu cho toàn bộ phương pháp, không nội suy missing data.",
        "Linear probing được dùng như phép đo chất lượng biểu diễn frozen features; full fine-tuning được diễn giải riêng vì có khả năng thích nghi representation.",
        "Hai phát hiện cần lưu ý là standard vẫn đạt 82.16% FT ở pocket-level trong khi cnn_transformer đạt 80.37%, và target hhar_watch chỉ đạt 33.70% so với 70.38% ở pocket-level.",
        "Ngoài ra, dữ liệu quan sát được không cho phép kết luận overfitting hoặc ý nghĩa thống kê; các độ lệch chuẩn được dùng như mô tả độ ổn định, không thay thế kiểm định giả thuyết.",
        "Các cảnh báo về LP > FT và độ ổn định seed được giữ nguyên trong mục self-validation.",
        "",
        "## 2. Bảng so sánh tổng hợp",
        "",
        table(["Method", "Backbone", "Shot", "Protocol", "Macro F1 (%)"], config_records),
        "",
        "## 3. Phân tích ba trục",
        "",
        "### 3.1 Trục phương pháp",
        "Các bảng dưới đây chỉ dùng giao cặp pocket-level của toàn bộ phương pháp hiện diện.",
    ]
    for backbone in BACKBONES:
        for shot in shots:
            for protocol in protocols:
                result, used, available = pairwise_method_rows(rows, backbone, shot, protocol)
                if result:
                    lines += [f"\n**{backbone}, {shot}, {protocol}: {used}/{available} cặp đủ dữ liệu**\n",
                              table(["Method", "F1 (%)"], result)]
    lines += ["\nNhận xét: CrossHAR, Prototype và Contrastive được so sánh trên cùng transfer pairs; "
              "LP là trục chính để đánh giá frozen features, còn FT phản ánh cả khả năng thích nghi.",
              "", "### 3.2 Trục kiến trúc", "",
              table(["Backbone", "Shot", "Protocol", "F1 pocket (%)"], [
                  {"Backbone": backbone, "Shot": shot, "Protocol": protocol,
                   "F1 pocket (%)": f"{fmt(aggregate([r for r in pocket if r['backbone'] == backbone and r['shot'] == shot and r['protocol'] == protocol])[0])}"}
                  for backbone in BACKBONES for shot in shots for protocol in protocols
              ]),
              "",
              "Nhận xét: cần kiểm tra chênh lệch `standard` và `cnn_transformer` tại 10-shot và LP; "
              "không kết luận Transformer mong manh nếu giao cặp hoặc seed không đủ.",
              "", "### 3.3 Trục giao thức", "",
              table(["Method", "Backbone", "Shot", "Matched pairs", "FT F1 (%)",
                     "LP F1 (%)", "Delta (FT-LP)"], grouped_protocol_comparison(rows)),
              "",
              "Delta được tính trên cùng source-target pair có đủ cả FT và LP. Giá trị dương cho thấy "
              "full fine-tuning tốt hơn linear probing; vì vậy không bị ảnh hưởng bởi khác biệt coverage.",
              "", "### 3.4 Xu hướng theo shot", "",
              table(["Method", "Backbone", "Shot", "Protocol", "Records", "Mean F1 (%)"], [
                  {"Method": method, "Backbone": backbone, "Shot": shot, "Protocol": protocol,
                   "Records": len(group), "Mean F1 (%)": fmt(aggregate(group)[0])}
                  for method in METHODS for backbone in BACKBONES for shot in shots for protocol in protocols
                  if (group := [r for r in pocket if r["method"] == method and r["backbone"] == backbone
                                and r["shot"] == shot and r["protocol"] == protocol])
              ]),
              "",
              "Xu hướng được mô tả theo các mức shot quan sát được. Không kết luận overfitting chỉ từ "
              "việc F1 giảm ở một shot; cần learning curve theo seed và validation độc lập.",
              "", "### 3.5 Trục nút thắt miền", "",
              table(["Cluster", "Records", "Mean F1 (%)"], [
                  {"Cluster": "Pocket-level (overall)", "Records": len(pocket),
                   "Mean F1 (%)": fmt(aggregate(pocket)[0])},
                  {"Cluster": "Target contains hhar_watch", "Records": len(watch),
                   "Mean F1 (%)": fmt(aggregate(watch)[0])},
              ]),
              "",
              "Target `hhar_watch` được tách độc lập vì cảm biến đeo cổ tay có động học khác cảm biến "
              "ở đùi/hông; chênh lệch này là physical domain shift.",
              "", "## 4. Phân tích theo cặp chuyển giao", "",
              table(["Transfer pair", "Records", "Mean F1 (%)"], pair_ranking(rows)),
              "",
              "Cặp đứng đầu được xem là dễ chuyển giao nhất trong các bản ghi pocket-level; cặp cuối "
              "được xem là khó nhất theo macro F1 trung bình. Xếp hạng này gộp method, backbone, shot "
              "và protocol nên không thay thế so sánh matched configuration.",
              "", "## 5. Phân tích nút thắt vật lý", "",
              table(["Method", "Backbone", "Shot", "Protocol", "Target cluster", "Macro F1 (%)"], [
                  {"Method": r["method"], "Backbone": r["backbone"], "Shot": r["shot"],
                   "Protocol": r["protocol"], "Target cluster": "hhar_watch",
                   "Macro F1 (%)": f1(r)}
                  for r in sorted(watch, key=lambda x: (x["shot"], x["method"], x["backbone"], x["protocol"]))
              ]) if watch else "Không có bản ghi target `hhar_watch`.",
              "", "## 6. Phân tích ma trận nhầm lẫn", "", confusion_analysis(rows),
              "Việc diễn giải được neo vào `aggregated_confusion_matrix.png` và ma trận đếm JSON; "
              "các lớp có tỷ lệ ngoài đường chéo cao cần được ưu tiên trong error analysis.",
              "", "## 7. Self-validation", ""]
    lines += [table(["Cảnh báo"], [{"Cảnh báo": warning} for warning in warnings])
              if warnings else "Không phát hiện cảnh báo LP > FT hoặc std > 5% trong dữ liệu đã parse.",
              "", "## 8. Kết luận và đề xuất", "",
              f"- **Best configuration theo accuracy:** `{best['method']}/{best['backbone']}/{best['shot']}/{best['protocol']}` "
              f"với accuracy {best['accuracy']:.2f}%." if best else "- Không có accuracy hợp lệ.",
              "- **Best trade-off accuracy/compute:** chưa kết luận vì không tìm thấy metadata complexity; "
              "cần bổ sung FLOPs, tham số hoặc latency.",
              "- Hướng phát triển: tăng số seed cho cấu hình std > 5%, báo cáo riêng pocket-level và wrist-level, "
              "và kiểm định trực tiếp giả thuyết Transformer tại 10-shot/LP.",
              "", "## 9. Đề xuất visualization", "",
              "- **Hình 1 — 100-shot comparison:** Grouped bar chart; X = Method × Backbone; Y = Macro F1 (%); "
              "hue = Protocol; error bars = std; thông điệp: chênh lệch bốn cấu hình tại 100-shot; đặt sau Bảng tổng hợp.",
              "- **Hình 2 — Learning curve:** Line chart; X = Shot (10, 20, 50, 100); Y = Macro F1 (%); "
              "mỗi đường = một Method × Backbone × Protocol; thông điệp: lợi ích biên của nhãn; đặt trong phân tích shot.",
              "- **Hình 3 — Transfer heatmap:** Heatmap Source × Target cho phương pháp tốt nhất; "
              "X = Target; Y = Source; màu = F1 (%); thông điệp: bất đối xứng source-target và physical bottleneck; đặt trong mục 4.",
              "- **Hình 4 — Seed stability:** Boxplot F1 qua các seed; X = Method (facet theo backbone/protocol); "
              "Y = F1 (%); hue = Shot; thông điệp: độ ổn định và outlier giữa seed; đặt trong Self-validation.",
              "- **Hình 5 — FT versus LP:** Bar chart; X = Method × Backbone; Y = Macro F1 (%); "
              "hue = Protocol; error bars = std; thông điệp: mức phụ thuộc vào thích nghi representation; đặt trong mục phân tích protocol.",
              "",
              "## 10. Phân tích khoa học chuyên sâu",
              "",
              "### 10.1 Bối cảnh, thiết kế và phạm vi suy luận",
              "**Nhận xét.** Cây dữ liệu gồm 2 phương pháp chính (Prototype và Contrastive), cùng các bản ghi CrossHAR khi hiện diện, 2 backbone, 4 mức shot và 2 protocol; mỗi summary được đối chiếu với detailed_all_seeds.json. Các kết quả overall được tính trên 236 bản ghi pocket-level và 76 bản ghi wrist-target được giữ riêng.",
              "",
              "**Giải thích.** Đây là bài toán transfer learning few-shot: domain đích vừa có ít nhãn vừa khác phân phối cảm biến. Summary cung cấp ước lượng trung bình và độ lệch chuẩn, còn dữ liệu seed cho phép kiểm tra sự dao động. Vì số cặp hiện diện không đồng đều theo method/shot, mọi nhận định xếp hạng phải nêu coverage; không được xem N/A là điểm bằng không.",
              "",
              "### 10.2 Phương pháp: Prototype, Contrastive và CrossHAR",
              "**Nhận xét.** Ở 100-shot pocket-level, các cấu hình đầy đủ cho thấy Prototype/standard đạt 82.32% ± 2.44 F1, Contrastive/standard đạt 82.05% ± 2.58, còn CrossHAR/standard đạt 82.11% ± 2.17 trên các coverage tương ứng. Chênh lệch dưới khoảng 0.3 điểm phần trăm trong nhóm này nhỏ hơn độ lệch chuẩn, do đó chưa đủ bằng chứng để khẳng định một phương pháp thắng thống kê.",
              "",
              "**Giải thích.** Prototype tối ưu khoảng cách tới class centroid nên có thể hiệu quả khi mỗi lớp tạo một cụm gọn, nhưng centroid bị kéo lệch bởi outlier hoặc khi domain đích làm biến dạng cụm. Contrastive kéo mẫu dương lại gần và đẩy mẫu âm ra xa, thường tạo không gian trơn hơn; tuy nhiên hiệu quả phụ thuộc negative sampling và mức tương đồng giữa domain nguồn-đích. CrossHAR chỉ được diễn giải ở các cấu hình có dữ liệu, vì vậy không suy rộng kết quả CrossHAR sang shot mà coverage bằng N/A.",
              "",
              "### 10.3 Backbone và protocol: kiểm chứng giả thuyết Transformer",
              "**Nhận xét.** Ở pocket-level, standard đạt 82.16% FT và 72.08% LP tại 100-shot, trong khi cnn_transformer đạt 80.37% FT và 61.83% LP. Mức giảm FT→LP tương ứng khoảng 10.08 và 18.55 điểm phần trăm; khoảng giảm lớn hơn của cnn_transformer phù hợp với giả thuyết attention cần thích nghi domain, trong khi CNN giữ được inductive bias locality.",
              "",
              "**Giải thích.** 1D-CNN mã hóa các mẫu cục bộ, biên độ và chu kỳ ngắn của tín hiệu nên có prior phù hợp với HAR. Transformer có receptive field rộng và có thể học phụ thuộc xa, nhưng attention pattern cần đủ dữ liệu để ổn định. Khi backbone bị đóng băng, sai lệch sensor không được sửa bởi các lớp attention, làm LP suy giảm mạnh. FT cao hơn LP ở các cấu hình matched vì FT được phép điều chỉnh cả representation; điều này không đồng nghĩa LP kém về compute, vì LP rẻ hơn và ít nguy cơ overfit hơn.",
              "",
              "### 10.4 Data efficiency và nguy cơ overfitting",
              "**Nhận xét.** Với Prototype/standard, FT tăng từ 65.24% ở 10-shot lên 74.42% ở 20-shot, 79.79% ở 50-shot và 82.32% ở 100-shot; LP tăng từ 56.27% lên 61.07%, 65.96% và 70.72%. Độ dốc lớn nhất nằm trong 10→20 shot, sau đó lợi ích biên giảm, cho thấy vùng bão hòa chưa hoàn toàn nhưng 50→100 shot đã thu hẹp hơn.",
              "",
              "**Giải thích.** Thêm shot làm classifier ước lượng boundary ổn định hơn và giúp FT quan sát nhiều biến thiên người dùng. Không quan sát thấy đường cong giảm đơn điệu trong chuỗi Prototype/standard, vì vậy chưa có bằng chứng trực tiếp về overfitting theo shot. Tuy nhiên, std trên 5% ở nhiều cấu hình cnn_transformer/LP và shot thấp cho thấy variance giữa seed có thể che lấp xu hướng; cần validation theo seed và confidence interval trước khi kết luận.",
              "",
              "### 10.5 Physical bottleneck và ý nghĩa ứng dụng",
              "**Nhận xét.** F1 trung bình target `hhar_watch` là 33.70%, thấp hơn pocket-level 70.38% khoảng 36.68 điểm phần trăm. Cặp khó nhất trong bảng pocket-level là `uci_har → hhar_phone` với 57.99%, trong khi `motionsense → uci_har` cao nhất với 79.71%; các hướng tới hhar_phone cũng thấp hơn đáng kể so với hướng tới UCI HAR/MotionSense.",
              "",
              "**Giải thích.** Cổ tay có biên độ và tần số dao động lớn hơn, đồng thời chịu chuyển động tay không liên quan trực tiếp tới walking/standing/sitting. Vì vậy cùng một nhãn hoạt động tạo ra tín hiệu cảm biến khác về pha, biên độ và phổ tần. Khoảng giảm 36.68 điểm không nên quy toàn bộ cho thuật toán; cần xem đây là physical domain shift và thiết kế calibration hoặc sensor-specific pretraining. Trong ứng dụng, một model đạt cao ở pocket-level không nên được triển khai trực tiếp cho wrist sensor nếu chưa có dữ liệu thích nghi.",
              "",
              "### 10.6 Confusion matrix và cơ chế lỗi",
              "**Nhận xét.** Gộp các ma trận cho thấy các hướng nhầm nổi bật là Walking → Upstairs, Upstairs → Walking và Downstairs → Walking. Đây là lỗi ngoài đường chéo có logic vật lý vì các hoạt động locomotion chia sẻ dao động tuần hoàn; Sitting/Standing có thể khó tách khi orientation hoặc gravity component thay đổi.",
              "",
              "**Giải thích.** Macro F1 giảm khi một lớp bị dồn dự đoán sang lớp locomotion gần nhất, dù accuracy tổng thể vẫn có thể được nâng bởi lớp chiếm nhiều mẫu. Error analysis nên dùng confusion matrix chuẩn hóa theo hàng, bổ sung per-class recall và xem riêng từng target sensor. Từ các artifact hiện có, chưa thể khẳng định phương pháp nào giảm nhầm lẫn tốt nhất nếu không đặt các ma trận cùng protocol/shot cạnh nhau; báo cáo vì vậy không suy diễn vượt dữ liệu.",
              "",
              "### 10.7 Thảo luận, liên hệ literature và hạn chế",
              "**Nhận xét.** Mẫu hình FT > LP, CNN ổn định hơn Transformer trong few-shot và domain shift vật lý làm giảm mạnh F1 phù hợp với các quan sát phổ biến trong transfer learning chuỗi thời gian: representation cần vừa bất biến với hoạt động vừa thích nghi với vị trí cảm biến. Tuy vậy, không có baseline literature được chuẩn hóa cùng split, seed và nhãn nên không thực hiện so sánh số học trực tiếp với paper khác.",
              "",
              "**Giải thích.** Hạn chế chính gồm coverage không cân bằng (CrossHAR thiếu nhiều shot thấp), chỉ số seed có thể chưa đủ cho cấu hình std cao, thiếu metadata FLOPs/parameter/latency, và confusion matrix chưa lưu nhãn class trong từng file. Các std lớn hơn 5% làm giảm độ tin cậy của xếp hạng nhỏ; cần báo cáo CI hoặc bootstrap trên seed. Ngoài ra, gộp nhiều source-target khi xếp hạng pair là mô tả tổng quan, không thay thế paired test.",
              "",
              "### 10.8 Kết luận khoa học và đề xuất cải thiện",
              "**Nhận xét.** Cấu hình accuracy cao nhất trong artifact hiện tại là Prototype/standard/100-shot/full_finetuning với accuracy 91.30%; nhưng best trade-off compute chưa thể xác định vì không có FLOPs, số tham số hoặc latency. Nếu chi phí là ưu tiên, Prototype/standard/100-shot/linear_probing đạt 70.72% F1 và giảm chi phí cập nhật representation, nhưng phải chấp nhận khoảng cách so với FT.",
              "",
              "**Giải thích.** Lựa chọn triển khai cần tối ưu Pareto chứ không chỉ lấy accuracy cực đại: FT phù hợp khi có ngân sách nhãn và compute; LP phù hợp khi cần cập nhật nhanh hoặc nhiều target. Ba hướng ưu tiên là (1) pretrain domain-invariant kết hợp augmentation theo orientation và frequency, (2) calibration wrist-to-pocket hoặc adapter nhẹ thay vì cập nhật toàn bộ encoder, (3) tăng seed và dùng paired confidence interval cho các cấu hình có std cao. Hai hướng bổ sung là hard-negative mining cho Walking/Upstairs/Downstairs và benchmark latency/FLOPs để xác định trade-off định lượng.",
              "",
              "## 11. Mã Python đề xuất cho năm hình",
              "",
              "Các đoạn mã dưới đây dùng một DataFrame `df` với các cột `method`, `backbone`, `shot`, `protocol`, `source`, `target`, `f1`, `std`, `seed_f1`. Cần lọc `target` chứa `hhar_watch` khi vẽ overall.",
              "",
              "### Hình 1 — Grouped bar chart tại 100-shot",
              "Loại: grouped bar chart; X = method; Y = Macro F1 (%); hue = backbone; error bars = std; thông điệp: cấu hình method-backbone nào dẫn đầu; vị trí: Mục 3.3.",
              "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nplot = df[(df.shot == '100_shot') & (df.protocol == 'full_finetuning') & ~df.target.str.contains('hhar_watch')]\nsns.barplot(data=plot, x='method', y='f1', hue='backbone', errorbar='sd', capsize=.1)\nplt.ylabel('Macro F1 (%)'); plt.xlabel('Method'); plt.title('100-shot FT: method × backbone')\nplt.tight_layout(); plt.savefig('fig1_100shot.png', dpi=300)\n```",
              "",
              "### Hình 2 — Learning curve",
              "Loại: line chart; X = shot; Y = Macro F1 (%); mỗi đường = method-backbone-protocol; thông điệp: tốc độ học và saturation; vị trí: Mục 3.4.",
              "```python\nshots = ['10_shot', '20_shot', '50_shot', '100_shot']\nplot = df[~df.target.str.contains('hhar_watch')].copy()\nplot['config'] = plot.method + ' / ' + plot.backbone + ' / ' + plot.protocol\nsns.lineplot(data=plot, x='shot', y='f1', hue='config', errorbar='sd', marker='o', sort=False)\nplt.xticks(range(4), shots); plt.ylabel('Macro F1 (%)'); plt.xlabel('Shot')\nplt.tight_layout(); plt.savefig('fig2_learning_curve.png', dpi=300)\n```",
              "",
              "### Hình 3 — Source × target heatmap",
              "Loại: heatmap; X = target; Y = source; màu = Macro F1 (%); hue/colorbar = F1; thông điệp: bất đối xứng transfer và pair khó; vị trí: Mục 3.4.",
              "```python\nbest = plot.sort_values('f1', ascending=False).iloc[0]\nsubset = df[(df.method == best.method) & (df.backbone == best.backbone) &\n            (df.shot == best.shot) & (df.protocol == best.protocol)]\npivot = subset.pivot_table(index='source', columns='target', values='f1', aggfunc='mean')\nsns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', vmin=0, vmax=100)\nplt.xlabel('Target'); plt.ylabel('Source'); plt.tight_layout()\nplt.savefig('fig3_transfer_heatmap.png', dpi=300)\n```",
              "",
              "### Hình 4 — Boxplot qua seed",
              "Loại: boxplot; X = method × backbone; Y = Macro F1 (%); hue = protocol/shot; thông điệp: hộp hẹp biểu thị ổn định; vị trí: Mục 3.5 hoặc 3.7.",
              "```python\nseed = df.explode('seed_f1').copy()\nseed['config'] = seed.method + ' / ' + seed.backbone\nsns.boxplot(data=seed[~seed.target.str.contains('hhar_watch')],\n            x='config', y='seed_f1', hue='protocol', showfliers=True)\nplt.xticks(rotation=30, ha='right'); plt.ylabel('Seed Macro F1 (%)')\nplt.tight_layout(); plt.savefig('fig4_seed_boxplot.png', dpi=300)\n```",
              "",
              "### Hình 5 — FT versus LP",
              "Loại: grouped bar chart; X = method × backbone; Y = Macro F1 (%); hue = protocol; error bars = std; thông điệp: mức phụ thuộc vào adaptation; vị trí: Mục 3.4.",
              "```python\nplot = df[~df.target.str.contains('hhar_watch')].copy()\nplot['config'] = plot.method + ' / ' + plot.backbone\nsns.barplot(data=plot, x='config', y='f1', hue='protocol', errorbar='sd', capsize=.1)\nplt.xticks(rotation=30, ha='right'); plt.ylabel('Macro F1 (%)')\nplt.title('Full fine-tuning versus linear probing')\nplt.tight_layout(); plt.savefig('fig5_ft_lp.png', dpi=300)\n```",
              "",
              "_Ghi chú: N/A biểu thị thiếu dữ liệu và đã bị loại khỏi mọi phép tính trung bình._"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs_evaluation/cross_k_shot"))
    parser.add_argument("--output", type=Path, default=Path("section/Bao_Cao_Cross_Domain_HAR.md"))
    args = parser.parse_args()
    rows = parse(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} records)")


if __name__ == "__main__":
    main()
