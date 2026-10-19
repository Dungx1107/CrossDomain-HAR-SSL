"""
===============================================================================
SCRIPT: BENCHMARK ANALYZER & VISUALIZER (TS-TCC VS PROTOTYPE) - V2
Vị trí: utils/analyze_few_shot_benchmark.py
Chức năng:
1. Đọc và nhận diện linh hoạt các mốc k (1, 5, 10, 20, 30, ...) từ thư mục checkpoints.
2. In bảng đối đầu chi tiết ra Terminal an toàn (không crash khi thiếu mốc k).
3. Thống kê tỷ số đối đầu trực diện: TS-TCC vs Prototype (thắng/thua/hòa).
4. Xuất biểu đồ so sánh F1 Score ra thư mục plots/few_shot_benchmark/.
===============================================================================
"""

import sys
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "cross_domain_fewshot"
PLOTS_DIR = PROJECT_ROOT / "plots" / "few_shot_benchmark"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PAIRS = [
    ("uci_har", "motionsense"),
    ("motionsense", "uci_har"),
    ("hhar_phone", "uci_har"),
    ("hhar_phone", "motionsense"),
    ("motionsense", "hhar_phone"),
    ("uci_har", "hhar_phone"),
    ("hhar_watch", "uci_har"),
    ("hhar_watch", "motionsense"),
    ("hhar_watch", "hhar_phone"),
    ("hhar_phone", "hhar_watch"),
    ("uci_har", "hhar_watch"),
    ("motionsense", "hhar_watch"),
]

PROTOCOLS = [
    ("linear_probing", "Linear Probing (LP)"),
    ("full_finetuning", "Full Fine-Tuning (FT)")
]


def parse_f1_value(f1_str: str) -> float:
    """Trích xuất giá trị Mean từ chuỗi 'Mean ± Std'."""
    try:
        if not f1_str or f1_str == "N/A":
            return -1.0
        return float(f1_str.split("±")[0].strip())
    except Exception:
        return -1.0


def load_method_results(method: str, backbone: str = "standard") -> dict:
    """Đọc dữ liệu few_shot_results.json của một phương pháp cụ thể."""
    results = {}
    base_dir = CHECKPOINTS_DIR / method / backbone
    for src, tgt in DEFAULT_PAIRS:
        pair_key = f"{src}_to_{tgt}"
        json_path = base_dir / pair_key / "few_shot_results.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    results[pair_key] = json.load(f)
            except Exception as e:
                print(f"⚠️ Lỗi đọc file {json_path}: {e}")
                results[pair_key] = {}
        else:
            results[pair_key] = {}
    return results


def discover_all_k_shots(tstcc_res: dict, proto_res: dict) -> list:
    """Tìm tất cả các mốc k có trong kết quả và sắp xếp theo thứ tự số tăng dần."""
    k_set = set()
    for res in [tstcc_res, proto_res]:
        for pair_key, pair_data in res.items():
            if isinstance(pair_data, dict):
                for k_key in pair_data.keys():
                    match = re.match(r"(\d+)_shot", k_key)
                    if match:
                        k_set.add(int(match.group(1)))

    if not k_set:
        return [1, 5, 10, 20]
    return sorted(list(k_set))


def print_comparison_table(tstcc_res: dict, proto_res: dict, proto_key: str, proto_title: str, k_shots: list):
    """In bảng so sánh chi tiết giữa TS-TCC và Prototype ở Terminal."""
    col_w = 22
    total_w = 34 + len(k_shots) * (col_w + 3)

    print("\n" + "=" * total_w)
    print(f"📊 BẢNG SO SÁNH MACRO F1 (%): TS-TCC vs PROTOTYPE | CHIẾN LƯỢC: {proto_title.upper()}")
    print("=" * total_w)

    header = f"{'Cặp Chuyển Giao':<32} | "
    for k in k_shots:
        header += f"{f'k={k} (TS-TCC / Proto)':<{col_w}} | "
    print(header)
    print("-" * total_w)

    head_to_head = {k: {"tstcc_win": 0, "proto_win": 0, "tie": 0} for k in k_shots}

    for src, tgt in DEFAULT_PAIRS:
        pair_key = f"{src}_to_{tgt}"
        pair_name = f"{src} -> {tgt}"
        row_str = f"{pair_name:<32} | "

        t_pair = tstcc_res.get(pair_key, {})
        p_pair = proto_res.get(pair_key, {})

        for k in k_shots:
            k_key = f"{k}_shot"
            t_f1_str = t_pair.get(k_key, {}).get(proto_key, {}).get("macro_f1", "N/A")
            p_f1_str = p_pair.get(k_key, {}).get(proto_key, {}).get("macro_f1", "N/A")

            t_val = parse_f1_value(t_f1_str)
            p_val = parse_f1_value(p_f1_str)

            if t_val >= 0 and p_val >= 0:
                if t_val > p_val:
                    head_to_head[k]["tstcc_win"] += 1
                    cell = f"{t_val:5.1f}* / {p_val:5.1f}"
                elif p_val > t_val:
                    head_to_head[k]["proto_win"] += 1
                    cell = f"{t_val:5.1f}  / {p_val:5.1f}*"
                else:
                    head_to_head[k]["tie"] += 1
                    cell = f"{t_val:5.1f}  / {p_val:5.1f} "
            elif t_val >= 0 and p_val < 0:
                cell = f"{t_val:5.1f}* /   N/A  "
            elif t_val < 0 and p_val >= 0:
                cell = f"  N/A   / {p_val:5.1f}*"
            else:
                cell = "     N/A     "

            row_str += f"{cell:<{col_w}} | "
        print(row_str)

    print("-" * total_w)
    win_row = f"{'🏆 TỔNG SỐ LẦN THẮNG (WIN)':<32} | "
    for k in k_shots:
        tw = head_to_head[k]["tstcc_win"]
        pw = head_to_head[k]["proto_win"]
        win_row += f"{f'TS:{tw:<2} vs PR:{pw:<2}':<{col_w}} | "
    print(win_row)
    print("=" * total_w)
    print("👉 Ký hiệu (*) đại diện cho phương pháp có điểm Macro F1 cao hơn ở cặp tương ứng.")
    return head_to_head


def plot_benchmark_charts(tstcc_res: dict, proto_res: dict, k_shots: list):
    """Vẽ biểu đồ đường tổng hợp trung bình và lưu ra file ảnh."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (proto_key, proto_title) in enumerate(PROTOCOLS):
        ax = axes[idx]
        tstcc_means = []
        proto_means = []
        valid_x = []
        valid_labels = []

        for k in k_shots:
            k_key = f"{k}_shot"
            t_vals = [
                parse_f1_value(tstcc_res.get(f'{s}_to_{t}', {}).get(k_key, {}).get(proto_key, {}).get('macro_f1', 'N/A'))
                for s, t in DEFAULT_PAIRS
            ]
            p_vals = [
                parse_f1_value(proto_res.get(f'{s}_to_{t}', {}).get(k_key, {}).get(proto_key, {}).get('macro_f1', 'N/A'))
                for s, t in DEFAULT_PAIRS
            ]

            t_valid = [v for v in t_vals if v >= 0]
            p_valid = [v for v in p_vals if v >= 0]

            if t_valid or p_valid:
                valid_x.append(len(valid_x))
                valid_labels.append(f"{k}-shot")
                tstcc_means.append(np.mean(t_valid) if t_valid else np.nan)
                proto_means.append(np.mean(p_valid) if p_valid else np.nan)

        if valid_x:
            ax.plot(valid_x, tstcc_means, marker='o', linewidth=2.5, color='#1f77b4', label='TS-TCC (Contrastive)')
            ax.plot(valid_x, proto_means, marker='s', linewidth=2.5, color='#ff7f0e', label='Prototype (Clustering)')

            for i, (tm, pm) in enumerate(zip(tstcc_means, proto_means)):
                if not np.isnan(tm):
                    ax.annotate(f"{tm:.1f}%", (valid_x[i], tm), textcoords="offset points", xytext=(0, 7), ha='center', fontweight='bold', color='#1f77b4')
                if not np.isnan(pm):
                    ax.annotate(f"{pm:.1f}%", (valid_x[i], pm), textcoords="offset points", xytext=(0, -14), ha='center', fontweight='bold', color='#ff7f0e')

            ax.set_title(f"Average Macro F1 across Available Pairs\n[{proto_title}]", fontsize=13, fontweight='bold')
            ax.set_xlabel("Few-Shot Value (k-shot)", fontsize=11)
            ax.set_ylabel("Macro F1 Score (%)", fontsize=11)
            ax.set_xticks(valid_x)
            ax.set_xticklabels(valid_labels)
            ax.set_ylim(15, 85)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    chart_path = PLOTS_DIR / "tstcc_vs_prototype_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n📈 Đã lưu biểu đồ so sánh tại: {chart_path}")


def main():
    print("🚀 BẮT ĐẦU ĐỌC DỮ LIỆU VÀ TỔNG HỢP BENCHMARK...")
    tstcc_res = load_method_results(method="tstcc", backbone="standard")
    proto_res = load_method_results(method="prototype", backbone="standard")

    k_shots = discover_all_k_shots(tstcc_res, proto_res)

    for proto_key, proto_title in PROTOCOLS:
        print_comparison_table(tstcc_res, proto_res, proto_key, proto_title, k_shots)

    plot_benchmark_charts(tstcc_res, proto_res, k_shots)
    print("🎉 HOÀN TẤT BÁO CÁO TỔNG HỢP!")


if __name__ == "__main__":
    main()