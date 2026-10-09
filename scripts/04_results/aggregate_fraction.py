import os
import json
import pandas as pd

# Thiết lập đường dẫn động
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FRACTION_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation", "cross_fraction")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation")


def extract_metrics(res):
    """Trích xuất metrics từ summary json (đã ở thang 100)"""
    if "overall_metrics" in res:
        metrics = res["overall_metrics"]
        f1_m = metrics.get("macro_f1_mean", 0.0)
        f1_s = metrics.get("macro_f1_std", 0.0)
        acc_m = metrics.get("accuracy_mean", 0.0)
        acc_s = metrics.get("accuracy_std", 0.0)

        # Phòng trường hợp file cũ chưa nhân 100
        if 0 < f1_m <= 1.0:
            f1_m *= 100
            f1_s *= 100
        if 0 < acc_m <= 1.0:
            acc_m *= 100
            acc_s *= 100

        return f1_m, f1_s, acc_m, acc_s
    return 0.0, 0.0, 0.0, 0.0


def parse_fraction_results():
    data = []
    if not os.path.exists(FRACTION_DIR):
        print(f"Không tìm thấy thư mục: {FRACTION_DIR}")
        return pd.DataFrame()

    for method in os.listdir(FRACTION_DIR):
        method_path = os.path.join(FRACTION_DIR, method)
        if not os.path.isdir(method_path): continue

        for backbone in os.listdir(method_path):
            backbone_path = os.path.join(method_path, backbone)
            if not os.path.isdir(backbone_path): continue

            for pair in os.listdir(backbone_path):
                pair_path = os.path.join(backbone_path, pair)
                if not os.path.isdir(pair_path): continue

                # Tách Source và Target
                if "_to_" in pair:
                    src, tgt = pair.split("_to_")
                else:
                    src, tgt = pair, pair

                for frac_str in os.listdir(pair_path):
                    frac_path = os.path.join(pair_path, frac_str)
                    if not os.path.isdir(frac_path): continue

                    for protocol in ["full_finetuning", "linear_probing"]:
                        summary_file = os.path.join(frac_path, protocol, f"summary_{protocol}_{frac_str}.json")

                        if os.path.exists(summary_file):
                            try:
                                with open(summary_file, 'r', encoding='utf-8') as f:
                                    res = json.load(f)
                                f1_m, f1_s, acc_m, acc_s = extract_metrics(res)

                                data.append({
                                    "Method": method,
                                    "Backbone": backbone,
                                    "Source": src,
                                    "Target": tgt,
                                    "Fraction": frac_str,
                                    "Protocol": protocol,
                                    "F1_Mean": round(f1_m, 2),
                                    "F1_Std": round(f1_s, 2),
                                    "F1_Score": f"{f1_m:.2f} ± {f1_s:.2f}",
                                    "Acc_Mean": round(acc_m, 2),
                                    "Acc_Std": round(acc_s, 2),
                                    "Accuracy": f"{acc_m:.2f} ± {acc_s:.2f}"
                                })
                            except Exception as e:
                                print(f"Lỗi đọc file {summary_file}: {e}")

    return pd.DataFrame(data)


def main():
    print("Đang tổng hợp dữ liệu Cross-Fraction...")
    df_frac = parse_fraction_results()

    if df_frac.empty:
        print("Không có dữ liệu fraction nào!")
        return

    # Sắp xếp lại thứ tự cột chuẩn
    cols = ['Method', 'Backbone', 'Source', 'Target', 'Fraction', 'Protocol', 'F1_Score', 'Accuracy', 'F1_Mean',
            'F1_Std', 'Acc_Mean', 'Acc_Std']
    df_frac = df_frac[cols]

    # Lưu file CSV chi tiết
    out_file = os.path.join(OUTPUTS_DIR, "aggregated_fraction_detailed.csv")
    df_frac.to_csv(out_file, index=False)
    print(f"\nĐã lưu chi tiết từng cặp vào: {out_file}")

    # ==========================================
    # BẢNG TỔNG KẾT PIVOT (TRUNG BÌNH CÁC CẶP)
    # ==========================================
    print("\n" + "=" * 80)
    print("BẢNG TỔNG KẾT: TRUNG BÌNH F1-SCORE THEO FRACTION (% NHÃN)")
    print("=" * 80)

    summary = df_frac.groupby(['Fraction', 'Protocol', 'Method', 'Backbone']).agg(
        Avg_F1_Mean=('F1_Mean', 'mean'),
        Avg_F1_Std=('F1_Std', 'mean')
    ).reset_index()

    summary['F1_Report'] = summary.apply(lambda r: f"{r['Avg_F1_Mean']:.2f} ± {r['Avg_F1_Std']:.2f}", axis=1)

    pivot_df = summary.pivot_table(
        values='F1_Report',
        index=['Fraction', 'Protocol'],
        columns=['Method', 'Backbone'],
        aggfunc='first'
    )

    print(pivot_df.to_string(na_rep='-'))

    pivot_out = os.path.join(OUTPUTS_DIR, "pivot_fraction_f1_report.csv")
    pivot_df.to_csv(pivot_out)
    print(f"\nĐã lưu bảng tổng hợp báo cáo vào: {pivot_out}")


if __name__ == "__main__":
    main()