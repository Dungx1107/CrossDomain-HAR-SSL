import os
import json
import argparse
import pandas as pd

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN TĨNH
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
K_SHOT_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation", "cross_k_shot")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation")
DEFAULT_OUT_FILE = os.path.join(OUTPUTS_DIR, "aggregated_k_shot_detailed.csv")

# =============================================================================
# XỬ LÝ THAM SỐ DÒNG LỆNH
# =============================================================================
parser = argparse.ArgumentParser(description="Tổng hợp kết quả Few-shot Cross-Domain")
parser.add_argument("--method", type=str, default=None,
                    help="Tên phương pháp cần tổng hợp (vd: tstcc). Mặc định: quét tất cả.")
parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "append"],
                    help="overwrite: Ghi đè file cũ. append: Ghi nối vào file cũ.")
parser.add_argument("--out_name", type=str, default="aggregated_k_shot_detailed.csv",
                    help="Tên file CSV đầu ra (mặc định: aggregated_k_shot_detailed.csv)")
args = parser.parse_args()

OUT_FILE = os.path.join(OUTPUTS_DIR, args.out_name)


def extract_metrics(res):
    """Trích xuất F1, Acc và Timing từ dict summary."""
    if "overall_metrics" in res:
        metrics = res["overall_metrics"]
        timing = res.get("timing", {})

        f1_m = metrics.get("macro_f1_mean", 0.0)
        f1_s = metrics.get("macro_f1_std", 0.0)
        acc_m = metrics.get("accuracy_mean", 0.0)
        acc_s = metrics.get("accuracy_std", 0.0)
        prec_m = metrics.get("macro_precision_mean", 0.0)
        rec_m = metrics.get("macro_recall_mean", 0.0)

        t_train = timing.get("train_time_mean_sec", 0.0)
        t_infer = timing.get("inference_time_mean_sec", 0.0)

        # Chuẩn hóa thang đo (nếu data cũ bị lưu dạng 0.x)
        if 0 < f1_m <= 1.0:
            f1_m *= 100
            f1_s *= 100
            acc_m *= 100
            acc_s *= 100

        return f1_m, f1_s, acc_m, acc_s, prec_m, rec_m, t_train, t_infer
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def parse_k_shot_results():
    data = []
    if not os.path.exists(K_SHOT_DIR):
        print(f"❌ Không tìm thấy thư mục: {K_SHOT_DIR}")
        return pd.DataFrame()

    # Lấy danh sách method (lọc theo tham số nếu có)
    methods = [args.method] if args.method else [m for m in os.listdir(K_SHOT_DIR) if
                                                 os.path.isdir(os.path.join(K_SHOT_DIR, m))]

    for method in methods:
        method_path = os.path.join(K_SHOT_DIR, method)
        if not os.path.isdir(method_path): continue

        for backbone in os.listdir(method_path):
            backbone_path = os.path.join(method_path, backbone)
            if not os.path.isdir(backbone_path): continue

            for pair in os.listdir(backbone_path):
                pair_path = os.path.join(backbone_path, pair)
                if not os.path.isdir(pair_path): continue

                src, tgt = pair.split("_to_") if "_to_" in pair else (pair, pair)

                for shot_str in os.listdir(pair_path):
                    shot_path = os.path.join(pair_path, shot_str)
                    if not os.path.isdir(shot_path): continue

                    # TỰ ĐỘNG QUÉT MỌI PROTOCOL (full, linear, partial_1blocks...)
                    for protocol in os.listdir(shot_path):
                        protocol_path = os.path.join(shot_path, protocol)
                        if not os.path.isdir(protocol_path): continue

                        summary_file = os.path.join(protocol_path, f"summary_{protocol}_{shot_str}.json")

                        if os.path.exists(summary_file):
                            try:
                                with open(summary_file, 'r') as f:
                                    res = json.load(f)
                                f1_m, f1_s, acc_m, acc_s, prec_m, rec_m, t_train, t_infer = extract_metrics(res)

                                data.append({
                                    "Method": method,
                                    "Backbone": backbone,
                                    "Source": src,
                                    "Target": tgt,
                                    "Shots": shot_str,
                                    "Protocol": protocol,
                                    "F1_Mean": round(f1_m, 2),
                                    "F1_Std": round(f1_s, 2),
                                    "F1_Score": f"{f1_m:.2f} ± {f1_s:.2f}",
                                    "Acc_Mean": round(acc_m, 2),
                                    "Acc_Std": round(acc_s, 2),
                                    "Accuracy": f"{acc_m:.2f} ± {acc_s:.2f}",
                                    "Precision_Mean": round(prec_m, 2),
                                    "Recall_Mean": round(rec_m, 2),
                                    "Train_Time_s": round(t_train, 2),
                                    "Infer_Time_s": round(t_infer, 4)
                                })
                            except Exception as e:
                                print(f"⚠️ Lỗi file {summary_file}: {e}")

    return pd.DataFrame(data)


def main():
    print(f"🔍 Đang tổng hợp dữ liệu K-Shot (Mode: {args.mode.upper()})...")
    if args.method:
        print(f"🎯 Chỉ lấy dữ liệu của Method: {args.method}")

    df_new = parse_k_shot_results()

    if df_new.empty:
        print("⚠️ Không tìm thấy dữ liệu mới!")
        return

    # Sắp xếp lại thứ tự cột
    cols = ['Method', 'Backbone', 'Source', 'Target', 'Shots', 'Protocol',
            'F1_Score', 'Accuracy', 'Precision_Mean', 'Recall_Mean',
            'Train_Time_s', 'Infer_Time_s', 'F1_Mean', 'F1_Std', 'Acc_Mean', 'Acc_Std']
    df_new = df_new[cols]

    # Xử lý ghi file (Overwrite vs Append)
    if args.mode == "append" and os.path.exists(OUT_FILE):
        print(f"➕ Đang ghi nối vào file hiện có: {OUT_FILE}")
        df_old = pd.read_csv(OUT_FILE)

        # Loại bỏ các dòng cũ trùng Method + Backbone + Pair + Shots + Protocol để tránh duplicate
        # (Nếu bạn chạy lại cùng 1 cấu hình, nó sẽ ghi đè dòng đó thay vì thêm 1 dòng trùng)
        merge_keys = ['Method', 'Backbone', 'Source', 'Target', 'Shots', 'Protocol']

        # Tạo key so sánh
        df_old['_key'] = df_old[merge_keys].astype(str).agg('_'.join, axis=1)
        df_new['_key'] = df_new[merge_keys].astype(str).agg('_'.join, axis=1)

        # Lọc bỏ những dòng cũ đã có trong new
        df_old_filtered = df_old[~df_old['_key'].isin(df_new['_key'])].drop(columns=['_key'])
        df_final = pd.concat([df_old_filtered, df_new.drop(columns=['_key'])], ignore_index=True)
    else:
        print(f"✍️ Ghi đè file: {OUT_FILE}")
        df_final = df_new

    # Lưu file CSV
    df_final.to_csv(OUT_FILE, index=False)
    print(f"✅ Đã lưu chi tiết {len(df_new)} dòng mới. Tổng cộng file có {len(df_final)} dòng.")

    # ==========================================
    # IN BẢNG TỔNG KẾT (PIVOT)
    # ==========================================
    print("\n" + "=" * 80)
    print("BẢNG TỔNG KẾT: TRUNG BÌNH F1-SCORE (MACRO) TRÊN TOÀN BỘ CÁC CẶP")
    print("=" * 80)

    # Chỉ lấy các dòng có F1 (tránh lỗi)
    df_pivot = df_final[df_final['F1_Mean'] > 0]

    summary = df_pivot.groupby(['Shots', 'Protocol', 'Method', 'Backbone']).agg(
        Avg_F1_Mean=('F1_Mean', 'mean'),
        Avg_F1_Std=('F1_Std', 'mean')
    ).reset_index()

    summary['F1_Report'] = summary.apply(lambda r: f"{r['Avg_F1_Mean']:.2f} ± {r['Avg_F1_Std']:.2f}", axis=1)

    pivot_df = summary.pivot_table(
        values='F1_Report',
        index=['Shots', 'Protocol'],
        columns=['Method', 'Backbone'],
        aggfunc='first'
    )

    print(pivot_df.to_string(na_rep='-'))

    pivot_out = os.path.join(OUTPUTS_DIR, "pivot_k_shot_f1_report.csv")
    pivot_df.to_csv(pivot_out)
    print(f"\n📊 Đã lưu bảng tổng hợp Pivot vào: {pivot_out}")


if __name__ == "__main__":
    main()