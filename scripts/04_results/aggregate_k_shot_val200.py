import os
import json
import argparse
import numpy as np
import pandas as pd

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN TĨNH
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VAL_DIAG_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation", "val_diagnostic")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation")

# =============================================================================
# XỬ LÝ THAM SỐ DÒNG LỆNH
# =============================================================================
parser = argparse.ArgumentParser(
    description="Tổng hợp kết quả Few-shot Cross-Domain khi CHỌN EPOCH BẰNG VAL (val_diagnostic) "
                "theo đúng định dạng của aggregate_k_shot.py. Method được đổi thành '<method>_val<N>'.")
parser.add_argument("--method", type=str, default=None,
                    help="Tên phương pháp cần tổng hợp (vd: tstcc). Mặc định: quét tất cả.")
parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "append"],
                    help="overwrite: Ghi đè file cũ. append: Ghi nối vào file cũ.")
parser.add_argument("--out_name", type=str, default="aggregated_k_shot_val200_detailed.csv",
                    help="Tên file CSV đầu ra (mặc định: aggregated_k_shot_val200_detailed.csv). "
                         "Muốn gộp chung với bảng k-shot thường: --out_name aggregated_k_shot_detailed.csv --mode append")
parser.add_argument("--pivot_name", type=str, default="pivot_k_shot_val200_f1_report.csv",
                    help="Tên file pivot đầu ra.")
parser.add_argument("--max_epoch", type=int, default=40,
                    help="Chỉ chọn epoch tốt nhất (theo Val F1) trong số epoch này đầu tiên (mặc định 40).")
parser.add_argument("--val", type=int, default=None,
                    help="Ép nhãn '<method>_val<val>'. Mặc định: tự suy ra từ val_per_class_counts trong JSON.")
parser.add_argument("--ddof", type=int, default=0,
                    help="ddof khi tính độ lệch chuẩn qua các seed (0 = np.std mặc định). "
                         "Đặt cho khớp với cách few_shot_report.py tính macro_f1_std.")
args = parser.parse_args()

OUT_FILE = os.path.join(OUTPUTS_DIR, args.out_name)
PIVOT_FILE = os.path.join(OUTPUTS_DIR, args.pivot_name)


def extract_seed_metrics(res):
    """
    Từ 1 file seed*.json: chọn epoch có Val F1 cao nhất trong max_epoch epoch đầu,
    trả (test_f1, test_acc, run_time_sec) tại epoch đó. Trả None nếu dữ liệu không hợp lệ.
    """
    hist = res.get("history", {}) or {}
    val_f1 = hist.get("val_f1") or []
    test_f1 = hist.get("test_f1") or []
    test_acc = hist.get("test_acc") or []

    n_all = len(test_f1)
    if n_all == 0 or len(val_f1) != n_all:
        return None

    n = min(max(args.max_epoch, 1), n_all)
    best = max(range(n), key=lambda i: val_f1[i])          # hòa -> epoch sớm nhất

    acc = test_acc[best] if len(test_acc) == n_all else np.nan
    return test_f1[best], acc, res.get("run_time_sec", np.nan)


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    m = float(arr.mean())
    s = float(arr.std(ddof=args.ddof)) if len(arr) > args.ddof else 0.0
    return m, s


def parse_val_k_shot_results():
    data = []
    if not os.path.exists(VAL_DIAG_DIR):
        print(f"❌ Không tìm thấy thư mục: {VAL_DIAG_DIR}")
        return pd.DataFrame()

    # Lấy danh sách method (lọc theo tham số nếu có)
    methods = [args.method] if args.method else [m for m in os.listdir(VAL_DIAG_DIR) if
                                                 os.path.isdir(os.path.join(VAL_DIAG_DIR, m))]

    for method in methods:
        method_path = os.path.join(VAL_DIAG_DIR, method)
        if not os.path.isdir(method_path): continue

        for backbone in os.listdir(method_path):
            backbone_path = os.path.join(method_path, backbone)
            if not os.path.isdir(backbone_path): continue

            for pair in os.listdir(backbone_path):
                pair_path = os.path.join(backbone_path, pair)
                if not os.path.isdir(pair_path): continue

                src, tgt = pair.split("_to_") if "_to_" in pair else (pair, pair)

                for shot_dir in os.listdir(pair_path):
                    shot_path = os.path.join(pair_path, shot_dir)
                    if not os.path.isdir(shot_path): continue

                    # Thư mục val_diagnostic là "k10"; bảng k-shot thường dùng "10_shot" -> đổi cho đồng nhất
                    try:
                        shot_str = f"{int(shot_dir.lstrip('k'))}_shot"
                    except ValueError:
                        shot_str = shot_dir

                    # TỰ ĐỘNG QUÉT MỌI PROTOCOL (full_finetuning, linear_probing, partial_Nblocks...)
                    for protocol in os.listdir(shot_path):
                        protocol_path = os.path.join(shot_path, protocol)
                        if not os.path.isdir(protocol_path): continue

                        f1s, accs, times, val_label = [], [], [], None
                        for fname in sorted(os.listdir(protocol_path)):
                            if not (fname.startswith("seed") and fname.endswith(".json")):
                                continue
                            fpath = os.path.join(protocol_path, fname)
                            try:
                                with open(fpath, 'r') as f:
                                    res = json.load(f)
                                out = extract_seed_metrics(res)
                                if out is None:
                                    continue
                                f1s.append(out[0])
                                accs.append(out[1])
                                times.append(out[2])
                                if val_label is None:
                                    counts = res.get("val_per_class_counts") or []
                                    val_label = args.val if args.val is not None else (max(counts) if counts else "NA")
                            except Exception as e:
                                print(f"⚠️ Lỗi file {fpath}: {e}")

                        if not f1s:
                            continue

                        f1_m, f1_s = _mean_std(f1s)
                        acc_m, acc_s = _mean_std(accs)
                        t_train, _ = _mean_std(times)

                        data.append({
                            "Method": f"{method}_val{val_label}",
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
                            "Precision_Mean": np.nan,      # val_diagnostic không lưu precision/recall
                            "Recall_Mean": np.nan,
                            "Train_Time_s": round(t_train, 2),   # gồm cả thời gian đánh giá val+test mỗi epoch
                            "Infer_Time_s": np.nan                # không đo trong val_diagnostic
                        })

    return pd.DataFrame(data)


def main():
    print(f"🔍 Đang tổng hợp dữ liệu K-Shot CHỌN EPOCH BẰNG VAL (Mode: {args.mode.upper()}, "
          f"trong {args.max_epoch} epoch đầu)...")
    if args.method:
        print(f"🎯 Chỉ lấy dữ liệu của Method: {args.method}")

    df_new = parse_val_k_shot_results()

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
    print("BẢNG TỔNG KẾT: TRUNG BÌNH F1-SCORE (MACRO) TRÊN TOÀN BỘ CÁC CẶP (CHỌN EPOCH BẰNG VAL)")
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

    # Sắp theo số shot (10, 30, 50, 80, 100...) thay vì theo chuỗi
    def _shot_key(idx):
        try:
            return int(str(idx[0]).split('_')[0]), str(idx[1])
        except ValueError:
            return 10 ** 9, str(idx[1])
    pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=_shot_key))

    print(pivot_df.to_string(na_rep='-'))

    pivot_df.to_csv(PIVOT_FILE)
    print(f"\n📊 Đã lưu bảng tổng hợp Pivot vào: {PIVOT_FILE}")


if __name__ == "__main__":
    main()