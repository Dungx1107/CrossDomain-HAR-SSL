"""
In bảng so sánh F1-Score theo K-Shot cho các config chỉ định.
Chỉ cần sửa CONFIG_LIST / PROTOCOL ở đầu file rồi chạy:
    python scripts/05_analysis/print_k_shot_table.py
"""
import pandas as pd

# ============================================================
# ⚙️ CẤU HÌNH — CHỈNH Ở ĐÂY
# ============================================================
# CONFIG_LIST = ["tstcc/standard_ts3", "tstcc/standard_bs64"]
CONFIG_LIST = ["tstcc/standard_ts3", "tstcc/cnn_lstm"]
# PROTOCOL = "full_finetuning"
PROTOCOL = "linear_probing"

CSV_PATH = "../../outputs_evaluation/aggregated_k_shot_detailed.csv"
DECIMALS = 2
# ============================================================


def _short(cfg_name: str) -> str:
    """Rút gọn tên config cho dễ đọc trong bảng."""
    return cfg_name.replace(" (", "\n(")  # xuống dòng trước backbone


def print_k_shot_table(config_list, protocol, csv_path=CSV_PATH, decimals=DECIMALS):
    df = pd.read_csv(csv_path)
    df["Shots_Int"] = df["Shots"].str.extract(r"(\d+)").astype(int)
    df["Pair"] = df["Source"] + " ➔ " + df["Target"]
    df["Config"] = df["Method"] + " (" + df["Backbone"] + ")"

    config_names = [f"{m} ({b})" for m, b in (c.split("/") for c in config_list)]
    short_names = {c: _short(c) for c in config_names}

    df_f = df[(df["Protocol"] == protocol) & (df["Config"].isin(config_names))].copy()
    if df_f.empty:
        print(f"⚠️ Không có dữ liệu cho configs={config_list} trong protocol={protocol}")
        return

    df_f = df_f.sort_values(["Target", "Source", "Config", "Shots_Int"])

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    print(f"\n{'=' * 90}")
    print(f"  So sánh F1-Score (%) — Protocol: {protocol}")
    print(f"  Configs: {config_names}")
    print(f"{'=' * 90}")

    # ---------- PHẦN 1: TỪNG CẶP × TỪNG SHOT ----------
    print(f"\n{'#' * 90}")
    print("  PHẦN 1 — CHI TIẾT TỪNG CẶP × TỪNG K-SHOT")
    print(f"{'#' * 90}")

    for (src, tgt), grp in df_f.groupby(["Source", "Target"]):
        pair = f"{src} ➔ {tgt}"
        pivot = grp.pivot_table(
            index="Shots_Int", columns="Config", values="F1_Mean", aggfunc="mean"
        )[config_names]

        # Δ so với config đầu tiên
        base = config_names[0]
        for c in config_names[1:]:
            pivot[f"Δ({short_names[base]} − {short_names[c]})"] = pivot[base] - pivot[c]

        pivot = pivot.round(decimals)
        pivot.columns = [c if "Δ(" in c else short_names[c] for c in pivot.columns]

        # Thêm cột Best mỗi shot
        pivot["Best"] = pivot[[short_names[c] for c in config_names]].idxmax(axis=1)

        print(f"\n▶ Cặp: {pair}")
        print(pivot.to_string())

    # ---------- PHẦN 2: TRUNG BÌNH THEO TỪNG CẶP ----------
    print(f"\n{'#' * 90}")
    print("  PHẦN 2 — TRUNG BÌNH THEO TỪNG CẶP (mean trên tất cả K-Shot)")
    print(f"{'#' * 90}")

    avg_pair = (
        df_f.groupby(["Source", "Target", "Config"])["F1_Mean"]
        .mean()
        .unstack("Config")[config_names]
        .round(decimals)
    )
    avg_pair.columns = [short_names[c] for c in avg_pair.columns]
    avg_pair["Best"] = avg_pair.idxmax(axis=1)
    print(avg_pair.to_string())

    # ---------- PHẦN 3: TRUNG BÌNH THEO TARGET ----------
    print(f"\n{'#' * 90}")
    print("  PHẦN 3 — TRUNG BÌNH THEO TARGET (mean trên Source × K-Shot)")
    print(f"{'#' * 90}")

    avg_tgt = (
        df_f.groupby(["Target", "Config"])["F1_Mean"]
        .mean()
        .unstack("Config")[config_names]
        .round(decimals)
    )
    avg_tgt.columns = [short_names[c] for c in avg_tgt.columns]
    avg_tgt["Best"] = avg_tgt.idxmax(axis=1)
    print(avg_tgt.to_string())

    # ---------- PHẦN 4: TRUNG BÌNH TỔNG THỂ ----------
    print(f"\n{'#' * 90}")
    print("  PHẦN 4 — TRUNG BÌNH TỔNG THỂ (mean trên tất cả Target × Source × Shots)")
    print(f"{'#' * 90}")

    overall = (
        df_f.groupby("Config")["F1_Mean"]
        .mean()
        .reindex(config_names)
        .round(decimals)
        .to_frame(name="F1_Mean")
    )
    overall.index = [short_names[c] for c in overall.index]
    overall["Rank"] = overall["F1_Mean"].rank(ascending=False).astype(int)
    overall = overall.sort_values("F1_Mean", ascending=False)
    print(overall.to_string())

    # ---------- PHẦN 5: SỐ LẦN WIN CỦA MỖI CONFIG ----------
    print(f"\n{'#' * 90}")
    print("  PHẦN 5 — SỐ LẦN 'WIN' (config có F1 cao nhất)")
    print(f"{'#' * 90}")

    win_count = {c: 0 for c in config_names}
    total_cases = 0

    for (src, tgt, shots), grp in df_f.groupby(["Source", "Target", "Shots_Int"]):
        vals = grp.set_index("Config")["F1_Mean"].reindex(config_names)
        if vals.isna().all():
            continue
        best_cfg = vals.idxmax()
        win_count[best_cfg] += 1
        total_cases += 1

    win_df = pd.DataFrame(
        {
            "Wins": [win_count[c] for c in config_names],
            "Win %": [100 * win_count[c] / total_cases for c in config_names],
        },
        index=[short_names[c] for c in config_names],
    ).round(1)
    win_df = win_df.sort_values("Wins", ascending=False)
    print(f"Tổng số case (Source × Target × Shot): {total_cases}\n")
    print(win_df.to_string())

    print(f"\n{'=' * 90}\n")


if __name__ == "__main__":
    print_k_shot_table(CONFIG_LIST, PROTOCOL)