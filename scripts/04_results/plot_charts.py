import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation")
CSV_PATH = os.path.join(OUTPUTS_DIR, "aggregated_k_shot_detailed.csv")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Tên hiển thị chuẩn báo cáo
DOMAIN_NAMES = {
    'motionsense': 'MotionSense',
    'uci_har': 'UCI-HAR',
    'hhar_phone': 'HHAR-Phone',
    'hhar_watch': 'HHAR-Watch'
}


def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy file: {CSV_PATH}")
        return None
    df = pd.read_csv(CSV_PATH)

    # Tạo cột Model_Config để hiển thị chú thích (Legend)
    df['Model_Config'] = df['Method'].str.title() + " (" + df['Backbone'] + ")"

    # Ánh xạ tên miền cho đẹp
    df['Source_Name'] = df['Source'].map(DOMAIN_NAMES)
    df['Target_Name'] = df['Target'].map(DOMAIN_NAMES)
    return df

    # Vẽ biểu đồ lưới cho mức 100-shot


def plot_2x2_source_grid(df, shot_value='100_shot', protocol='full_finetuning'):
    # Lọc dữ liệu theo mốc (100_shot) và kịch bản (full_finetuning)
    df_plot = df[(df['Shots'] == shot_value) & (df['Protocol'] == protocol)].copy()

    if df_plot.empty:
        print(f"Không có dữ liệu cho {shot_value} và {protocol}")
        return

    # Khởi tạo lưới 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    sns.set_theme(style="whitegrid")
    sources = ['motionsense', 'uci_har', 'hhar_phone', 'hhar_watch']

    # Thứ tự cố định để màu sắc các biểu đồ đồng nhất với nhau
    if shot_value == '100_shot':
        hue_order = [
            "Prototype (cnn_transformer)", "Prototype (standard)",
            "Contrastive (cnn_transformer)", "Contrastive (standard)"
        ]
        cluster_width = 0.8
    else:
        hue_order = [
            "Prototype (cnn_transformer)", "Prototype (standard)",
        ]
        cluster_width = 0.4

        # Bảng màu cố định để Prototype không bị đổi màu giữa các mốc
    palette_dict = {
        "Prototype (cnn_transformer)": "#66c2a5",
        "Prototype (standard)": "#fc8d62",
        "Contrastive (cnn_transformer)": "#8da0cb",
        "Contrastive (standard)": "#e78ac3"
    }

    for i, src in enumerate(sources):
        ax = axes[i]
        df_src = df_plot[df_plot['Source'] == src]

        # Vẽ cột
        sns.barplot(
            data=df_src,
            x='Target_Name',
            y='F1_Mean',
            hue='Model_Config',
            hue_order=hue_order,
            ax=ax,
            palette=palette_dict,
            edgecolor='black',
            linewidth=0.8,
            width=cluster_width
        )

        # Trang trí từng ô phụ (Subplot)
        src_pretty = DOMAIN_NAMES.get(src, src)
        ax.set_title(f"Source: {src_pretty}", fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Target Domain", fontsize=12)
        ax.set_ylabel("Macro F1-Score (%)" if i % 2 == 0 else "", fontsize=12)
        ax.set_ylim(0, 100)  # Ép khung Y luôn là 100% để trực quan

        # Dọn dẹp Legend (Chỉ hiện 1 bảng ghi chú chung ở góc trên bên trái, xóa ở các ô khác để tránh rác)
        if i == 0:
            ax.legend(title="Method & Backbone", fontsize=10, title_fontsize=11, loc='upper left', framealpha=0.9)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Tiêu đề tổng quát cho toàn bức ảnh
    fig.suptitle(f"Cross-Domain Adaptation Performance ({shot_value.replace('_', ' ').title()} - Full Finetuning)",
                 fontsize=20, fontweight='bold', y=0.96)

    # Căn lề tự động tránh đè chữ
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out_path = os.path.join(FIGURES_DIR, f"source_grid_2x2_{shot_value}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Đã vẽ xong lưới 2x2! Ảnh được lưu tại: {out_path}")


def main():
    df = load_data()
    if df is not None:
        plot_2x2_source_grid(df, shot_value='10_shot', protocol='full_finetuning')
        plot_2x2_source_grid(df, shot_value='20_shot', protocol='full_finetuning')
        plot_2x2_source_grid(df, shot_value='50_shot', protocol='full_finetuning')
        plot_2x2_source_grid(df, shot_value='100_shot', protocol='full_finetuning')


if __name__ == "__main__":
    main()
