import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines  # Thêm thư viện để vẽ Legend tùy chỉnh

# Thiết lập đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs_evaluation")
CSV_PATH = os.path.join(OUTPUTS_DIR, "aggregated_k_shot_detailed.csv")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

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
    df['Model_Config'] = df['Method'].str.title() + " (" + df['Backbone'] + ")"
    df['Source_Name'] = df['Source'].map(DOMAIN_NAMES)
    df['Target_Name'] = df['Target'].map(DOMAIN_NAMES)
    return df


def plot_2x2_source_grid_hybrid(df, shot_value='100_shot'):
    # Lọc riêng tập dữ liệu cho 2 giao thức
    df_ft = df[(df['Shots'] == shot_value) & (df['Protocol'] == 'full_finetuning')]
    df_lp = df[(df['Shots'] == shot_value) & (df['Protocol'] == 'linear_probing')]

    if df_ft.empty or df_lp.empty:
        print(f"Không đủ dữ liệu cho {shot_value}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    sources = ['motionsense', 'uci_har', 'hhar_phone', 'hhar_watch']

    # Điều hướng tự động cấu hình theo mức Shot
    if shot_value == '100_shot':
        hue_order = [
            "Prototype (cnn_transformer)", "Prototype (standard)",
            "Contrastive (cnn_transformer)", "Contrastive (standard)"
        ]
        cluster_width = 0.8
    else:
        hue_order = [
            "Prototype (cnn_transformer)", "Prototype (standard)"
        ]
        cluster_width = 0.4

        # Cố định màu để cột (Bars) luôn cùng tông màu
    palette_dict = {
        "Prototype (cnn_transformer)": "#66c2a5",
        "Prototype (standard)": "#fc8d62",
        "Contrastive (cnn_transformer)": "#8da0cb",
        "Contrastive (standard)": "#e78ac3"
    }

    for i, src in enumerate(sources):
        ax = axes[i]
        df_src_ft = df_ft[df_ft['Source'] == src]
        df_src_lp = df_lp[df_lp['Source'] == src]

        # 1. Vẽ CỘT (Đại diện Full Finetuning)
        sns.barplot(
            data=df_src_ft, x='Target_Name', y='F1_Mean',
            hue='Model_Config', hue_order=hue_order,
            ax=ax, palette=palette_dict, edgecolor='black', linewidth=0.8,
            width=cluster_width  # Ép width để cột mốc shot nhỏ vẫn giữ độ thanh mảnh
        )

        # 2. Đè CHẤM KIM CƯƠNG (Đại diện Linear Probing)
        sns.stripplot(
            data=df_src_lp, x='Target_Name', y='F1_Mean',
            hue='Model_Config', hue_order=hue_order, dodge=True,
            ax=ax, palette=["#222222"] * len(hue_order),  # Cố định chấm đen kim cương
            marker='D', size=7, linewidth=0.5, edgecolor='white', legend=False
        )

        # Trang trí Subplot
        src_pretty = DOMAIN_NAMES.get(src, src)
        ax.set_title(f"Source: {src_pretty}", fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Target Domain", fontsize=12)
        ax.set_ylabel("Macro F1-Score (%)" if i % 2 == 0 else "", fontsize=12)
        ax.set_ylim(0, 100)

        # Chỉ cấu hình Legend ở biểu đồ đầu tiên
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Cắt bớt phần legend bị trùng của seaborn
            handles, labels = handles[:len(hue_order)], labels[:len(hue_order)]

            # Khởi tạo custom legend cho Dấu Kim Cương
            lp_marker = mlines.Line2D([], [], color='none', marker='D', markerfacecolor='#222222',
                                      markeredgecolor='white', markersize=8, label='Linear Probing')

            # Gắn vào khung chú thích chung
            handles.append(lp_marker)
            labels.append('Linear Probing (♦)')

            ax.legend(handles=handles, labels=labels, title="Metrics Configuration",
                      fontsize=11, title_fontsize=12, loc='upper left', framealpha=0.95)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Tổng thể ảnh
    fig.suptitle(f"Cross-Domain F1-Score: Full Finetuning vs Linear Probing ({shot_value.replace('_', ' ').title()})",
                 fontsize=20, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out_path = os.path.join(FIGURES_DIR, f"hybrid_grid_2x2_{shot_value}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Thành công! Ảnh biểu đồ Lai cho {shot_value} đã lưu tại: {out_path}")


def main():
    df = load_data()
    if df is not None:
        plot_2x2_source_grid_hybrid(df, shot_value='20_shot')
        plot_2x2_source_grid_hybrid(df, shot_value='100_shot')


if __name__ == "__main__":
    main()