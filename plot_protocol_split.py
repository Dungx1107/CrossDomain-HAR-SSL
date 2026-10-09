import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# Cấu hình thẩm mỹ phong cách Academic Journal
# =========================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

os.makedirs('plots', exist_ok=True)

# =========================================================
# ĐỌC VÀ LỌC DỮ LIỆU TỪ CSV
# =========================================================
csv_path = '/home/dungx/Workspace/AI_ML_Projects/Human_Activity_Recognition/CrossDomain-HAR-SSL/outputs_evaluation/aggregated_k_shot_detailed.csv'
df = pd.read_csv(csv_path)

# Lọc lấy tất cả kịch bản có 100-shot
df_100 = df[df['Shots'] == '100_shot']

# Tách thành 2 tập: Pocket-level (không chứa watch) và Wrist-level (chỉ có watch)
df_pocket = df_100[df_100['Target'] != 'hhar_watch']
df_watch = df_100[df_100['Target'] == 'hhar_watch']

configs_info = [
    ('crosshar', 'standard', 'CrossHAR\n(Standard)'),
    ('contrastive', 'standard', 'Contrastive\n(Standard)'),
    ('prototype', 'standard', 'Prototype\n(Standard)'),
    ('prototype', 'cnn_transformer', 'Prototype\n(CNN-Trans)'),
    ('crosshar', 'cnn_transformer', 'CrossHAR\n(CNN-Trans)'),
    ('contrastive', 'cnn_transformer', 'Contrastive\n(CNN-Trans)')
]


# =========================================================
# HÀM TẠO BIỂU ĐỒ (DÙNG CHUNG CHO CẢ 2 TRƯỜNG HỢP)
# =========================================================
def plot_protocol_gap(df_subset, title, save_name, y_min, y_max):
    configs = [c[2] for c in configs_info]
    ft_scores = []
    lp_scores = []

    for method, bb, _ in configs_info:
        # Lấy điểm Full Fine-tuning
        ft = df_subset[(df_subset['Method'] == method) &
                       (df_subset['Backbone'] == bb) &
                       (df_subset['Protocol'] == 'full_finetuning')]['F1_Mean'].mean()
        # Lấy điểm Linear Probing
        lp = df_subset[(df_subset['Method'] == method) &
                       (df_subset['Backbone'] == bb) &
                       (df_subset['Protocol'] == 'linear_probing')]['F1_Mean'].mean()

        ft_scores.append(ft)
        lp_scores.append(lp)

    # Tính độ chênh lệch
    gaps = [ft - lp for ft, lp in zip(ft_scores, lp_scores)]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width / 2, ft_scores, width, label='Full Fine-Tuning (FT)', color='#2ca02c', alpha=0.85)
    rects2 = ax.bar(x + width / 2, lp_scores, width, label='Linear Probing (LP)', color='#d62728', alpha=0.85)

    ax.set_ylabel('Macro F1-Score (%) tại 100-Shot')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right')

    # Vẽ số % chênh lệch lên trên đỉnh mỗi cặp cột
    for i, gap in enumerate(gaps):
        sign = "+" if gap >= 0 else ""
        offset = (y_max - y_min) * 0.02  # Căn chữ lùi lên một chút so với cột

        # Để đảm bảo text ko bị đè, ưu tiên ghi text tại độ cao của cột lớn hơn
        text_y = max(ft_scores[i], lp_scores[i]) + offset

        ax.annotate(f'$\\Delta={sign}{gap:.1f}\\%$',
                    xy=(x[i], text_y),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(f'plots/{save_name}.png')
    plt.close()


# =========================================================
# VẼ VÀ LƯU ẢNH
# =========================================================

# 1. Ảnh dành riêng cho Pocket-to-Pocket (Loại bỏ Watch)
plot_protocol_gap(
    df_subset=df_pocket,
    title='Khoảng cách Thích ứng (FT vs LP) - Cấu hình Pocket-to-Pocket (100-Shot)',
    save_name='protocol_adaptation_gap_pocket',
    y_min=45, y_max=90  # Trục Y cho nhóm Pocket thường từ 50-85%
)

# 2. Ảnh dành riêng cho Pocket-to-Wrist (Chỉ Target là Watch)
plot_protocol_gap(
    df_subset=df_watch,
    title='Khoảng cách Thích ứng (FT vs LP) - Cấu hình Pocket-to-Wrist (100-Shot)',
    save_name='protocol_adaptation_gap_wrist',
    y_min=10, y_max=60  # Trục Y cho nhóm Watch thường rất thấp, từ 20-50%
)

print("Đã tạo thành công 2 biểu đồ Protocol Gap riêng biệt tại thư mục plots/!")