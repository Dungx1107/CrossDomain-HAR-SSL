import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Cấu hình thẩm mỹ phong cách Academic Journal
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

os.makedirs('plots', exist_ok=True)

# =========================================================
# ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# =========================================================
csv_path = '/home/dungx/Workspace/AI_ML_Projects/Human_Activity_Recognition/CrossDomain-HAR-SSL/outputs_evaluation/aggregated_k_shot_detailed.csv'
df = pd.read_csv(csv_path)

# Trích xuất số Shot ra dạng int để dễ sort (vd: '50_shot' -> 50)
df['Shots_int'] = df['Shots'].str.replace('_shot', '').astype(int)

# Subset 1: Lấy các kịch bản Pocket-level (Loại bỏ đích là Wrist/hhar_watch)
df_pocket = df[df['Target'] != 'hhar_watch']


# =========================================================
# Biểu đồ 1: Đường cong tăng trưởng theo K-Shot
# =========================================================
shots = [10, 20, 50, 100]

def get_k_shot_curve(method, backbone, protocol):
    sub = df_pocket[(df_pocket['Method'] == method) &
                    (df_pocket['Backbone'] == backbone) &
                    (df_pocket['Protocol'] == protocol)]
    # Tính mean theo từng số shot và sort
    grouped = sub.groupby('Shots_int')['F1_Mean'].mean().sort_index()
    return grouped.loc[shots].tolist()

# Lấy tự động array F1-score từ DataFrame
proto_std_ft = get_k_shot_curve('prototype', 'standard', 'full_finetuning')
proto_std_lp = get_k_shot_curve('prototype', 'standard', 'linear_probing')
proto_trans_ft = get_k_shot_curve('prototype', 'cnn_transformer', 'full_finetuning')
proto_trans_lp = get_k_shot_curve('prototype', 'cnn_transformer', 'linear_probing')

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(shots, proto_std_ft, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Standard 1D-CNN (Full FT)')
ax.plot(shots, proto_std_lp, 's--', color='#1f77b4', linewidth=2.0, markersize=7, alpha=0.8, label='Standard 1D-CNN (Linear Probe)')
ax.plot(shots, proto_trans_ft, 'd-', color='#ff7f0e', linewidth=2.5, markersize=8, label='CNN-Transformer (Full FT)')
ax.plot(shots, proto_trans_lp, '^--', color='#ff7f0e', linewidth=2.0, markersize=7, alpha=0.8, label='CNN-Transformer (Linear Probe)')

ax.set_xlabel('Số lượng mẫu gán nhãn ở miền đích ($K$-shot)')
ax.set_ylabel('Macro F1-Score (%)')
ax.set_title('Tác động của Số lượng $K$-Shot đến Hiệu năng Thích ứng Miền (Prototype SSL)')
ax.set_xticks(shots)
ax.set_ylim(50, 85)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('plots/k_shot_scaling_curves.png')
plt.close()


# =========================================================
# Biểu đồ 2: Khoảng cách Thích ứng Protocol Adaptation Gap
# =========================================================
# Lọc lấy 100-shot pocket-level
df_100 = df_pocket[df_pocket['Shots_int'] == 100]

configs_info = [
    ('crosshar', 'standard', 'CrossHAR\n(Standard)'),
    ('contrastive', 'standard', 'Contrastive\n(Standard)'),
    ('prototype', 'standard', 'Prototype\n(Standard)'),
    ('prototype', 'cnn_transformer', 'Prototype\n(CNN-Trans)'),
    ('crosshar', 'cnn_transformer', 'CrossHAR\n(CNN-Trans)'),
    ('contrastive', 'cnn_transformer', 'Contrastive\n(CNN-Trans)')
]

configs = [c[2] for c in configs_info]
ft_scores = []
lp_scores = []

# Tự động tính trung bình FT và LP cho từng config
for method, bb, _ in configs_info:
    ft = df_100[(df_100['Method'] == method) & (df_100['Backbone'] == bb) & (df_100['Protocol'] == 'full_finetuning')]['F1_Mean'].mean()
    lp = df_100[(df_100['Method'] == method) & (df_100['Backbone'] == bb) & (df_100['Protocol'] == 'linear_probing')]['F1_Mean'].mean()
    ft_scores.append(ft)
    lp_scores.append(lp)

gaps = [ft - lp for ft, lp in zip(ft_scores, lp_scores)]
x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
rects1 = ax.bar(x - width/2, ft_scores, width, label='Full Fine-Tuning (FT)', color='#2ca02c', alpha=0.85)
rects2 = ax.bar(x + width/2, lp_scores, width, label='Linear Probing (LP)', color='#d62728', alpha=0.85)

ax.set_ylabel('Macro F1-Score (%) tại 100-Shot')
ax.set_title('So sánh Đặc trưng Đóng băng (LP) và Khả năng Thích ứng (FT) tại 100-Shot')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(45, 90)
ax.legend(loc='upper right')

# Hiển thị độ chênh lệch Δ(FT-LP) trên đỉnh
for i, gap in enumerate(gaps):
    ax.annotate(f'$\\Delta=+{gap:.1f}\\%$',
                xy=(x[i], ft_scores[i] + 1.2),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('plots/protocol_adaptation_gap.png')
plt.close()


# =========================================================
# Biểu đồ 3: Nút thắt Vị trí Cảm biến (Pocket/Waist vs Wrist)
# =========================================================
domains = ['Pocket-to-Pocket\n(UCI-HAR $\\leftrightarrow$ MotionSense)', 'Pocket-to-Wrist\n(MotionSense $\\rightarrow$ HHAR Watch)']

# Tự động tính F1_Mean tổng trên các tập Source/Target tương ứng
p2p_df = df[((df['Source'] == 'uci_har') & (df['Target'] == 'motionsense')) |
            ((df['Source'] == 'motionsense') & (df['Target'] == 'uci_har'))]

p2w_df = df[(df['Source'] == 'motionsense') & (df['Target'] == 'hhar_watch')]

f1_means = [p2p_df['F1_Mean'].mean(), p2w_df['F1_Mean'].mean()]

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
bars = ax.bar(domains, f1_means, color=['#1f77b4', '#d62728'], width=0.45, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Macro F1-Score Trung bình (%)')
ax.set_title('Sụt giảm Hiệu năng do Nút thắt Động học Cảm biến Wrist (Watch)')
ax.set_ylim(0, 90)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('plots/wrist_domain_bottleneck.png')
plt.close()

print("Đã tạo thành công 3 biểu đồ (đọc trực tiếp từ aggregated_k_shot_detailed.csv) tại thư mục plots/!")