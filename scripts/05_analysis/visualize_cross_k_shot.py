import os
import matplotlib.pyplot as plt
import numpy as np
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

# ---------------------------------------------------------
# Biểu đồ 1: Đường cong tăng trưởng theo K-Shot (Shot-Scaling Curves)
# ---------------------------------------------------------
shots = [10, 20, 50, 100]

# Số liệu Pocket-level Prototype
proto_std_ft = [65.24, 74.42, 79.79, 82.32]
proto_std_lp = [56.27, 61.07, 65.96, 70.72]

proto_trans_ft = [65.63, 73.37, 77.59, 81.13]
proto_trans_lp = [55.13, 60.94, 65.64, 69.09]

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

# ---------------------------------------------------------
# Biểu đồ 2: Khoảng cách Thích ứng Protocol Adaptation Gap (FT vs LP)
# ---------------------------------------------------------
configs = [
    'CrossHAR\n(Standard)', 'Contrastive\n(Standard)', 'Prototype\n(Standard)',
    'Prototype\n(CNN-Trans)', 'CrossHAR\n(CNN-Trans)', 'Contrastive\n(CNN-Trans)'
]

ft_scores = [82.11, 82.05, 82.32, 81.13, 80.33, 79.65]
lp_scores = [73.68, 72.20, 70.72, 69.09, 61.79, 54.61]
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

# Hiển thị độ chênh lệch Δ(FT-LP) trên đỉnh mỗi cột
for i, gap in enumerate(gaps):
    ax.annotate(f'$\\Delta=+{gap:.1f}\\%$',
                xy=(x[i], ft_scores[i] + 1.2),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('plots/protocol_adaptation_gap.png')
plt.close()

# ---------------------------------------------------------
# Biểu đồ 3: Nút thắt Vị trí Cảm biến (Pocket/Waist vs Wrist Target)
# ---------------------------------------------------------
domains = ['Pocket-to-Pocket\n(UCI-HAR $\\leftrightarrow$ MotionSense)', 'Pocket-to-Wrist\n(MotionSense $\\rightarrow$ HHAR Watch)']
f1_means = [76.5, 33.7]

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

print("Đã tạo thành công 3 biểu đồ đồ họa tại thư mục plots/!")