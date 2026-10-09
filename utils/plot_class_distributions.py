import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# ==============================================================================
# CẤU HÌNH THẨM MỸ (ACADEMIC JOURNAL STYLE)
# ==============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10.5
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

os.makedirs('plots', exist_ok=True)

# 5 lớp hoạt động chuẩn dùng trong kịch bản Cross-Domain
COMMON_CLASSES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']

# ==============================================================================
# ĐƯỜNG DẪN DỮ LIỆU ĐÃ XỬ LÝ (.pt)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in locals() else Path.cwd()
PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"

DATASETS_CONFIG = {
    'motionsense': {
        'title': 'MotionSense (Túi quần trước)',
        'dir': PROCESSED_BASE / 'motionsense',
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Jogging'],
    },
    'uci_har': {
        'title': 'UCI-HAR (Thắt lưng)',
        'dir': PROCESSED_BASE / 'uci_har',
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'],
    },
    'hhar_phone': {
        'title': 'HHAR Phone (Túi quần / Cầm tay)',
        'dir': PROCESSED_BASE / 'hhar_phone',
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
    },
    'hhar_watch': {
        'title': 'HHAR Watch (Cổ tay)',
        'dir': PROCESSED_BASE / 'hhar_watch',
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
    }
}

def load_split_percentages(dataset_dir: Path, target_classes: list):
    """
    Đọc các file train.pt, val.pt, test.pt.
    Trả về:
      - percentages: Tỷ lệ % của 5 lớp hoạt động chung.
      - sample_counts: Tổng số cửa sổ trượt (samples).
      - subject_counts: Số người tham gia (unique subjects).
    """
    splits = ['train', 'val', 'test']
    percentages = {}
    sample_counts = {}
    subject_counts = {}

    for split in splits:
        file_path = dataset_dir / f"{split}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        data = torch.load(file_path, map_location='cpu')
        labels = data['labels'].numpy()
        total_samples = len(labels)
        sample_counts[split] = total_samples

        # Đếm số lượng người tham gia duy nhất từ tensor subjects
        if 'subjects' in data and data['subjects'] is not None:
            subs = data['subjects'].numpy()
            unique_subs = np.unique(subs)
            subject_counts[split] = len(unique_subs)
        else:
            subject_counts[split] = 0

        # Đếm số lượng mẫu cho từng lớp trong target_classes (id: 0..4)
        pcts = []
        for class_idx in range(len(target_classes)):
            count = np.sum(labels == class_idx)
            pct = (count / total_samples * 100.0) if total_samples > 0 else 0.0
            pcts.append(pct)
        percentages[split] = pcts

    return percentages, sample_counts, subject_counts

# ==============================================================================
# VẼ LƯỚI BIỂU ĐỒ 2x2
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300, sharey=True)
axes_flat = axes.flatten()

# Cấu hình thanh bar
bar_width = 0.25
split_colors = {
    'train': '#2b5c8f',  # Xanh dương đậm
    'val': '#d95f02',    # Cam đất
    'test': '#7570b3'    # Tím nhạt
}
split_labels = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

for idx, (ds_key, ds_info) in enumerate(DATASETS_CONFIG.items()):
    ax = axes_flat[idx]
    pct_dict, counts, subs = load_split_percentages(ds_info['dir'], COMMON_CLASSES)

    x = np.arange(len(COMMON_CLASSES))

    # Vẽ 3 cột nhóm cho Train, Val, Test
    rects_train = ax.bar(x - bar_width, pct_dict['train'], bar_width,
                         label=split_labels['train'], color=split_colors['train'], alpha=0.9, edgecolor='black', linewidth=0.5)
    rects_val = ax.bar(x, pct_dict['val'], bar_width,
                       label=split_labels['val'], color=split_colors['val'], alpha=0.9, edgecolor='black', linewidth=0.5)
    rects_test = ax.bar(x + bar_width, pct_dict['test'], bar_width,
                        label=split_labels['test'], color=split_colors['test'], alpha=0.9, edgecolor='black', linewidth=0.5)

    # Hiển thị số % trên đầu mỗi cột
    for rects in [rects_train, rects_val, rects_test]:
        for rect in rects:
            height = rect.get_height()
            if height > 1.5:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=90)

    # Tiêu đề gồm tên tập dữ liệu, số mẫu và số người ở mỗi tập
    title_text = (
        f"({chr(97 + idx)}) {ds_info['title']}\n"
        f"[Train: {counts['train']:,} ({subs['train']} subs) | "
        f"Val: {counts['val']:,} ({subs['val']} subs) | "
        f"Test: {counts['test']:,} ({subs['test']} subs)]"
    )
    ax.set_title(title_text, pad=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(COMMON_CLASSES)
    ax.set_ylim(0, 48)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if idx % 2 == 0:
        ax.set_ylabel('Tỷ lệ mẫu (%)')

    if idx == 0:
        ax.legend(loc='upper right', frameon=True)

plt.suptitle('Phân bố tỷ lệ 5 lớp hoạt động chung trên các tập Train / Val / Test',
             fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()

save_path = 'plots/dataset_class_distributions.png'
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"✅ Đã tạo thành công biểu đồ lưới 2x2 (kèm số người/subjects) tại: {save_path}")