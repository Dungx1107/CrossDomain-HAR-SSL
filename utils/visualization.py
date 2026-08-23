"""
===============================================================================
MỤC ĐÍCH:
    - Đọc file CSV metrics để vẽ đồ thị Loss & Macro F1 qua các Epoch.
    - Vẽ Heatmap Ma trận nhầm lẫn (Confusion Matrix Heatmap %).
    - Vẽ biểu đồ t-SNE không gian biểu diễn đặc trưng.
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_learning_curves(csv_path, save_path="learning_curves.png"):
    """Vẽ biểu đồ Loss & Macro F1 từ metrics.csv."""
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='#d62728', lw=2, linestyle='--')
    axes[0].set_title('Cross-Entropy Loss qua các Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Macro F1 Curve
    axes[1].plot(df['epoch'], df['train_macro_f1'], label='Train Macro F1', color='#2ca02c', lw=2)
    axes[1].plot(df['epoch'], df['val_macro_f1'], label='Val Macro F1', color='#ff7f0e', lw=2, linestyle='--')
    axes[1].set_title('Macro F1-Score (Thước đo chính)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png", title="Confusion Matrix"):
    """Vẽ Heatmap Confusion Matrix chuẩn hóa theo %."""
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Tỷ lệ nhận dạng đúng (%)'}
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=10, fontweight='bold')
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_latent_space_tsne(embeddings, labels, class_names=None, title="t-SNE Embedding Space", save_path=None):
    """Chiếu giảm chiều 128D xuống 2D bằng t-SNE."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if class_names else f"Class {lbl}"
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], label=name, alpha=0.6, s=20)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()