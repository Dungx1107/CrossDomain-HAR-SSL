"""
===============================================================================
MODULE: FEW-SHOT REPORTING (ĐÁNH GIÁ + GHI BÁO CÁO)
===============================================================================
- evaluate_model      : chạy mô hình trên test loader, trả về metric của 1 seed.
- save_k_shot_report  : gom kết quả các seed -> ghi detailed_all_seeds.json,
                        summary_*.json và aggregated_confusion_matrix.png.
Định dạng file JSON giữ nguyên như bản cũ của run_few_shot.py.
===============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from engines.evaluation.evaluator import predict


def evaluate_model(model, loader, device: str, class_names: List[str]) -> Dict[str, Any]:
    """Đánh giá mô hình trên loader. Các metric tính theo thang 0-100 (trừ kappa)."""
    y_true, y_pred = predict(model, loader, device)

    # Cố định labels để báo cáo/ma trận luôn đủ số lớp, kể cả khi test thiếu 1 lớp
    label_ids = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=class_names,
        output_dict=True, zero_division=0
    )

    return {
        "accuracy": float((y_true == y_pred).mean() * 100),
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "macro_precision": report["macro avg"]["precision"] * 100,
        "macro_recall": report["macro avg"]["recall"] * 100,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "per_class_f1": {c: round(report[c]["f1-score"] * 100, 2) for c in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids),
    }


def save_k_shot_report(
        save_dir: Union[str, Path],
        display_name: str,
        k_shot_name: str,
        k: int,
        seed_results: Dict[int, Dict[str, Any]],
        class_names: List[str],
) -> Dict[str, Any]:
    """
    Ghi báo cáo cho một cặp (k_shot, protocol) từ kết quả của nhiều seed.
    seed_results[seed] = dict trả về từ evaluate_model + khóa "train_samples_used".
    Trả về dict summary (để in log).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seed_results)

    acc_list = [seed_results[s]["accuracy"] for s in seeds]
    f1_list = [seed_results[s]["macro_f1"] for s in seeds]
    prec_list = [seed_results[s]["macro_precision"] for s in seeds]
    rec_list = [seed_results[s]["macro_recall"] for s in seeds]
    aggregated_cm = sum(seed_results[s]["confusion_matrix"] for s in seeds)

    # 1. Chi tiết từng seed
    detailed = {
        f"seed_{s}": {
            "train_samples_used": r["train_samples_used"],
            "metrics": {
                "accuracy": round(r["accuracy"], 2),
                "macro_f1": round(r["macro_f1"], 2),
                "cohen_kappa": round(r["kappa"], 4),
            },
            "per_class_f1": r["per_class_f1"],
            "confusion_matrix": r["confusion_matrix"].tolist(),
        }
        for s, r in seed_results.items()
    }
    with open(save_dir / "detailed_all_seeds.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=4)

    # 2. Summary trung bình
    summary = {
        "protocol": display_name, "k_shot_name": k_shot_name, "k_value": k,
        "num_seeds_tested": len(seeds),
        "overall_metrics": {
            "accuracy_mean": round(np.mean(acc_list), 2), "accuracy_std": round(np.std(acc_list), 2),
            "macro_f1_mean": round(np.mean(f1_list), 2), "macro_f1_std": round(np.std(f1_list), 2),
            "macro_precision_mean": round(np.mean(prec_list), 2), "macro_precision_std": round(np.std(prec_list), 2),
            "macro_recall_mean": round(np.mean(rec_list), 2), "macro_recall_std": round(np.std(rec_list), 2),
        },
        "per_class_f1_mean": {
            c: round(np.mean([seed_results[s]["per_class_f1"][c] for s in seeds]), 2) for c in class_names
        },
        "seed_quick_lookup": {
            str(s): {"accuracy": round(r["accuracy"], 2), "macro_f1": round(r["macro_f1"], 2)}
            for s, r in seed_results.items()
        },
        "aggregated_confusion_matrix": aggregated_cm.tolist(),
    }
    with open(save_dir / f"summary_{display_name}_{k_shot_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # 3. Ảnh confusion matrix trung bình qua các seed
    plt.figure(figsize=(8, 6))
    sns.heatmap(aggregated_cm / len(seeds), annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Average Confusion Matrix ({k_shot_name} | {display_name})", fontsize=12, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(save_dir / "aggregated_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return summary
