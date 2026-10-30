"""
===============================================================================
UTILS: VẼ BIỂU ĐỒ HUẤN LUYỆN
===============================================================================
- PlottingUtils: class gom các hàm vẽ biểu đồ theo từng phương pháp.
  + plot_tstcc_loss_history: TS-TCC (loss_total / loss_tc / loss_cc + tỉ lệ TC/CC).
  + (sau này) plot_crosshar_loss_history, plot_supervised_loss_history, ...
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt


class PlottingUtils:
    """Gom các hàm vẽ biểu đồ huấn luyện. Gọi trực tiếp qua class (static method)."""

    @staticmethod
    def plot_tstcc_loss_history(history: list, save_path: Path, title: str):
        """
        Vẽ biểu đồ loss + tỉ lệ TC/CC theo epoch cho TS-TCC.

        Args:
            history: list các dict {"epoch", "loss_total", "loss_tc", "loss_cc", "grad_norm"}
            save_path: đường dẫn file PNG đầu ra
            title: tiêu đề biểu đồ
        """
        epochs = [h["epoch"] for h in history]
        loss_total = [h["loss_total"] for h in history]
        loss_tc = [h["loss_tc"] for h in history]
        loss_cc = [h["loss_cc"] for h in history]
        tc_cc_ratio = [tc / (cc + 1e-8) for tc, cc in zip(loss_tc, loss_cc)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax1.plot(epochs, loss_total, label="Total Loss", color="#1f77b4", linewidth=2.0)
        ax1.plot(epochs, loss_tc, label="Temporal Contrasting (TC)", color="#ff7f0e", linestyle="--")
        ax1.plot(epochs, loss_cc, label="Contextual Contrasting (CC)", color="#2ca02c", linestyle=":")
        ax1.set_title(f"Pretraining Convergence: {title}", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Loss Value", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=9)

        ax2.plot(epochs, tc_cc_ratio, label="Ratio (TC / CC)", color="#9467bd", linewidth=1.5)
        ax2.axhline(y=1.0, color="gray", linestyle="-.", alpha=0.6)
        ax2.set_title("Balance Dynamics: TC / CC Ratio", fontsize=11)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Ratio", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"📈 Đã lưu biểu đồ phân tích loss tại: {save_path}")

    # -------------------------------------------------------------------------
    # Sau này thêm các hàm plot khác ở đây, ví dụ:
    #
    # @staticmethod
    # def plot_crosshar_loss_history(history: list, save_path: Path, title: str):
    #     ...
    #
    # @staticmethod
    # def plot_supervised_loss_history(history: list, save_path: Path, title: str):
    #     ...
    # -------------------------------------------------------------------------