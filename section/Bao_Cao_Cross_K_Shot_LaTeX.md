# Báo cáo Thực nghiệm Cross-K-Shot Domain Adaptation trong HAR với Học Tự Giám Sát (SSL)

Bản tài liệu này cung cấp mã nguồn **LaTeX chuẩn công báo khoa học** (Conference/Journal Style) kèm theo hướng dẫn **trực quan hóa dữ liệu (plotting guide)** để phân tích và trình bày kết quả thực nghiệm **Cross-K-Shot Adaptation** cho bài toán Nhận dạng Hoạt động Con người (Human Activity Recognition - HAR).

---

## 1. Hướng dẫn Trực quan hóa Số liệu (Visualization Guide & Python Code)

Để biến các bảng số liệu khô khan thành biểu đồ trực quan ấn tượng trong bài báo/báo cáo, bạn có thể chạy đoạn mã Python dưới đây để tự động xuất ra các file đồ thị dạng PNG/PDF chất lượng cao (300 DPI) lưu vào thư mục `plots/`.

```python
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
```

---

## 2. Mã nguồn LaTeX Báo cáo Chuyên nghiệp (Full LaTeX Source Code)

Dưới đây là toàn bộ đoạn mã LaTeX được thiết kế chuẩn mực, chuyên nghiệp để chèn trực tiếp vào báo cáo hoặc luận văn của bạn.

```latex
\section{Đánh giá Thích ứng Liên miền Cross-K-Shot (Cross-Domain K-Shot Evaluation)}
\label{sec:cross_k_shot_eval}

Trong kịch bản thực tế triển khai ứng dụng Nhận dạng Hoạt động Con người (HAR), việc thu thập dữ liệu gán nhãn tại miền đích (Target Domain) thường gặp khó khăn và tiêu tốn nhiều chi phí. Nhằm đánh giá khả năng chuyển giao tri thức của các mô hình Học Tự Giám Sát (SSL) khi dữ liệu miền đích bị giới hạn nghiêm trọng, chúng tôi tiến hành chuỗi thực nghiệm \textit{Cross-Domain $K$-Shot Adaptation}. Trong kịch bản này, mô hình được tiền huấn luyện SSL trên miền nguồn (Source Domain) không nhãn, sau đó được thích ứng trên miền đích với số lượng mẫu gán nhãn rất nhỏ $K \in \{10, 20, 50, 100\}$ mẫu mỗi lớp.

\subsection{Thiết lập Thực nghiệm và Phương pháp Đánh giá}

Thực nghiệm được tổng hợp từ $312$ bản ghi đánh giá độc lập trên $96$ cấu hình chuyển giao liên miền. Để đảm bảo tính khách quan và phản ánh đúng bản chất vật lý của tín hiệu cảm biến quán tính (IMU), các phân tích hiệu năng tổng thể được tập trung vào các cặp miền cùng cấp độ chuyển động thân người (Pocket-level transfer, bao gồm dữ liệu từ thiết bị đặt tại túi quần và thắt lưng như UCI-HAR, MotionSense, HHAR Phone). Các miền có sự chênh lệch lớn về đặc tính động lực học cơ thể (Wrist-level transfer từ cảm biến đeo cổ tay \texttt{hhar\_watch}) được tách riêng để phân tích điểm nghẽn vật lý trong Mục~\ref{subsec:physical_bottleneck}.

Chúng tôi đánh giá trên hai giao thức thích ứng (Adaptation Protocols):
\begin{enumerate}
    \item \textbf{Linear Probing (LP)}: Đóng băng hoàn toàn mạng trích xuất đặc trưng (\textit{Frozen Backbone}) và chỉ huấn luyện một bộ phân loại tuyến tính (Linear Classifier) ở tầng cuối. Giao thức này đo lường trực tiếp tính tổng quát và khả năng chuyển giao (\textit{Transferability}) của không gian biểu diễn thu được từ quá trình Pretrain SSL.
    \item \textbf{Full Fine-Tuning (FT)}: Cập nhật đồng thời toàn bộ trọng số của Backbone và Classifier bằng dữ liệu miền đích. Giao thức này phản ánh khả năng thích nghi (\textit{Adaptation Capacity}) của mô hình nhưng có rủi ro quá khớp (\textit{Overfitting}) khi $K$ nhỏ.
\end{enumerate}

Chỉ số đánh giá chính là \textbf{Macro F1-Score (\%)} kèm độ lệch chuẩn ($\text{Mean} \pm \text{Std}$) qua các đợt chạy độc lập nhằm đảm bảo tính cân bằng giữa các lớp hoạt động động (Walking, Upstairs, Downstairs) và các lớp hoạt động tĩnh (Sitting, Standing).

\subsection{Bảng Kết quả Tổng hợp và Phân tích Xu hướng $K$-Shot}

Bảng~\ref{tab:cross_k_shot_summary} trình bày kết quả chi tiết của các phương pháp SSL (SwAV Prototype, TS-TCC Contrastive, CrossHAR) kết hợp với các kiến trúc Backbone (Standard 1D-CNN và CNN-Transformer) qua các mức nhãn $K$-shot khác nhau tại các miền Pocket-level.

\begin{table}[h!]
\centering
\caption{Kết quả đánh giá Thích ứng Liên miền Cross-K-Shot (Macro F1-Score \% $\pm$ Std) trên các cặp dữ liệu Pocket-level. LP: Linear Probing, FT: Full Fine-Tuning.}
\label{tab:cross_k_shot_summary}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\textwidth}{lllXXXX}
\toprule
\textbf{SSL Method} & \textbf{Backbone} & \textbf{Protocol} & \textbf{10-shot} & \textbf{20-shot} & \textbf{50-shot} & \textbf{100-shot} \\
\midrule
\multirow{4}{*}{\textbf{Prototype (SwAV)}} 
  & \multirow{2}{*}{Standard 1D-CNN} & LP & $56.27 \pm 4.39$ & $61.07 \pm 3.17$ & $65.96 \pm 3.04$ & $70.72 \pm 2.61$ \\
  &                                   & FT & $\mathbf{65.24} \pm 4.84$ & $\mathbf{74.42} \pm 3.21$ & $\mathbf{79.79} \pm 2.88$ & $\mathbf{82.32} \pm 2.44$ \\
  \cmidrule(lr){2-7}
  & \multirow{2}{*}{CNN-Transformer}  & LP & $55.13 \pm 3.87$ & $60.94 \pm 3.33$ & $65.64 \pm 2.93$ & $69.09 \pm 2.50$ \\
  &                                   & FT & $65.63 \pm 5.06$ & $73.37 \pm 4.36$ & $77.59 \pm 2.91$ & $81.13 \pm 2.44$ \\
\midrule
\multirow{4}{*}{\textbf{Contrastive (TS-TCC)}} 
  & \multirow{2}{*}{Standard 1D-CNN} & LP & -- & -- & -- & $72.20 \pm 2.64$ \\
  &                                   & FT & -- & -- & -- & $82.05 \pm 2.58$ \\
  \cmidrule(lr){2-7}
  & \multirow{2}{*}{CNN-Transformer}  & LP & -- & -- & -- & $54.61 \pm 2.65$ \\
  &                                   & FT & -- & -- & -- & $79.65 \pm 2.86$ \\
\midrule
\multirow{4}{*}{\textbf{CrossHAR}} 
  & \multirow{2}{*}{Standard 1D-CNN} & LP & -- & -- & $69.81 \pm 1.63$ & $\mathbf{73.68} \pm 1.68$ \\
  &                                   & FT & -- & -- & $79.46 \pm 2.92$ & $82.11 \pm 2.17$ \\
  \cmidrule(lr){2-7}
  & \multirow{2}{*}{CNN-Transformer}  & LP & -- & -- & $58.28 \pm 2.25$ & $61.79 \pm 1.89$ \\
  &                                   & FT & -- & -- & $76.55 \pm 2.27$ & $80.33 \pm 2.77$ \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[h!]
\centering
% Đồ thị TikZ tự tạo đại diện cho diễn biến K-Shot Scaling Curves
\begin{tikzpicture}
\begin{axis}[
    width=0.85\textwidth,
    height=6.5cm,
    xlabel={Số lượng mẫu gán nhãn ở miền đích ($K$-shot)},
    ylabel={Macro F1-Score (\%)},
    xmin=5, xmax=105,
    ymin=50, ymax=86,
    xtick={10,20,50,100},
    ytick={55,60,65,70,75,80,85},
    legend pos=south east,
    ymajorgrids=true,
    grid style=dashed,
    title={\textbf{Diễn biến Hiệu năng Thích ứng theo Số lượng $K$-Shot (Prototype SSL)}},
    title style={font=\small\bfseries}
]

% Standard 1D-CNN Full FT
\addplot[color=blue, mark=*, line width=1.2pt]
    coordinates {(10,65.24)(20,74.42)(50,79.79)(100,82.32)};
    \addlegendentry{Standard 1D-CNN (Full FT)}

% Standard 1D-CNN Linear Probe
\addplot[color=blue, mark=square*, stroke draw=blue, fill=white, dashed, line width=1.0pt]
    coordinates {(10,56.27)(20,61.07)(50,65.96)(100,70.72)};
    \addlegendentry{Standard 1D-CNN (Linear Probe)}

% CNN-Transformer Full FT
\addplot[color=orange, mark=diamond*, line width=1.2pt]
    coordinates {(10,65.63)(20,73.37)(50,77.59)(100,81.13)};
    \addlegendentry{CNN-Transformer (Full FT)}

% CNN-Transformer Linear Probe
\addplot[color=orange, mark=triangle*, stroke draw=orange, fill=white, dashed, line width=1.0pt]
    coordinates {(10,55.13)(20,60.94)(50,65.64)(100,69.09)};
    \addlegendentry{CNN-Transformer (Linear Probe)}

\end{axis}
\end{tikzpicture}
\caption{Đồ thị mô phỏng sự gia tăng Macro F1-Score theo số lượng $K$-shot. Tốc độ tăng trưởng diễn ra mạnh nhất trong khoảng $K \in [10, 20]$ và bắt đầu bão hòa dần khi tiến tới $K=100$.}
\label{fig:k_shot_scaling}
\end{figure}

\subsection{Phân tích & So sánh Đa chiều (Comparative Evaluation)}

\subsubsection{So sánh giữa các Kiến trúc Backbone: Standard 1D-CNN vs. CNN-Transformer}

Một phát hiện thực nghiệm quan trọng là \textbf{kiến trúc nhẹ Standard 1D-CNN đạt hiệu năng tổng thể vượt trội và có độ ổn định cao hơn hẳn so với CNN-Transformer} trong hầu hết các cấu hình thích ứng miền few-shot.

\begin{itemize}
    \item \textbf{Tại $100$-shot (Full Fine-Tuning)}: Standard 1D-CNN đạt trung bình $82.16\%$ Macro F1, cao hơn $1.79\%$ so với CNN-Transformer ($80.37\%$).
    \item \textbf{Tại $100$-shot (Linear Probing)}: Khoảng cách này mở rộng lên tới \textbf{$10.25\%$} ($72.08\%$ ở Standard 1D-CNN so với $61.83\%$ ở CNN-Transformer).
\end{itemize}

\textbf{Giải thích bản chất cơ học và quán tính quy nạp (Inductive Bias)}:
Dữ liệu cảm biến gia tốc và con quay hồi chuyển 6 trục mang tính chất đặc trưng của chuỗi thời gian liên tục với các chu kỳ bước chân và biến thiên biên độ cục bộ. Mạng tích chập 1D-CNN với các Kernel cục bộ sở hữu quán tính quy nạp (\textit{Temporal Locality Inductive Bias}) rất mạnh. Cơ chế này tự động cô lập các hình thái bước chân (Gait Motifs) trong các cửa sổ thời gian ngắn mà không đòi hỏi số lượng mẫu dữ liệu quá lớn để học.

Ngược lại, mô-đun Self-Attention trong CNN-Transformer có số bậc tự do cực lớn. Khi số lượng mẫu gán nhãn ở miền đích bị hạn chế ($K \le 100$), cơ chế Attention rất dễ bị nhiễu bởi sự dịch chuyển phân phối liên miền (Domain Shift) và rơi vào hiện tượng quá khớp (Overfitting). Đặc biệt khi áp dụng giao thức Linear Probing (đóng băng Backbone), các trọng số Self-Attention không được điều chỉnh theo miền đích sẽ tạo ra các Feature Map bị lệch, khiến bộ phân loại tuyến tính không thể phân biệt chính xác các lớp.

\subsubsection{So sánh giữa các Phương pháp Học Tự Giám Sát (SSL Methods)}

Phân tích đối sánh giữa ba hàm mất mát học tự giám sát tại cấu hình chuẩn $100$-shot cho thấy:

\begin{enumerate}
    \item \textbf{Phương pháp Prototype (SwAV)}:
    Là phương pháp đạt trạng thái cân bằng tốt nhất giữa khả năng đại diện đặc trưng (Linear Probing) và tính thích ứng (Full Fine-Tuning). Với Backbone Standard 1D-CNN, Prototype đạt $82.32\%$ F1 ở giao thức FT và duy trì $70.72\%$ ở giao thức LP. Việc tối ưu hóa các nguyên mẫu cụm mềm (\textit{Soft-clustering Prototypes}) tạo ra cấu trúc hình học khoảng cách phân biệt rõ ràng giữa các nhóm hoạt động, giúp mô hình giữ vững khung tri thức ngay cả khi bị đóng băng.
    
    \item \textbf{Phương pháp Masked Signal Modeling (CrossHAR)}:
    Thể hiện năng lực vượt trội ở giao thức Linear Probing trên Backbone Standard 1D-CNN, đạt mức điểm cao nhất bài báo là \textbf{$73.68\%$} F1 tại $100$-shot (cao hơn Prototype $2.96\%$ và Contrastive $1.48\%$). Sự kết hợp giữa tác vụ tái tạo tín hiệu bị che (Masked Signal Reconstruction) và lực ép điều hòa tương phát giúp CrossHAR trích xuất được các đặc trưng bất biến miền (\textit{Domain-Invariant Features}) tốt hơn. Tuy nhiên, CrossHAR đòi hỏi Backbone tích chập cục bộ để phát huy hiệu quả và sụt giảm mạnh khi kết hợp với CNN-Transformer ($61.79\%$ LP).
    
    \item \textbf{Phương pháp Contrastive (TS-TCC)}:
    Phụ thuộc nghiêm trọng vào giao thức Full Fine-Tuning. Với CNN-Transformer, Contrastive ghi nhận mức tăng kỷ lục lên tới \textbf{$+25.04\%$} F1 khi chuyển từ LP ($54.61\%$) sang FT ($79.65\%$). Điều này chứng minh rằng biểu diễn tương phát theo cặp mẫu (Pairwise Contrastive Views) tạo ra các đặc trưng có tiềm năng cao nhưng đòi hỏi phải tinh chỉnh toàn bộ mạng mới có thể căn chỉnh phù hợp với phân phối dữ liệu ở miền đích.
\end{enumerate}

\subsubsection{Phân tích Giao thức Thích ứng: Full Fine-Tuning (FT) vs. Linear Probing (LP)}

Hiệu năng chênh lệch $\Delta_{\text{FT}-\text{LP}} = \text{F1}_{\text{FT}} - \text{F1}_{\text{LP}}$ phản ánh lượng thông tin mà mô hình phải tự điều chỉnh lại cấu trúc trích xuất đặc trưng nhằm vượt qua rào cản Domain Shift.

\begin{itemize}
    \item Trên Backbone Standard 1D-CNN, khoảng cách $\Delta_{\text{FT}-\text{LP}}$ duy trì ở mức vừa phải: $+8.43\%$ (CrossHAR), $+9.85\%$ (Contrastive), và $+11.60\%$ (Prototype). Mức chênh lệch nhỏ này chứng tỏ không gian biểu diễn của 1D-CNN đã tương đối sẵn sàng cho việc phân loại ở miền đích mà không phụ thuộc quá nhiều vào việc cập nhật lại toàn bộ trọng số mạng.
    \item Trên Backbone CNN-Transformer, khoảng cách này bị nới rộng mạnh mẽ: $+12.04\%$ (Prototype), $+18.54\%$ (CrossHAR), và đặc biệt là $+25.04\%$ (Contrastive). 
\end{itemize}

\textit{Khuyến nghị kỹ thuật}: Khi tài nguyên tính toán bị hạn chế hoặc thiết bị biên đòi hỏi đóng băng mô hình, bộ đôi \textbf{Standard 1D-CNN + CrossHAR/Prototype} là sự lựa chọn tối ưu nhất. Khi có đủ tài nguyên tính toán để fine-tune toàn mạng, \textbf{Standard 1D-CNN + Prototype} mang lại F1-Score cao nhất ($82.32\%$).

\subsubsection{Tác động của Số lượng Shot ($K \in \{10, 20, 50, 100\}$)}

Quá trình gia tăng dữ liệu nhãn ở miền đích thể hiện rõ hiệu ứng biên giảm dần (\textit{Diminishing Returns}):
\begin{itemize}
    \item \textbf{Giai đoạn 1 ($10 \rightarrow 20$ shot)}: Hiệu năng FT của Prototype tăng vọt từ $65.24\%$ lên $74.42\%$ (tăng \textbf{$+9.18\%$} F1). Chỉ cần bổ sung thêm $10$ mẫu mỗi lớp, mô hình đã giải quyết được phần lớn sự mơ hồ giữa các lớp tĩnh (Sitting vs. Standing).
    \item \textbf{Giai đoạn 2 ($20 \rightarrow 50$ shot)}: F1-Score tăng tiếp $+5.37\%$ đạt $79.79\%$.
    \item \textbf{Giai đoạn 3 ($50 \rightarrow 100$ shot)}: Tốc độ tăng trưởng chậm lại rõ rệt, chỉ tăng thêm $+2.53\%$ để đạt tiệm cận $82.32\%$.
\end{itemize}

Kết quả này cho thấy tại mốc $K=50$ shot, mô hình đã tiếp cận được ngưỡng bão hòa thích ứng miền đối với các hoạt động cơ bản. Việc tiếp tục thu thập thêm nhãn từ $50$ lên $100$ shot mang lại cải thiện không quá lớn so với chi phí dán nhãn bỏ ra.

\subsection{Phân tích Nút thắt Vật lý & Động học Cảm biến (Domain Bottleneck)}
\label{subsec:physical_bottleneck}

Một quan sát thực nghiệm có ý nghĩa thực tiễn quan trọng là sự sụt giảm hiệu năng nghiêm trọng khi chuyển giao từ miền cơ thể (Pocket/Waist) sang miền thiết bị đeo cổ tay (\texttt{hhar\_watch}).

\begin{itemize}
    \item \textbf{Chuyển giao Pocket-to-Pocket (UCI-HAR $\leftrightarrow$ MotionSense)}: Đạt hiệu năng trung bình cao từ $70.38\%$ đến $82.32\%$ F1 tại $100$-shot.
    \item \textbf{Chuyển giao Pocket-to-Wrist (MotionSense $\rightarrow$ \texttt{hhar\_watch})}: Hiệu năng sụt giảm nghiêm trọng xuống trung bình chỉ còn \textbf{$33.70\%$} F1 trên mọi phương pháp SSL và kiến trúc Backbone.
\end{itemize}

\textbf{Phân tích bản chất Động lực học Cơ thể (Biomechanics Analysis)}:
Cảm biến đặt tại túi quần hoặc thắt lưng phản ánh trực tiếp chuyển động của trọng tâm cơ thể (\textit{Center of Mass}) và góc xoay của đùi. Các véc-tơ gia tốc trọng trường $g$ và chuyển động tịnh tiến có tính nhất quán cao giữa các hoạt động đi bộ và leo cầu thang.

Ngược lại, cảm biến đeo tại cổ tay (\texttt{hhar\_watch}) chịu tác động tổng hòa của hai chuyển động độc lập: (1) chuyển động di chuyển toàn thân và (2) chuyển động tự do của cánh tay (vẩy tay, khoanh tay, cầm điện thoại, thao tác cá nhân). Tín hiệu cổ tay chứa nhiễu tần số cao với biên độ xoay góc lớn, làm triệt tiêu tính bất biến miền mà mô hình SSL học được từ dữ liệu túi quần. Kết quả này khẳng định rằng \textbf{nút thắt lớn nhất trong Cross-Domain HAR không chỉ nằm ở thuật toán loss mà xuất phát từ sự khác biệt bản chất sinh học trong vị trí gán cảm biến}.

\subsection{Kết luận & Quy tắc Thiết kế Hệ thống (Design Guidelines)}

Từ các phân tích thực nghiệm trên, chúng tôi rút ra 4 quy tắc vàng cho việc thiết kế hệ thống Few-Shot Cross-Domain HAR:

\begin{enumerate}
    \item \textbf{Ưu tiên Backbone Tích chập Cục bộ (Standard 1D-CNN)}: Trong điều kiện ít nhãn miền đích, 1D-CNN luôn là sự lựa chọn an toàn, ổn định và đạt hiệu năng cao hơn Transformer nhờ quán tính quy nạp thời gian mạnh.
    \item \textbf{Lựa chọn Phương pháp SSL Phù hợp với Giao thức}:
    \begin{itemize}
        \item Nếu buộc phải đóng băng mô hình (Linear Probing), sử dụng \textbf{CrossHAR} hoặc \textbf{Prototype (SwAV)}.
        \item Nếu được phép tinh chỉnh toàn mạng (Full Fine-Tuning), chọn \textbf{Prototype (SwAV)}.
    \end{itemize}
    \item \textbf{Điểm tối ưu Chi phí Nhãn ($K=50$ shot)}: Lựa chọn $K=50$ mẫu/lớp là điểm thỏa hiệp tối ưu giữa chi phí gán nhãn và hiệu năng nhận dạng ($\approx 97\%$ hiệu năng so với $100$-shot).
    \item \textbf{Xử lý Khác biệt Vị trí Đeo Cảm biến}: Không sử dụng mô hình pretrain từ cảm biến thân người (Pocket/Waist) để đánh giá trực tiếp trên cảm biến cổ tay (Wrist) mà không có mô-đun thích ứng khoảng cách phân phối đặc thù.
\end{enumerate}
```

---

## 3. Tóm tắt các Con số Trọng tâm để Trích dẫn nhanh (Quick Citation Cheat-Sheet)

Dưới đây là bảng tra cứu nhanh các chỉ số F1-Score chính xác để trích dẫn vào các phần thảo luận, tóm tắt hoặc kết luận của bài báo:

| Cấu hình Thực nghiệm | Full Fine-Tuning (FT) | Linear Probing (LP) | Chênh lệch $\Delta_{\text{FT}-\text{LP}}$ | Ghi chú & Đặc điểm chính |
| :--- | :---: | :---: | :---: | :--- |
| **Standard 1D-CNN + Prototype (100-shot)** | **82.32%** | **70.72%** | $+11.60\%$ | **Best Overall FT Score** (Cân bằng & ổn định nhất) |
| **Standard 1D-CNN + CrossHAR (100-shot)** | **82.11%** | **73.68%** | **$+8.43\%$** | **Best Frozen Feature (LP)** (Biểu diễn đóng băng tốt nhất) |
| **Standard 1D-CNN + TS-TCC (100-shot)** | **82.05%** | **72.20%** | $+9.85\%$ | Đạt hiệu năng FT tương đương Prototype |
| **CNN-Transformer + Prototype (100-shot)** | **81.13%** | **69.09%** | $+12.04\%$ | Backbone Transformer tốt nhất |
| **CNN-Transformer + CrossHAR (100-shot)** | **80.33%** | **61.79%** | $+18.54\%$ | Giảm hiệu năng LP đáng kể trên Transformer |
| **CNN-Transformer + TS-TCC (100-shot)** | **79.65%** | **54.61%** | **$+25.04\%$** | Phụ thuộc FT cao nhất (LP rất thấp) |
| **Prototype Standard (10-shot FT $\rightarrow$ 100-shot FT)** | $65.24\% \rightarrow 82.32\%$ | $56.27\% \rightarrow 70.72\%$ | -- | Tăng mạnh từ 10$\rightarrow$20 shot, bão hòa từ 50$\rightarrow$100 shot |
| **Nút thắt Miền Đích Wrist (\texttt{hhar\_watch})** | **33.70%** (vs 70.38% Pocket) | -- | $-36.68\%$ | Sụt giảm do khác biệt động lực học cổ tay vs thắt lưng |
