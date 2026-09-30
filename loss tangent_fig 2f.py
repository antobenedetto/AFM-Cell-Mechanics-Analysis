from matplotlib.ticker import LogFormatter, FormatStrFormatter, NullFormatter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# ---------- 全局画图风格 ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],

    "axes.labelweight": "bold",
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,

    "axes.labelsize": 14,
    "axes.titlesize": 14,

    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.8),

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
})

# ---------- 选单个文件 ----------
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select one txt file",
    filetypes=[("Text files", "*.txt")]
)
if not file_path:
    raise SystemExit("No file selected.")

file_path = Path(file_path)

fig_dir = file_path.parent / "tangent figure"
fig_dir.mkdir(parents=True, exist_ok=True)

# ---------- 读取单条曲线 ----------
def read_single_curve(file_path, skiprows=8):
    data = np.loadtxt(file_path, skiprows=skiprows)
    f, tan = data[:, 0], data[:, 2]
    mask = np.isfinite(f) & np.isfinite(tan) & (f > 0) & (tan > 0)
    f, tan = f[mask], tan[mask]
    if f.size == 0:
        raise ValueError("No valid (f, tanδ) after filtering.")
    idx = np.argsort(f)
    return f[idx], tan[idx]

# ---------- 导出到 txt ----------
def save_curve_txt(out_path, freq, tan):
    arr = np.column_stack([freq, tan])
    header = "Frequency(Hz)\tLoss_tangent"
    np.savetxt(
        out_path,
        arr,
        fmt="%.6f",
        delimiter="\t",
        header=header,
        comments=""
    )

# ---------- 读取数据 ----------
F, T = read_single_curve(file_path, skiprows=8)

# ---------- 输出到 txt ----------
txt_path = fig_dir / f"{file_path.stem}_tangent.txt"
save_curve_txt(txt_path, F, T)

# ---------- 画图 ----------
fig, ax = plt.subplots()
ax.set_xscale("log")

ax.plot(
    F, T,
    "o-",
    markersize=3.8,
    linewidth=1.5,
    color="#1fb421d2",
    markerfacecolor="#1fb421d2",
    markeredgecolor="#1fb421d2",
    alpha=0.9,
    zorder=3
)

# ---------- x 轴范围 ----------
if F.size:
    xmin = F.min()
    xmax = F.max()
    log_min = np.log10(xmin)
    log_max = np.log10(xmax)
    pad = 0.03 * (log_max - log_min) if log_max > log_min else 0.1
    xmin_pad = 10 ** (log_min - pad)
    xmax_pad = 10 ** (log_max + pad)
    ax.set_xlim(xmin_pad, xmax_pad)
    ax.autoscale(enable=False, axis="x")

# ---------- y 轴范围 ----------
if T.size:
    ymin = 0
    ymax = np.max(T) * 1.08
    ax.set_ylim(ymin, ymax)

# ---------- 坐标轴样式 ----------
ax.tick_params(which="both")
ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_xlabel("Frequency (Hz)", fontweight="bold")
ax.set_ylabel("Loss tangent (E\"/E')", fontweight="bold")

for lab in ax.get_xticklabels():
    lab.set_fontweight("bold")
for lab in ax.get_yticklabels():
    lab.set_fontweight("bold")

for spine in ax.spines.values():
    spine.set_visible(True)

ax.axhline(1.0, color="grey", lw=1.0, ls="--", alpha=0.4)

fig.tight_layout()

png_path = fig_dir / f"{file_path.stem}_tan.png"
plt.savefig(png_path, dpi=600, transparent=True)
plt.show()