
# This script processes NHF files from a batch of AFM measurements, specifically focusing on the "Wait 1" segment. It extracts time and deflection data, converts deflection to force using the spring constant, and fits a bi-exponential model to the force relaxation curves. The results are saved in a tab-separated text file, and plots of the data and fits can be generated and saved as PNG files.
# --by Jiajia Wei, 2026-01


import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from nanosurf.lib.util import nhf_reader
import tkinter as tk
from tkinter import filedialog
import scipy.optimize as opt
from datetime import datetime

# ==============================
# User options
# ==============================
deflection_sensitivity = 0  # m/V, updated from each file
spring_constant = 0         # N/m, updated from each file

y_mode = "force"            # "force" (nN) vs Time; "deflection" (nm) vs Time for plotting
plot_all_points = True      # True: overlay all acquired points; False: only block at point_index
point_index = 0             # used only when plot_all_points=False
thin_alpha = 0.7            # alpha for multi-trace overlay
save_png = False            # save per-file plot PNG
overlay_fit = True          # draw fitted curve on top of data
save_fit_plots_per_block = False  # if True, saves one figure per block with fit
downsample_every = 1        # e.g., 5 -> use every 5th point for fitting (speeds up)
import matplotlib as mpl
# ==============================
# Global plot style (house style)
# ==============================


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "figure.figsize": (3.3, 2.8),
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    
    "lines.markersize": 6,
    "axes.labelweight": "bold",
    
    "xtick.major.width": 1,
    "ytick.major.width": 1,

    
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def _prettify_axes(ax):
    ax.tick_params(axis='both', which='major', length=5, width=2)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

# ==============================
# Data structure
# ==============================
class MeasurementData:
    ch_time_wait = None
    ch_deflection_wait = None  # stored as meters after conversion
    data_offsets_wait = None
    datapoints_acquired_wait = None

sample_data = MeasurementData()
wait_segment_exists = True

# ==============================
# NHF helpers
# ==============================
def get_offset_datapoints(segment, time_channel):
    """Treat entire Wait 1 as one continuous block."""
    n = int(len(time_channel.dataset))
    return np.array([0], dtype=int), np.array([n], dtype=int)

def fix_deflection_unit_to_meters(channel, measurement):
    """
    Convert deflection channel to meters.
    Always uses calibration from the current file.
    """
    global deflection_sensitivity, spring_constant
    ds = channel.dataset.astype(float, copy=False)

    deflection_sensitivity = measurement.attribute['spm_probe_calibration_deflection_sensitivity']  # m/V, current file
    spring_constant = measurement.attribute['spm_probe_calibration_spring_constant']  # N/m, current file

    if channel.unit == 'V':
        ds = ds * deflection_sensitivity
    elif channel.unit == 'm':
        pass
    elif channel.unit == 'N':
        k = spring_constant
        if not k or not np.isfinite(k):
            k = measurement.attribute.get('spm_probe_calibration_spring_constant', np.nan)
        if not k or not np.isfinite(k):
            raise ValueError("Spring constant unavailable; cannot convert N -> m.")
        ds = ds / k
    else:
        print(f"{channel.name}: unknown unit '{channel.unit}', values left unconverted.")

    channel.dataset = ds  # store as meters

def import_data(file_path):
    """Load Wait 1 time + deflection(m) + block metadata into sample_data."""
    nhf_file = nhf_reader.NHFFileReader(verbose=True)
    if not nhf_file.read(file_path):
        print(f"Could not read file: {file_path}")
        return False

    measurement_name = nhf_file.measurement_name(0)
    measurement = nhf_file.measurement[measurement_name]

    global wait_segment_exists
    wait_segment_exists = True
    try:
        segment = measurement.segment["Wait 1"]
    except KeyError:
        wait_segment_exists = False
        print("No 'Wait 1' segment found. Skipping.")
        return False

    data = sample_data
    data.ch_time_wait = segment.read_channel('Time')
    data.ch_deflection_wait = segment.read_channel('Deflection')

    # Ensure deflection in meters
    fix_deflection_unit_to_meters(data.ch_deflection_wait, measurement)

    # Block slicing info
    off, pts = get_offset_datapoints(segment, data.ch_time_wait)
    data.data_offsets_wait = off
    data.datapoints_acquired_wait = pts
    return True

def slice_wait_block(i):
    """Return time (s), deflection (m), and force (nN) arrays for block i."""
    start = int(sample_data.data_offsets_wait[i])
    end = start + int(sample_data.datapoints_acquired_wait[i])

    # ---- Time ----
    t = np.asarray(sample_data.ch_time_wait.dataset[start:end], dtype=np.float64)
    if len(t):
        t = t - t[0]  # start from 0 s

    # ---- Deflection ----
    d_m = np.asarray(sample_data.ch_deflection_wait.dataset[start:end], dtype=np.float64)  # in meters

    # 可选：过滤明显异常的挠度点（>50 µm 一般是不合理的）
    d_m[np.abs(d_m) > 5e-5] = np.nan

    # ---- Force conversion ----
    k = float(spring_constant)
    F_nN = np.full_like(d_m, np.nan, dtype=np.float64)

    # 只对有限值做乘法，避免 inf 或 overflow 警告
    mask = np.isfinite(d_m) & np.isfinite(k)
    if np.isfinite(k) and k != 0.0:
        F_nN[mask] = d_m[mask] * k * 1e9  # m * (N/m) = N → nN
    else:
        print("[WARN] Invalid spring constant; returning NaN force.")

    return t, d_m, F_nN

# ==============================
# Force preprocessing: make non-negative
# ==============================
def make_force_nonnegative(F, strategy="shift_then_clip"):
    """
    strategy:
      - "shift_then_clip": shift upward so min(F)=0, then clip tiny negatives to 0
      - "clip": directly clip negatives to 0
    Returns: F_proc, offset_added (nN)
    """
    F = np.array(F, dtype=float)
    offset = 0.0
    if strategy == "shift_then_clip":
        mn = np.nanmin(F)
        if np.isfinite(mn) and mn < 0:
            offset = -mn
            F = F + offset
        F = np.maximum(F, 0.0)
    elif strategy == "clip":
        F = np.maximum(F, 0.0)
    else:
        raise ValueError("Unknown strategy")
    return F, float(offset)

# ==============================
# Fitting helpers
# ==============================
def bi_exp(t, F0, F1, tau1, F2, tau2):
    tau1 = np.clip(tau1, 1e-6, 1e6)
    tau2 = np.clip(tau2, 1e-6, 1e6)
    return F0 + F1*np.exp(-t/tau1) + F2*np.exp(-t/tau2)

def rmse_percent(y, yfit):
    if len(y) == 0:
        return np.nan
    rng = np.nanmax(y) - np.nanmin(y)
    if rng == 0:
        return 0.0
    return 100.0*np.sqrt(np.nanmean((y - yfit)**2))/rng

def r2_score(y, yfit):
    if len(y) == 0:
        return np.nan
    ss_res = np.nansum((y - yfit)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def initial_guess_and_bounds(t, F):
    if len(F) < 5:
        # lower bounds >=0, taus >0; upper bounds large
        return [np.nan]*5, (np.zeros(5), np.array([np.inf, np.inf, 1e6, np.inf, 1e6]))

    F_start = np.nanmedian(F[:max(3, len(F)//50)])
    F_end   = np.nanmedian(F[-max(3, len(F)//50):])
    F0_guess = max(F_end, 0.0)
    A_guess  = max(F_start - F_end, 0.0)
    F1_guess = 0.7 * A_guess
    F2_guess = 0.3 * A_guess

    tspan = np.nanmax(t) - np.nanmin(t)
    tspan = tspan if np.isfinite(tspan) and tspan > 0 else 1.0
    tau1_guess = max(0.05 * tspan, 1e-4)
    tau2_guess = max(0.50 * tspan, 1e-3)

    p0 = [F0_guess, F1_guess, tau1_guess, F2_guess, tau2_guess]
    lb = [0.0, 0.0, 1e-6, 0.0, 1e-6]   # non-negative amplitudes, taus > 0
    ub = [np.inf, np.inf, 1e6, np.inf, 1e6]
    return p0, (lb, ub)

def fit_block(t, F_nN):
    """Fit bi-exponential to (t, F) in nN with non-negative preprocessing."""
    # Clean / downsample
    mask = np.isfinite(t) & np.isfinite(F_nN)
    t = t[mask]
    F = F_nN[mask]

    if downsample_every and downsample_every > 1:
        t = t[::downsample_every]
        F = F[::downsample_every]

    if len(t) < 10:
        return None

    # Non-negative preprocessing
    F, offset_added = make_force_nonnegative(F, strategy="shift_then_clip")

    # Sort by time
    idx = np.argsort(t)
    t = t[idx]
    F = F[idx]

    # Initial guess & bounds
    p0, bounds = initial_guess_and_bounds(t, F)

    try:
        popt, pcov = opt.curve_fit(
            bi_exp, t, F, p0=p0, bounds=bounds, maxfev=20000
        )
        F_fit = bi_exp(t, *popt)
        metrics = {
            "RMSE_percent": rmse_percent(F, F_fit),
            "R2": r2_score(F, F_fit),
            "n_pts": len(t),
            "F_start": float(F[0]),
            "F_end": float(F[-1]),
            "t0": float(t[0]),
            "t_end": float(t[-1]),
            "offset_added": offset_added,   # nN
        }
        return popt, pcov, F_fit, metrics, t, F  # return preprocessed (t, F)
    except Exception as e:
        return {"error": str(e)}

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select NHF Files",
        filetypes=[("NHF files","*.nhf"), ("All files", "*.*")]
    )
    if not file_paths:
        sys.exit("No files selected.")

    out_dir = pathlib.Path(file_paths[0]).parent
    out_txt = out_dir / f"Tau re.txt"  ##change 1

    # ------------------------------
    # Header for the TSV
    # ------------------------------
    header_cols = [
        "file", "block_index",
        "F0_nN", "F1_nN", "tau1_s", "F2_nN", "tau2_s",
        "RMSE_percent(%)", "offset_added_nN"
    ]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\t".join(header_cols) + "\n")

    for fp in map(pathlib.Path, file_paths):
        if not import_data(str(fp)) or not wait_segment_exists:
            continue

        n_blocks = int(len(sample_data.datapoints_acquired_wait))
        if n_blocks == 0:
            print(f"{fp.name}: no Wait blocks found.")
            continue

        # Per-file combined plot (all blocks)
        fig_all, ax_all = plt.subplots()
        _prettify_axes(ax_all)
        ax_all.set_xlabel("Time (s)")
        ax_all.set_ylabel(r'$F_{wait}$ (nN)' if y_mode.lower() == "force" else "Deflection (nm)")
        
        for spine in ax_all.spines.values():
          spine.set_visible(True)
          spine.set_linewidth(1.5)

        for i in range(n_blocks):
            t, d_m, F_nN = slice_wait_block(i)
            if not len(t):
                continue

            # Fit on force (preprocessing inside)
            res = fit_block(t, F_nN)
            if isinstance(res, dict) or res is None:
                popt = [np.nan]*5
                metrics = {"RMSE_percent": np.nan, "offset_added": 0.0}
                t_fit, F_fit, F_used = None, None, None
                # Raw plot fallback
                if y_mode.lower() == "force":
                    ax_all.plot(t, F_nN,'k.', markersize=3,  alpha=thin_alpha,label="Relaxation curve")
                else:
                    ax_all.plot(t, d_m*1e9, 'k.', markersize=3,  alpha=thin_alpha,label="Deflection")
            else:
                popt, pcov, F_fit, metrics, t_used, F_used = res
                # Plot preprocessed raw (non-negative)
                if y_mode.lower() == "force":
                    ax_all.plot(t_used, F_used,  'k.', markersize=3,alpha=thin_alpha,label="Relaxation curve")
                else:
                    ax_all.plot(t, d_m*1e9, 'k.', markersize=3,alpha=thin_alpha,label="Deflection")

                # Overlay fit
                if overlay_fit and (t_used is not None):
                    ax_all.plot(t_used, F_fit, alpha=0.9,linewidth=2,color='orange',label="Maxwell Fit")
            ax_all.legend(frameon=False,loc='best')
            
            # Write row with 3 decimals
            def fmt3(x):
                return f"{x:.3f}" if (x is not None and np.isfinite(x)) else "nan"

            row = [
                fp.name, str(i),
                fmt3(popt[0]),  # F0_nN
                fmt3(popt[1]),  # F1_nN
                fmt3(popt[2]),  # tau1_s
                fmt3(popt[3]),  # F2_nN
                fmt3(popt[4]),  # tau2_s
                fmt3(metrics.get("RMSE_percent", np.nan)),
                fmt3(metrics.get("offset_added", 0.0)),
            ]
            with open(out_txt, "a", encoding="utf-8") as f:
                f.write("\t".join(row) + "\n")

        # Finalize plot
        #title = f"{fp.name} — Wait: {'Force (nN)' if y_mode.lower()=='force' else 'Deflection (nm)'} vs Time (n={n_blocks})"
        #ax_all.set_title(title)

        # y-baseline at 0 (for display only)
        if y_mode.lower() == "force":
            ax_all.relim(); ax_all.autoscale_view(scalex=False, scaley=True)
            y0, y1 = ax_all.get_ylim()
            ax_all.set_ylim(bottom=0, top=max(y1*1.08, 1e-6))
        elif y_mode.lower() == "deflection":
            ax_all.relim(); ax_all.autoscale_view(scalex=False, scaley=True)
            y0, y1 = ax_all.get_ylim()
            ax_all.set_ylim(bottom=0, top=max(y1*1.08, 1e-6))

        fig_all.tight_layout()
        if save_png:
            suffix = "_wait_force_fit.png" if y_mode.lower() == "force" else "_wait_deflection.png"
            out_png = fp.with_suffix("").as_posix() + suffix
            fig_all.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
            print(f"[Saved] {out_png}")

        
        save_path = pathlib.Path(file_paths[0]).parent / "Maxwell_Fits2.png"
        #plt.savefig(save_path, dpi=600, transparent=True)
        #plt.close(fig_all) 
        plt.show()
        
    print(f"\n[Saved params] {out_txt}")
    print("\nBatch processing done :)")
