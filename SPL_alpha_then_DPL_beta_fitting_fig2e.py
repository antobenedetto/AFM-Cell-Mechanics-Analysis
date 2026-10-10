##### MODIFIED VERSION #####

## open the vea txt file (frequency G' G'' loss tangent data)
## step 1: fit G' with a single power law to obtain alpha
## step 2: use this fixed alpha to fit G'' with a double power law to obtain beta
## apply IQR + RMSE filter to remove outliers
## average fitting parameters of remaining curves
## use the ave fitting parameters to plot prediction curve and calculate crossover frequency

## Add a column for loss tangent at 80 Hz in the summary file

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from matplotlib.ticker import LogLocator, NullLocator

# ---------- Plot style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],

    "axes.labelsize": 14,
    "axes.labelweight": "bold",

    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 2,
    "ytick.major.width": 2,
})

# ---------- File selection ----------
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select multiple .txt files",
    filetypes=[("Text files", "*.txt")]
)
if not file_paths:
    raise SystemExit("No files selected.")

file_paths = [Path(p) for p in file_paths]
output_folder = file_paths[0].parent / "SPL_alpha_DPL_beta"
output_folder.mkdir(exist_ok=True)

# ---------- Models ----------
def gp_single_power_law(f, A_gp, alpha):
    return A_gp * np.cos(np.pi * alpha / 2) * f**alpha

def gpp_double_power_law_fixed_alpha(f, A_gpp_alpha, B, beta, alpha):
    return (
        A_gpp_alpha * np.sin(np.pi * alpha / 2) * f**alpha
        + B * np.sin(np.pi * beta / 2) * f**beta
    )

def fitted_components(f, A_gp, alpha, A_gpp_alpha, B, beta):
    Gp = gp_single_power_law(f, A_gp, alpha)
    Gpp = gpp_double_power_law_fixed_alpha(f, A_gpp_alpha, B, beta, alpha)
    return Gp, Gpp

def gp_single_power_law_weighted(f, A_gp, alpha):
    return gp_single_power_law(f, A_gp, alpha) / np.sqrt(f)

def gpp_double_power_law_fixed_alpha_weighted(f, A_gpp_alpha, B, beta, alpha):
    return gpp_double_power_law_fixed_alpha(f, A_gpp_alpha, B, beta, alpha) / np.sqrt(f)

# ---------- Compute crossover ----------
def compute_crossover(popt):
    """Return the positive crossover predicted by the fitted models.

    The calculation is not restricted to the measured frequency range.  It
    solves G'(f) = G''(f) analytically using the fitted parameters.
    """
    A_gp, alpha, A_gpp_alpha, B, beta = popt

    # After cancelling f**alpha from G'(f) = G''(f):
    # coefficient_alpha = coefficient_beta * f**(beta - alpha)
    coefficient_alpha = (
        A_gp * np.cos(np.pi * alpha / 2)
        - A_gpp_alpha * np.sin(np.pi * alpha / 2)
    )
    coefficient_beta = B * np.sin(np.pi * beta / 2)
    exponent = beta - alpha

    if (
        not np.all(np.isfinite([coefficient_alpha, coefficient_beta, exponent]))
        or coefficient_beta == 0
        or exponent == 0
    ):
        return np.nan

    ratio = coefficient_alpha / coefficient_beta
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan

    # Evaluate through logarithms to reduce overflow/underflow for crossover
    # frequencies far outside the experimental window.
    log_crossover = np.log(ratio) / exponent
    if not np.isfinite(log_crossover):
        return np.nan

    crossover = np.exp(log_crossover)
    return crossover if np.isfinite(crossover) and crossover > 0 else np.nan

# ---------- Manual log y ticks ----------
def set_manual_log_yticks(ax, y_values):
    y_values = np.asarray(y_values)
    y_values = y_values[np.isfinite(y_values) & (y_values > 0)]
    if y_values.size == 0:
        return

    ymin = np.min(y_values)
    ymax = np.max(y_values)
    y_pad = (ymax / ymin) ** 0.04

    exp_min = int(np.floor(np.log10(ymin)))
    exp_max = int(np.ceil(np.log10(ymax)))

    yticks = [10.0**e for e in range(exp_min, exp_max + 1)]
    ylabels = [f"{tick:g}" for tick in yticks]

    ax.set_ylim(min(ymin / y_pad, yticks[0]), max(ymax * y_pad, yticks[-1]))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=16)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='y', which='major', labelleft=True) 

# ---------- Fit single file ----------
def fit_single_file(file_path):
    data = np.loadtxt(file_path, skiprows=8) ##change

    # Assume column order: frequency, G', loss tangent, G''
    f_raw, Gp_raw, loss_tangent_raw, Gpp_raw = (
        data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    )

    # Use the value measured at 80 Hz, or at the nearest available frequency.
    loss_tangent_mask = (
        np.isfinite(f_raw) & np.isfinite(loss_tangent_raw) & (f_raw > 0)
    )
    if np.any(loss_tangent_mask):
        valid_indices = np.flatnonzero(loss_tangent_mask)
        nearest_index = valid_indices[
            np.argmin(np.abs(f_raw[valid_indices] - 80.0))
        ]
        loss_tangent_at_80_hz = loss_tangent_raw[nearest_index]
    else:
        loss_tangent_at_80_hz = np.nan

    mask = (
        np.isfinite(f_raw) & np.isfinite(Gp_raw) & np.isfinite(Gpp_raw) &
        (f_raw > 0) & (Gp_raw > 0) & (Gpp_raw > 0)
    )
    f, Gp, Gpp = f_raw[mask], Gp_raw[mask], Gpp_raw[mask]

    if len(f) < 5:
        raise ValueError("Too few valid points")

    # Step 1: fit E' with a single power law and obtain alpha.
    Gp_weighted = Gp / np.sqrt(f)

    p0_gp = [1000, 0.1]
    bounds_gp = ([0, 0], [1e8, 2])
    popt_gp, _ = curve_fit(
        gp_single_power_law_weighted,
        f,
        Gp_weighted,
        p0=p0_gp,
        bounds=bounds_gp,
        maxfev=200000
    )
    A_gp, alpha = popt_gp

    # Step 2: fit G'' with a double power law while keeping alpha fixed.
    Gpp_weighted = Gpp / np.sqrt(f)

    def gpp_model_for_fit(freq, A_gpp_alpha, B, beta):
        return gpp_double_power_law_fixed_alpha_weighted(
            freq,
            A_gpp_alpha,
            B,
            beta,
            alpha
        )

    p0_gpp = [100, 100, 1.0]
    bounds_gpp = ([0, 0, 0], [1e8, 1e8, 2])
    popt_gpp, _ = curve_fit(
        gpp_model_for_fit,
        f,
        Gpp_weighted,
        p0=p0_gpp,
        bounds=bounds_gpp,
        maxfev=200000
    )
    A_gpp_alpha, B, beta = popt_gpp
    popt = np.array([A_gp, alpha, A_gpp_alpha, B, beta])

    # ---------- RMSE calculation (%) ----------
    # Keep RMSE evaluation at the measured frequencies only.
    Gp_pred, Gpp_pred = fitted_components(f, *popt)
    rmse_percent = (
        np.sqrt(np.mean((Gp - Gp_pred)**2 + (Gpp - Gpp_pred)**2))
        / np.mean(np.concatenate([Gp, Gpp])) * 100
    )

    crossover = compute_crossover(popt)

    # Draw the fitted prediction curves over a range that includes both the
    # measured frequencies and the predicted crossover, even when the
    # crossover lies outside the experimental window.
    plot_f_min = np.nanmin(f)
    plot_f_max = np.nanmax(f)
    if np.isfinite(crossover) and crossover > 0:
        plot_f_min = min(plot_f_min, crossover)
        plot_f_max = max(plot_f_max, crossover)

    if plot_f_max > plot_f_min:
        f_plot = np.logspace(
            np.log10(plot_f_min),
            np.log10(plot_f_max),
            500
        )
    else:
        f_plot = np.array([plot_f_min])

    Gp_plot, Gpp_plot = fitted_components(f_plot, *popt)

    # ---------- Save single figure ----------
    fig, ax = plt.subplots(figsize=(4, 3.2))

    ax.loglog(f, Gp, 'o', color='#1f77b4', alpha=0.5, label="E'  ")
    ax.loglog(f, Gpp, 'o', color='#ff7f0e', alpha=0.5, label="E''  ")
    ax.loglog(f_plot, Gp_plot, color='#1f77b4', label="E' SPL fit")
    ax.loglog(f_plot, Gpp_plot, color='#ff7f0e', label="E'' DPL fit")

    if np.isfinite(crossover) and crossover > 0:
        Gp_cross, _ = fitted_components(np.array([crossover]), *popt)
        ax.plot(
            crossover,
            Gp_cross[0],
            marker='x',
            markersize=7,
            markeredgewidth=1.8,
            linestyle='none',
            color='black',
            label=f"Crossover: {crossover:.3g} Hz"
        )

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())

    y_all = np.concatenate([Gp, Gpp, Gp_plot, Gpp_plot])
    set_manual_log_yticks(ax, y_all)

    fig.subplots_adjust(left=0.22, bottom=0.16, right=0.97, top=0.90)

    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel("Complex Modulus E* (Pa)", fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")

    x_min, x_max = plot_f_min, plot_f_max
    x_pad = (x_max / x_min) ** 0.04
    ax.set_xlim(x_min / x_pad, x_max * x_pad)
    x_tick_min = int(np.floor(np.log10(x_min)))
    x_tick_max = int(np.ceil(np.log10(x_max)))
    x_tick_exponents = np.arange(x_tick_min, x_tick_max + 1)
    if x_tick_exponents.size > 8:
        x_tick_exponents = np.unique(
            np.rint(np.linspace(x_tick_min, x_tick_max, 8)).astype(int)
        )
    x_ticks = 10.0 ** x_tick_exponents
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    ax.set_xticks(x_ticks)

    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks], fontsize=16)

    ax.legend(frameon=False)

    plt.savefig(output_folder / f"{file_path.stem}_fit.png", dpi=600, transparent=True)
    plt.close()

    return {
        "file": file_path.name,
        "popt": popt,
        "crossover": crossover,
        "rmse_percent": rmse_percent,
        "loss_tangent_at_80_hz": loss_tangent_at_80_hz,
    }

# ---------- IQR + RMSE filter ----------
def iqr_rmse_filter(results_list, multiplier=1.5, rmse_thresh=20):
    popts = np.array([res['popt'] for res in results_list])
    rmse_list = np.array([res['rmse_percent'] for res in results_list])
    filenames = [res['file'] for res in results_list]

    Q1 = np.percentile(popts, 25, axis=0)
    Q3 = np.percentile(popts, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    mask_iqr = np.all((popts >= lower_bound) & (popts <= upper_bound), axis=1)

    mask_rmse = rmse_list <= rmse_thresh
    mask_keep = mask_iqr & mask_rmse
    popts_filtered = popts[mask_keep]

    removed_files = []
    removal_reasons = {}
    for i, keep in enumerate(mask_keep):
        if not keep:
            removed_files.append(filenames[i])
            reasons = []
            if not mask_iqr[i]:
                reasons.append("IQR")
            if not mask_rmse[i]:
                reasons.append("RMSE")
            removal_reasons[filenames[i]] = "+".join(reasons)

    if len(popts_filtered) > 0:
        popt_mean = np.mean(popts_filtered, axis=0)
    else:
        popt_mean = np.full(popts.shape[1], np.nan)

    return popt_mean, popts_filtered, removed_files, removal_reasons

# ---------- Loop over files ----------
results_list = []
for path in file_paths:
    try:
        res = fit_single_file(path)
        results_list.append(res)
    except Exception as e:
        print(f"Error fitting {path.name}: {e}")

if len(results_list) == 0:
    raise SystemExit("No files were successfully fitted.")

# ---------- Apply IQR + RMSE filter ----------
popt_mean, popts_filtered, removed_files, removal_reasons = iqr_rmse_filter(
    results_list,
    multiplier=3,
    rmse_thresh=20
)

# ---------- Average crossover of filtered curves ----------
crossovers_filtered = np.array([
    res['crossover'] for res in results_list
    if res['file'] not in removed_files and not np.isnan(res['crossover'])
])

if len(crossovers_filtered) > 0:
    crossover_mean = np.mean(crossovers_filtered)
    crossover_std = np.std(crossovers_filtered)
else:
    crossover_mean = np.nan
    crossover_std = np.nan

print(f"Average crossover (filtered curves) = {crossover_mean:.3f} +/- {crossover_std:.3f} Hz")

# ---------- Save summary ----------  change 
summary_file = output_folder / "spl_alpha_dpl_beta.txt"
stats_file = output_folder / "fit_statistics.txt"

def fmt_value(value, decimals=6):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}g}"

def fixed_width_row(values, widths):
    cells = []
    for value, width in zip(values, widths):
        text = str(value)
        cells.append(text.ljust(width))
    return "".join(cells).rstrip() + "\n"

summary_widths = [34, 14, 14, 16, 14, 14, 14, 22, 12, 10]
summary_header = [
    "File",
    "A_Gp",
    "alpha",
    "A_Gpp_alpha",
    "B",
    "beta",
    "crossover",
    "loss tangent at 80 Hz",
    "RMSE(%)",
    "Status",
]
removed_header = summary_header[:-1] + ["Reason"]

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(fixed_width_row(summary_header, summary_widths))
    for res in results_list:
        if res['file'] in removed_files:
            continue
        row = [
            res['file'],
            *[fmt_value(x) for x in res['popt']],
            fmt_value(res['crossover']),
            fmt_value(res['loss_tangent_at_80_hz']),
            f"{res['rmse_percent']:.3f}",
            "Kept",
        ]
        f.write(fixed_width_row(row, summary_widths))

print(f"Saved filtered parameter summary to {summary_file}")

with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("Average parameters (after IQR+RMSE filtering):\n")
    f.write(fixed_width_row(summary_header[1:6], summary_widths[1:6]))
    f.write(fixed_width_row([fmt_value(x) for x in popt_mean], summary_widths[1:6]))

    f.write("\nAverage crossover of filtered curves:\n")
    f.write(f"Mean +/- Std: {crossover_mean:.6g} +/- {crossover_std:.6g} Hz\n")

    f.write("\nRemoved files with parameters:\n")
    f.write(fixed_width_row(removed_header, summary_widths))
    for res in results_list:
        if res['file'] in removed_files:
            reason = removal_reasons[res['file']]
            row = [
                res['file'],
                *[fmt_value(x) for x in res['popt']],
                fmt_value(res['crossover']),
                fmt_value(res['loss_tangent_at_80_hz']),
                f"{res['rmse_percent']:.3f}",
                reason,
            ]
            f.write(fixed_width_row(row, summary_widths))

print(f"Saved statistics (with removed files + parameters) to {stats_file}")
