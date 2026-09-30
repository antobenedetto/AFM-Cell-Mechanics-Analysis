# ==============================
# Adhesion analysis from Retract 1 only
# For Nanosurf NHF files
# With automatic h5py fallback reader
# ==============================

import pathlib
import shutil
import tempfile

import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl

try:
    from nanosurf.lib.util import nhf_reader
except ImportError:
    nhf_reader = None


# ==============================
# Plot style
# ==============================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],

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


# ==============================
# Parameters
# ==============================
deflection_sensitivity = 0   # m/V, 0 means use calibration from file
spring_constant = 0          # N/m, 0 means use calibration from file

baseline_start_fraction = 0.8
baseline_end_fraction = 1.0  # check this first

smooth_force = True
savgol_window = 5
savgol_polyorder = 2

save_plots = True

# For batch checking, False gives cleaner figures.
# Set True if you want legend on every plot.
show_legend = False


# ==============================
# Data structure
# ==============================
class MeasurementData:
    def __init__(self):
        self.ch_deflection_retract = None
        self.ch_z_position_retract = None
        self.ch_time_retract = None

        self.data_offsets_retract = None
        self.datapoints_acquired_retract = None


class SimpleChannel:
    """
    Minimal channel object for h5py fallback.

    It mimics the parts of the Nanosurf channel object used later:
        channel.dataset
        channel.unit
        channel.name
        channel.attribute
    """
    def __init__(self, name, dataset, unit, attribute=None):
        self.name = name
        self.dataset = dataset
        self.unit = unit
        self.attribute = attribute if attribute is not None else {}


class SimpleMeasurement:
    """
    Minimal measurement object for h5py fallback.

    It mimics:
        measurement.attribute
    """
    def __init__(self, attribute):
        self.attribute = attribute


sample_data = MeasurementData()


# ==============================
# Utility functions
# ==============================
def linear_fit(x, slope, intercept):
    return slope * x + intercept


def smooth_savgol(data, window_length=5, polyorder=2):
    data = np.asarray(data)

    if len(data) < window_length:
        return data

    if window_length % 2 == 0:
        window_length += 1

    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1

    if window_length < 3:
        return data

    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def decode_attr(value):
    """
    Convert h5py attribute values to normal Python strings when needed.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    return value


def attr_to_string(value):
    """
    Robust conversion of h5py attributes to string.
    """

    value = decode_attr(value)
    return str(value)


def make_safe_nhf_copy(file_path):
    """
    Copy NHF file to a simple ASCII temporary folder with a simple ASCII filename.

    This avoids common Nanosurf reader failures caused by:
    - OneDrive paths
    - Chinese/special characters in folder names
    - Chinese/special characters in file names
    - very long Windows paths
    - cloud-synced folders
    """

    file_path = pathlib.Path(file_path).resolve()

    safe_folder = pathlib.Path(tempfile.gettempdir()) / "nanosurf_nhf_temp"
    safe_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".nhf",
        prefix="nhf_",
        dir=safe_folder,
        delete=False
    ) as tmp:
        safe_file = pathlib.Path(tmp.name)

    shutil.copy2(file_path, safe_file)

    return safe_file


def get_short_unit(unit_string):
    """
    Convert Nanosurf unit strings to short unit symbols.

    Examples:
        'volt (V)' -> 'V'
        'meter (m)' -> 'm'
        'newton (N)' -> 'N'
        'second (s)' -> 's'
    """

    if unit_string is None:
        return ""

    unit_string = attr_to_string(unit_string)

    if "(" in unit_string and ")" in unit_string:
        return unit_string.split("(")[-1].split(")")[0]

    lower = unit_string.lower()

    if "volt" in lower:
        return "V"

    if "meter" in lower:
        return "m"

    if "newton" in lower:
        return "N"

    if "second" in lower:
        return "s"

    return unit_string


def calibrate_h5_dataset(dataset):
    """
    Convert raw HDF5 dataset to calibrated physical values.

    For integer Nanosurf channels:
        use signal_minmax and signal_calibration_min/max.

    For already-calibrated float channels:
        return data directly if calibration attributes are absent.
    """

    raw = np.asarray(dataset[:], dtype=float)
    attrs = dataset.attrs

    if "signal_calibration_min" not in attrs or "signal_calibration_max" not in attrs:
        return raw

    cal_min = float(attrs["signal_calibration_min"])
    cal_max = float(attrs["signal_calibration_max"])

    if "signal_minmax" in attrs:
        raw_min = float(attrs["signal_minmax"][0])
        raw_max = float(attrs["signal_minmax"][1])
    else:
        raw_min = float(np.nanmin(raw))
        raw_max = float(np.nanmax(raw))

    if raw_max == raw_min:
        return raw

    calibrated = (
        (raw - raw_min)
        / (raw_max - raw_min)
        * (cal_max - cal_min)
        + cal_min
    )

    return calibrated


def find_h5_measurement_group(h5_file):
    """
    Find the spectroscopy measurement group.
    """

    available_groups = []

    for key in h5_file.keys():
        obj = h5_file[key]

        if not isinstance(obj, h5py.Group):
            continue

        group_type = attr_to_string(obj.attrs.get("group_type", ""))
        group_name = attr_to_string(obj.attrs.get("name", key))

        available_groups.append((key, group_name, group_type))

        if group_type == "spectroscopy":
            return obj

    print("Available top-level groups:")
    for key, name, group_type in available_groups:
        print(f"   {key}: name={name}, group_type={group_type}")

    raise KeyError("No spectroscopy measurement group found.")


def find_h5_segment_group(measurement_group, segment_name="Retract 1"):
    """
    Find subgroup with selected segment_name.
    """

    available_segments = []

    for key in measurement_group.keys():
        obj = measurement_group[key]

        if not isinstance(obj, h5py.Group):
            continue

        this_segment_name = obj.attrs.get("segment_name", None)

        if this_segment_name is not None:
            this_segment_name = attr_to_string(this_segment_name)
            available_segments.append(this_segment_name)

            if this_segment_name == segment_name:
                return obj

    print("Available segments:")
    for name in available_segments:
        print("   ", name)

    raise KeyError(f"Segment not found: {segment_name}")


def find_h5_channel_dataset(segment_group, signal_name):
    """
    Find dataset by signal_name.

    Examples:
        'Time'
        'Deflection'
        'Position Z'
    """

    available_channels = []

    for key in segment_group.keys():
        obj = segment_group[key]

        if not isinstance(obj, h5py.Dataset):
            continue

        this_signal_name = obj.attrs.get("signal_name", None)

        if this_signal_name is not None:
            this_signal_name = attr_to_string(this_signal_name)
            available_channels.append(this_signal_name)

            if this_signal_name == signal_name:
                return obj

    print("Available channels:")
    for name in available_channels:
        print("   ", name)

    raise KeyError(f"Channel not found: {signal_name}")


def find_h5_datapoints(segment_group):
    """
    Find number_of_datapoints_acquired equivalent from HDF5.

    In this NHF structure, this is usually the dataset with:
        dataset_block_size_id

    It gives the number of data points for each force curve.
    """

    for key in segment_group.keys():
        obj = segment_group[key]

        if not isinstance(obj, h5py.Dataset):
            continue

        if "dataset_block_size_id" in obj.attrs:
            datapoints = np.asarray(obj[:], dtype=int)
            offsets = np.insert(np.cumsum(datapoints), 0, 0)

            return offsets, datapoints

    raise KeyError("Could not find datapoints dataset in segment.")


def get_offset_datapoints(segment, channel):
    try:
        offset = segment.read_channel("channel_data_offsets").dataset
        datapoints = segment.read_channel("number_of_datapoints_acquired").dataset

    except Exception:
        try:
            block_size_id = channel.attribute["dataset_block_size_source"]
            offsets_list = segment.find_dataset_by_attribute_value(
                "dataset_block_size_id",
                block_size_id
            )
            offset = np.insert(np.cumsum(offsets_list), 0, 0)
            datapoints = np.diff(offset)

        except Exception as e:
            print(f"Warning: Failed to get offsets and datapoints: {e}")
            offset = None
            datapoints = None

    return offset, datapoints


def fix_deflection_unit(channel, measurement):
    """
    Convert deflection channel to meters.

    After this function:
        channel.dataset = cantilever deflection in m
    """
    global deflection_sensitivity
    global spring_constant

    file_deflection_sensitivity = measurement.attribute[
        "spm_probe_calibration_deflection_sensitivity"
    ]
    file_spring_constant = measurement.attribute[
        "spm_probe_calibration_spring_constant"
    ]

    if channel.unit == "V":
        channel.dataset *= deflection_sensitivity

    elif channel.unit == "m":
        channel.dataset = (
            channel.dataset
            / file_deflection_sensitivity
            * deflection_sensitivity
        )

    elif channel.unit == "N":
        channel.dataset = (
            channel.dataset
            / (file_deflection_sensitivity * file_spring_constant)
            * deflection_sensitivity
        )

    else:
        print(f"{channel.name} unit unknown and not converted")


# ==============================
# Import Retract 1 only
# ==============================
def import_data_with_nhf_reader(file_path, data_type="Measurement"):
    global deflection_sensitivity
    global spring_constant
    global sample_data

    if nhf_reader is None:
        print("nanosurf package not available. Skipping nhf_reader.")
        return False

    original_file_path = pathlib.Path(file_path).resolve()

    try:
        safe_file_path = make_safe_nhf_copy(original_file_path)
    except Exception as e:
        print(f"Failed to create safe NHF copy: {original_file_path}")
        print(e)
        return False

    nhf_file = nhf_reader.NHFFileReader(verbose=True)

    try:
        read_ok = nhf_file.read(str(safe_file_path))
    except Exception as e:
        print("NHF reader crashed when reading safe copy:")
        print(safe_file_path)
        print(e)
        return False

    if not read_ok:
        print("Could not read NHF file even after safe copy.")
        print("Original file:", original_file_path)
        print("Safe copy:", safe_file_path)
        return False

    measurement_name = nhf_file.measurement_name(0)
    measurement = nhf_file.measurement[measurement_name]

    if deflection_sensitivity == 0:
        deflection_sensitivity = measurement.attribute[
            "spm_probe_calibration_deflection_sensitivity"
        ]

    if spring_constant == 0:
        spring_constant = measurement.attribute[
            "spm_probe_calibration_spring_constant"
        ]

    if data_type != "Measurement":
        return False

    data = sample_data

    try:
        segment = measurement.segment["Retract 1"]

        data.ch_time_retract = segment.read_channel("Time")
        data.ch_deflection_retract = segment.read_channel("Deflection")
        data.ch_z_position_retract = segment.read_channel("Position Z")

        fix_deflection_unit(data.ch_deflection_retract, measurement)

        offset, points = get_offset_datapoints(
            segment,
            data.ch_time_retract
        )

        data.data_offsets_retract = offset
        data.datapoints_acquired_retract = points

        if offset is None or points is None:
            print("Failed to read curve offsets from Retract 1.")
            return False

    except KeyError:
        print("Error: Retract 1 segment missing in measurement data")

        try:
            print("Available segments:")
            for seg_name in measurement.segment.keys():
                print("   ", seg_name)
        except Exception:
            pass

        return False

    return True


def import_data_with_h5py(file_path, data_type="Measurement"):
    global deflection_sensitivity
    global spring_constant
    global sample_data

    if data_type != "Measurement":
        return False

    file_path = pathlib.Path(file_path).resolve()

    try:
        with h5py.File(file_path, "r") as h5_file:
            measurement_group = find_h5_measurement_group(h5_file)

            segment_group = find_h5_segment_group(
                measurement_group,
                segment_name="Retract 1"
            )

            measurement_attrs = dict(measurement_group.attrs)

            if deflection_sensitivity == 0:
                deflection_sensitivity = measurement_attrs[
                    "spm_probe_calibration_deflection_sensitivity"
                ]

            if spring_constant == 0:
                spring_constant = measurement_attrs[
                    "spm_probe_calibration_spring_constant"
                ]

            measurement = SimpleMeasurement(measurement_attrs)

            time_ds = find_h5_channel_dataset(segment_group, "Time")
            deflection_ds = find_h5_channel_dataset(segment_group, "Deflection")
            z_position_ds = find_h5_channel_dataset(segment_group, "Position Z")

            time_data = calibrate_h5_dataset(time_ds)
            deflection_data = calibrate_h5_dataset(deflection_ds)
            z_position_data = calibrate_h5_dataset(z_position_ds)

            time_unit = get_short_unit(
                time_ds.attrs.get("signal_calibration_unit", "second (s)")
            )

            deflection_unit = get_short_unit(
                deflection_ds.attrs.get("signal_calibration_unit", "volt (V)")
            )

            z_position_unit = get_short_unit(
                z_position_ds.attrs.get("signal_calibration_unit", "meter (m)")
            )

            data = sample_data

            data.ch_time_retract = SimpleChannel(
                name="Time",
                dataset=time_data,
                unit=time_unit,
                attribute=dict(time_ds.attrs)
            )

            data.ch_deflection_retract = SimpleChannel(
                name="Deflection",
                dataset=deflection_data,
                unit=deflection_unit,
                attribute=dict(deflection_ds.attrs)
            )

            data.ch_z_position_retract = SimpleChannel(
                name="Position Z",
                dataset=z_position_data,
                unit=z_position_unit,
                attribute=dict(z_position_ds.attrs)
            )

            fix_deflection_unit(data.ch_deflection_retract, measurement)

            offset, points = find_h5_datapoints(segment_group)

            data.data_offsets_retract = offset
            data.datapoints_acquired_retract = points

            if offset is None or points is None:
                print("Failed to read curve offsets from Retract 1 using h5py.")
                return False

            print("Successfully imported data using h5py fallback.")
            print("Retract curves found:", len(points))

            return True

    except Exception as e:
        print("h5py fallback failed:")
        print(file_path)
        print(e)
        return False


def import_data(file_path, data_type="Measurement"):
    """
    Main import function.

    First try original Nanosurf nhf_reader.
    If it fails, automatically use h5py fallback.

    This protects the code from:
        - path problems
        - OneDrive/cloud folder problems
        - special character path problems
        - Nanosurf Studio / NHF version compatibility problems
    """

    file_path = pathlib.Path(file_path).resolve()

    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return False

    if file_path.stat().st_size == 0:
        print(f"Empty NHF file: {file_path}")
        return False

    if file_path.suffix.lower() != ".nhf":
        print(f"Not an NHF file: {file_path}")
        return False

    print("Trying Nanosurf nhf_reader...")

    ok = import_data_with_nhf_reader(
        file_path=file_path,
        data_type=data_type
    )

    if ok:
        print("Successfully imported data using Nanosurf nhf_reader.")
        return True

    print("Nanosurf nhf_reader failed.")
    print("Trying h5py fallback reader...")

    ok = import_data_with_h5py(
        file_path=file_path,
        data_type=data_type
    )

    return ok


# ==============================
# Adhesion calculation
# ==============================
def find_adhesion(z_um, force_nN):
    """
    Adhesion force is the absolute value of the most negative force.
    """
    i_min = np.argmin(force_nN)

    z_peak_um = z_um[i_min]
    min_force_nN = force_nN[i_min]
    adhesion_nN = abs(min_force_nN)

    return z_peak_um, min_force_nN, adhesion_nN, i_min


def calculate_adhesion_work_main_event(z_um, force_nN):
    """
    Calculate adhesion work for the main adhesion event only.

    Main event:
        the continuous negative force region containing the minimum force point.

    Units:
        z_um: μm
        force_nN: nN

    Output:
        work_nN_um: nN·μm, numerically equivalent to fJ
        mask_main: boolean mask for plotting
    """
    z_um = np.asarray(z_um)
    force_nN = np.asarray(force_nN)

    if len(z_um) < 2:
        return 0.0, np.zeros_like(force_nN, dtype=bool)

    i_min = np.argmin(force_nN)

    if force_nN[i_min] >= 0:
        return 0.0, np.zeros_like(force_nN, dtype=bool)

    left = i_min
    while left > 0 and force_nN[left - 1] < 0:
        left -= 1

    right = i_min
    while right < len(force_nN) - 1 and force_nN[right + 1] < 0:
        right += 1

    mask_main = np.zeros_like(force_nN, dtype=bool)
    mask_main[left:right + 1] = True

    if np.sum(mask_main) < 2:
        return 0.0, mask_main

    z_seg = z_um[mask_main]
    f_seg = force_nN[mask_main]

    # Sort before integration to avoid negative area if Z decreases
    order = np.argsort(z_seg)
    z_seg = z_seg[order]
    f_seg = f_seg[order]

    try:
        work_nN_um = np.trapezoid(-f_seg, z_seg)
    except AttributeError:
        work_nN_um = np.trapz(-f_seg, z_seg)

    return work_nN_um, mask_main


# ==============================
# Evaluate one retract curve
# ==============================
def evaluate_force_distance_curve(point_index=0):
    """
    Extract one curve from Retract 1, baseline correct it,
    then calculate adhesion force and adhesion work.
    """

    start_index_retract = int(sample_data.data_offsets_retract[point_index])
    end_index_retract = start_index_retract + int(
        sample_data.datapoints_acquired_retract[point_index]
    )

    z_position_retract = sample_data.ch_z_position_retract.dataset[
        start_index_retract:end_index_retract
    ].copy()

    deflection_retract = sample_data.ch_deflection_retract.dataset[
        start_index_retract:end_index_retract
    ].copy()

    if len(z_position_retract) < 5:
        print(f"Point {point_index}: curve too short")
        return None

    # ------------------------------
    # Baseline correction
    # ------------------------------
    baseline_start_index = int(len(z_position_retract) * baseline_start_fraction)
    baseline_end_index = int(len(z_position_retract) * baseline_end_fraction)

    if baseline_end_index <= baseline_start_index + 2:
        print(f"Point {point_index}: baseline region too small")
        return None

    try:
        optimized_baseline_params, *_ = opt.curve_fit(
            linear_fit,
            z_position_retract[baseline_start_index:baseline_end_index],
            deflection_retract[baseline_start_index:baseline_end_index],
            [0.0, 0.0]
        )

        tilt_retract = linear_fit(
            z_position_retract,
            *optimized_baseline_params
        )

        deflection_corrected = deflection_retract - tilt_retract

        # Force the selected baseline region to zero
        baseline_mean = np.mean(
            deflection_corrected[baseline_start_index:baseline_end_index]
        )

        deflection_corrected = deflection_corrected - baseline_mean

    except Exception as e:
        print(f"Point {point_index}: baseline correction failed: {e}")
        return None

    # ------------------------------
    # Convert units
    # ------------------------------
    z_um = z_position_retract * 1e6
    force_nN = deflection_corrected * spring_constant * 1e9

    # Make curve direction consistent for plotting
    if z_um[0] > z_um[-1]:
        z_um = z_um[::-1]
        force_nN = force_nN[::-1]

    # Optional smoothing
    if smooth_force:
        force_nN = smooth_savgol(
            force_nN,
            window_length=savgol_window,
            polyorder=savgol_polyorder
        )

    # ------------------------------
    # Adhesion force and work
    # ------------------------------
    z_peak_um, min_force_nN, adhesion_nN, i_min = find_adhesion(
        z_um,
        force_nN
    )

    adhesion_work_nN_um, mask_work = calculate_adhesion_work_main_event(
        z_um,
        force_nN
    )

    result = {
        "point_index": point_index,
        "z_um": z_um,
        "force_nN": force_nN,
        "z_peak_um": z_peak_um,
        "min_force_nN": min_force_nN,
        "adhesion_nN": adhesion_nN,
        "adhesion_work_nN_um": adhesion_work_nN_um,
        "mask_work": mask_work,
        "i_min": i_min,
    }

    return result


# ==============================
# Plot result
# ==============================
def plot_adhesion_result(result, file_name, output_folder):
    z_um = result["z_um"]
    force_nN = result["force_nN"]
    mask_work = result["mask_work"]

    z_peak_um = result["z_peak_um"]
    min_force_nN = result["min_force_nN"]
    adhesion_nN = result["adhesion_nN"]
    adhesion_work_nN_um = result["adhesion_work_nN_um"]
    point_index = result["point_index"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(
        z_um,
        force_nN,
        lw=1.8,
        color="#d62728",
        label="Retract curve"
    )

    ax.scatter(
        z_peak_um,
        min_force_nN,
        facecolors="none",
        edgecolors="#1f77b4",
        zorder=5,
        label="Adhesion force"
    )

    ax.fill_between(
        z_um,
        force_nN,
        0,
        where=mask_work,
        interpolate=True,
        alpha=0.35,
        label="Adhesion work"
    )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    text = (
        f"Adhesion = {adhesion_nN:.3f} nN\n"
        f"Work = {adhesion_work_nN_um:.3f} nN·μm"
    )

    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="none",
            ec="none",
            alpha=0.85
        )
    )

    ax.set_xlabel("Z position (μm)", fontweight="bold")
    ax.set_ylabel("$F_{retract}$ (nN)", fontweight="bold")
    #ax.set_title(f"{pathlib.Path(file_name).stem}")

    if show_legend:
        ax.legend(
            frameon=False,
            loc="lower left",
            fontsize=8
        )

    for spine in ["top", "right"]:
        #ax.spines[spine].set_visible(False)
        ax.spines[spine].set_linewidth(1.5)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.5)

    fig.tight_layout()

    fig_path = (
        pathlib.Path(output_folder)
        / f"{pathlib.Path(file_name).stem}_adhesion.png"
    )

    fig.savefig(fig_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return fig_path


# ==============================
# Process one file
# ==============================
def process_file(file_path, output_folder):
    global sample_data
    global deflection_sensitivity
    global spring_constant

    # Reset data object for each file
    sample_data = MeasurementData()

    # Reset calibration so each file uses its own calibration
    deflection_sensitivity = 0
    spring_constant = 0

    ok = import_data(file_path)

    if not ok:
        print(f"Skipping file: {file_path}")
        return []

    file_name = pathlib.Path(file_path).name

    n_points = len(sample_data.datapoints_acquired_retract)

    results = []

    for point_index in range(n_points):
        result = evaluate_force_distance_curve(point_index)

        if result is None:
            continue

        if save_plots:
            plot_adhesion_result(
                result=result,
                file_name=file_name,
                output_folder=output_folder
            )

        results.append({
            "filename": file_name,
            "point_index": point_index,
            "adhesion_nN": result["adhesion_nN"],
            "z_peak_um": result["z_peak_um"],
            "min_force_nN": result["min_force_nN"],
            "adhesion_work_nN_um": result["adhesion_work_nN_um"],
        })

    return results


# ==============================
# Main batch function
# ==============================
def main():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select NHF files",
        filetypes=[
            ("NHF files", "*.nhf"),
            ("All files", "*.*")
        ]
    )

    if not file_paths:
        print("No files selected.")
        return

    output_folder = filedialog.askdirectory(
        title="Select output folder"
    )

    if not output_folder:
        output_folder = pathlib.Path(file_paths[0]).parent

    output_folder = pathlib.Path(output_folder)

    save_path = output_folder / "batch_adhesion_results.txt"

    with open(save_path, "w") as f:
        f.write(
            "filename\tpoint_index\tadhesion_nN\tz_peak_um\t"
            "min_force_nN\tadhesion_work_nN_um\n"
        )

    all_results = []

    for file_path in file_paths:
        print(f"Processing: {file_path}")

        results = process_file(
            file_path=file_path,
            output_folder=output_folder
        )

        all_results.extend(results)

        with open(save_path, "a") as f:
            for r in results:
                f.write(
                    f"{r['filename']}\t"
                    f"{r['point_index']}\t"
                    f"{r['adhesion_nN']:.6f}\t"
                    f"{r['z_peak_um']:.6f}\t"
                    f"{r['min_force_nN']:.6f}\t"
                    f"{r['adhesion_work_nN_um']:.6f}\n"
                )

    print("All done.")
    print("Results saved to:", save_path)


if __name__ == "__main__":
    main()