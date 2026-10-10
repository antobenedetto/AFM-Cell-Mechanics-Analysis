#Young's modulus from force-distance curves using Hertz model
#Data from Nanosurf NHF files version 13.6

##!!!!!!!check first !!!!!!!!!
##baseline_end_fraction = 0.5 
##tip_radius = 5e-6           # m



import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from nanosurf.lib.util import nhf_reader, fileutil
import scipy.optimize as opt
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl

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
# Probe and Analysis Parameters
# ==============================
deflection_sensitivity = 0  # m/V, 0 -> use calibration
spring_constant = 0         # N/m, 0 -> use calibration
analysis_model = "Hertz"
tip_radius = 5e-6           # m
cone_half_angle = 15        # deg
poisson_ratio = 0.5
baseline_start_fraction = 0.0
baseline_end_fraction = 0.3
ignore_adhesion_effect = True
plot_force_spectroscopy = True
wait_segment_exists = True

# ==============================
# Data Structures
# ==============================
class MeasurementData:
    ch_deflection_advance = ch_deflection_wait = ch_deflection_retract = None
    ch_z_position_advance = ch_z_position_wait = ch_z_position_retract = None
    ch_time_advance = ch_time_wait = ch_time_retract = None
    ch_meta_advance = ch_meta_wait = ch_meta_retract = None
    calc_indentation_advance = calc_indentation_wait = calc_indentation_retract = None
    calc_force_advance = calc_force_wait = calc_force_retract = None
    data_offsets_advance = data_offsets_wait = data_offsets_retract = None
    datapoints_acquired_advance = datapoints_acquired_wait = datapoints_acquired_retract = None

sample_data = MeasurementData()

# ==============================
# Utility Functions
# ==============================
def get_offset_datapoints(segment, channel):
    try:
        offset = segment.read_channel('channel_data_offsets').dataset
        datapoints = segment.read_channel('number_of_datapoints_acquired').dataset
    except Exception:
        try:
            block_size_id = channel.attribute["dataset_block_size_source"]
            offsets_list = segment.find_dataset_by_attribute_value("dataset_block_size_id", block_size_id)
            offset = np.insert(np.cumsum(offsets_list), 0, 0)
            datapoints = np.diff(offset)
        except Exception as e:
            print(f"Warning: Failed to get offsets and datapoints: {e}")
            offset = datapoints = None
    return offset, datapoints

def fix_deflection_unit(channel, measurement):
    global deflection_sensitivity
    global spring_constant
    if channel.unit == 'V':
        channel.dataset *= deflection_sensitivity
    elif channel.unit == 'm':
        channel.dataset = channel.dataset / measurement.attribute['spm_probe_calibration_deflection_sensitivity'] * deflection_sensitivity
    elif channel.unit == 'N':
        channel.dataset = channel.dataset / (measurement.attribute['spm_probe_calibration_deflection_sensitivity'] * measurement.attribute['spm_probe_calibration_spring_constant']) * deflection_sensitivity
    else:
        print(f"{channel.name} unit unknown and not converted")

def import_data(file_path, data_type="Measurement"):
    nhf_file = nhf_reader.NHFFileReader(verbose=True)
    if not nhf_file.read(file_path):
        print(f"Could not read file: {file_path}")
        return False
    measurement_name = nhf_file.measurement_name(0)
    measurement = nhf_file.measurement[measurement_name]

    global deflection_sensitivity, spring_constant
    deflection_sensitivity = deflection_sensitivity or measurement.attribute['spm_probe_calibration_deflection_sensitivity']
    spring_constant = spring_constant or measurement.attribute['spm_probe_calibration_spring_constant']

    if data_type != "Measurement":
        return False

    data = sample_data
    segments = {"Advance": "Advance to Setpoint 1", "Wait": "Wait 1", "Retract": "Retract 1"}
    for key, name in segments.items():
        try:
            segment = measurement.segment[name]
            setattr(data, f'ch_time_{key.lower()}', segment.read_channel('Time'))
            setattr(data, f'ch_deflection_{key.lower()}', segment.read_channel('Deflection'))
            setattr(data, f'ch_z_position_{key.lower()}', segment.read_channel('Position Z'))
            fix_deflection_unit(getattr(data, f'ch_deflection_{key.lower()}'), measurement)
            offset, points = get_offset_datapoints(segment, getattr(data, f'ch_time_{key.lower()}'))
            setattr(data, f'data_offsets_{key.lower()}', offset)
            setattr(data, f'datapoints_acquired_{key.lower()}', points)
        except KeyError:
            if key == "Wait":
                global wait_segment_exists
                wait_segment_exists = False
            else:
                print(f"Error: {key}-Segment missing in measurement data")
                return False
    return True

# ==============================
# Fit Functions
# ==============================
def linear_fit(x, slope, intercept):
    return slope * x + intercept

def hertz_sphere_fit(indentation, effective_modulus, contact_point):
    coefficient = (4 / 3) / (1 - poisson_ratio ** 2) * np.sqrt(tip_radius) * 1e9
    return coefficient * effective_modulus * np.power(np.maximum(indentation - contact_point, 0), 1.5)

def hertz_residuals(parameters, indentation, force):
    effective_modulus, contact_point = parameters
    residual = force - hertz_sphere_fit(indentation, effective_modulus, contact_point)
    if ignore_adhesion_effect:
        residual = np.where(force > 0, residual, 0)
    return residual

def calculate_rmse_percent(force_exp, force_fit):
    rmse = np.sqrt(np.mean((force_exp - force_fit) ** 2))
    range_force = np.max(force_exp) - np.min(force_exp)
    return 100 * rmse / range_force if range_force != 0 else 0

# ==============================
# Evaluate Single Point
# ==============================
def evaluate_force_distance_curve(point_index=0):
    # -------------------------------
    # Extract raw advance data
    # -------------------------------
    start_index_advance = int(sample_data.data_offsets_advance[point_index])
    end_index_advance = start_index_advance + int(sample_data.datapoints_acquired_advance[point_index])

    z_position_advance = sample_data.ch_z_position_advance.dataset[start_index_advance:end_index_advance]
    deflection_advance = sample_data.ch_deflection_advance.dataset[start_index_advance:end_index_advance]

    # -------------------------------
    # Baseline correction
    # -------------------------------
    baseline_start_index = int(len(z_position_advance) * baseline_start_fraction)
    baseline_end_index = int(len(z_position_advance) * baseline_end_fraction)

    try:
        optimized_baseline_params, *_ = opt.curve_fit(
            linear_fit,
            z_position_advance[baseline_start_index:baseline_end_index],
            deflection_advance[baseline_start_index:baseline_end_index],
            [0.0, 0.0]
        )
        tilt_advance = linear_fit(z_position_advance, *optimized_baseline_params)
        deflection_advance -= tilt_advance
    except Exception as e:
        print(f"Baseline correction failed: {e}")
        return None

    # -------------------------------
    # Compute indentation and force
    # -------------------------------
    indentation = -(z_position_advance + deflection_advance)
    force = deflection_advance * spring_constant

    # -------------------------------
    # Initial Hertz fit (2 parameters)
    # -------------------------------
    initial_guess = [1.0, indentation[baseline_end_index]]
    bounds = ([1.0, np.min(indentation)], [1e7, np.max(indentation)])
    fit_result = opt.least_squares(hertz_residuals, initial_guess, bounds=bounds, args=(indentation, force*1e9))
    E_full, CP_initial = fit_result.x
    indentation_shifted = indentation - CP_initial

    # -------------------------------
    # Refined fit ±600 nm
    # -------------------------------
    range_limit = 600e-9
    mask_refined = (indentation_shifted >= -range_limit) & (indentation_shifted <= range_limit)
    indentation_refined = indentation_shifted[mask_refined]
    force_refined = force[mask_refined]

    refined_result = opt.least_squares(
        hertz_residuals,
        [E_full, 0.0],
        bounds=([1.0, -range_limit], [1e7, range_limit]),
        args=(indentation_refined, force_refined*1e9)
    )
    E_refined, CP_refined = refined_result.x
    indentation_refined_zeroed = indentation_refined - CP_refined

    # -------------------------------
    # Final local fit: right 500 nm (E only)
    # -------------------------------
    right_limit = 500e-9
    mask_final = (indentation_refined_zeroed >= 0) & (indentation_refined_zeroed <= right_limit)
    indentation_final = indentation_refined_zeroed[mask_final]
    force_final = force_refined[mask_final]

    def hertz_residual_E_only(E, ind_local, f_local):
        return f_local - hertz_sphere_fit(ind_local, E[0], 0)

    final_result = opt.least_squares(hertz_residual_E_only, [E_refined], bounds=([1.0], [1e7]), args=(indentation_final, force_final*1e9))
    E_500nm = final_result.x[0]

    RMSE_500nm = calculate_rmse_percent(force_final, hertz_sphere_fit(indentation_final, E_500nm, 0)*1e-9)

    # -------------------------------
    # Plot three fits in one figure
    # -------------------------------
    if plot_force_spectroscopy:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

        axes[0].plot(indentation_shifted* 1e6, force* 1e9, 'k.', markersize=0.5, label='Datapoints')
        axes[0].plot(indentation_shifted* 1e6, hertz_sphere_fit(indentation_shifted, E_full, 0),
                     'b-', linewidth=2, label='Initial Hertz Fit')
        
        #axes[0].set_title('Initial Hertz Fit')
        axes[0].set_xlabel('Indentation (μm)')
        axes[0].set_ylabel(r'Force (nN)')
        axes[0].legend()
        axes[0].text(0.05, 0.7, f"$E^{{(1)}}$ = {E_full:.2e} Pa", transform=axes[0].transAxes,
                 verticalalignment='top')

        for spine in axes[0].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            
        axes[1].plot(indentation_refined_zeroed* 1e9, force_refined* 1e9, 'k.', markersize=3, label=r"Datapoints $\delta_1 \pm 600\ \mathrm{nm}$")
        axes[1].plot(indentation_refined_zeroed* 1e9, hertz_sphere_fit(indentation_refined_zeroed, E_refined, 0),
                     'r-', linewidth=2, label='Refined Hertz Fit')
        #axes[1].set_title('Refined Hertz Fit')
        axes[1].set_xlabel('Indentaion (nm)')
        #axes[1].set_ylabel(r'$F_{Approach}$ (nN)')
        axes[1].legend()
        axes[1].text(0.05, 0.7, f"$E^{{(2)}}$= {E_refined:.2e} Pa", transform=axes[1].transAxes,
                 verticalalignment='top')

        for spine in axes[1].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        axes[2].plot(indentation_final* 1e9, force_final* 1e9, 'k.', markersize=3, label='Datapoints (500 nm)')
        axes[2].plot(indentation_final* 1e9, hertz_sphere_fit(indentation_final, E_500nm, 0),
                     'g-', linewidth=2, label='Final Hertz Fit')
        #axes[2].set_title('Final Local Fit (E only)')
        axes[2].set_xlabel('Indentation (nm)')
        #axes[2].set_ylabel(r'$F_{Approach}$ (nN)')
        axes[2].legend()
        axes[2].text(0.05, 0.7, f"E = {E_500nm:.2e} Pa", transform=axes[2].transAxes,
                 verticalalignment='top')
        
        for spine in axes[2].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        plt.tight_layout()
        save_path = pathlib.Path(file_paths[0]).parent / "Hertz_Fits2.png"
        plt.savefig(save_path, dpi=600, transparent=True)
        plt.show()

    return E_full, E_500nm, RMSE_500nm

# ==============================
# MAIN: Batch Processing
# ==============================
if __name__ == "__main__":
    # Select multiple NHF files
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select NHF Files", filetypes=[("NHF files","*.nhf")])
    if not file_paths:
        sys.exit("No files selected.")

    results = []
    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        if import_data(file_path):
            result = evaluate_force_distance_curve(point_index=0)
            if result:
                E_full, E_500nm, RMSE_500nm = result
                results.append((pathlib.Path(file_path).name, E_full, E_500nm, RMSE_500nm))
                print(f"File: {file_path}\nE_full: {E_full:.2e} Pa, E_500nm: {E_500nm:.2e} Pa, RMSE_500nm: {RMSE_500nm:.2f}%")
            else:
                print(f"Failed to analyze: {file_path}")
        else:
            print(f"Failed to import: {file_path}")

    # Save results to TXT in same folder as first file
    if results:
        save_path = pathlib.Path(file_paths[0]).parent / "E_figure.txt"
        with open(save_path, "w") as f:
            f.write("Filename\tE_full (Pa)\tE_500nm (Pa)\tRMSE_500nm (%)\n")
            for r in results:
                f.write(f"{r[0]}\t{r[1]:.6e}\t{r[2]:.6e}\t{r[3]:.2f}\n")
        print(f"\nResults saved to: {save_path}")

    print("\nBatch processing done :)")
