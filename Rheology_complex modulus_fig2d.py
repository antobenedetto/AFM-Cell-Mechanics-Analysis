""" Script to be used to analyze mult VEA measurements.
Copyright Nanosurf AG 2024
License - MIT
"""

#%%
##DEBUG
"""import os
os.environ['PYDEV_DISABLE_GUI'] = '1'  # 禁用调试器的GUI集成

# 如果是matplotlib相关的问题，也加上：
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
except ImportError:
    pass"""

##debug
####！！！！check tip_radius = 5e-6   
#####!!!!! baseline_end = 0.4

#Import libraries
import datetime as datetime
import numpy as np
import io
import pathlib
import sys
import json
import math
import os
from pathlib import Path
import traceback

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from nanosurf.lib.util import nhf_reader, fileutil, gwy_export

import scipy.optimize as opt
from scipy.signal import butter, filtfilt, savgol_filter


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

    "figure.figsize": (10, 2.8),
    
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# Probe calibration is read separately from EACH NHF file.
# Filename rule from the single-file example: <key>_1.nhf / <key>_2.nhf.
CALIBRATION_SUFFIX = "_1"
MEASUREMENT_SUFFIX = "_2"

### Set FD-Analysis Parameters
analysis_model = "Hertz"                    # Available: Hertz, Sneddon, Pyramid, DMT_Sphere, DMT_Cone
fit_direction = "Advance"                   # Available: Advance, Retract
tip_radius = 5e-6                          # m
cone_half_angle = 15                        # deg
poisson_ratio = 0.5
baseline_start = 0.0
baseline_end = 0.4
ignore_adhesion = True                      # For Hertz, Sneddon & Pyramid
ignore_baseline = True                      # For DMT_Sphere & DMT_Cone

### Set Demodulation Parameters (advanced)
fit_portion = 0.4
use_reference = True  ##  change 1
correct_drag = True
#correct_drag = False
bessel_on_raw = False
savitzky_golay_on_demod = True
#savitzky_golay_on_demod = False

### Graphical Output Options
plot_moduli = True
plot_transient = 10
plot_spec = True

### Define global measurement data channels
class Measurement_Data:
    deflection_sensitivity: float = None
    spring_constant: float = None
    ch_deflection_advance: nhf_reader.NHFDataset = None
    ch_deflection_wait: nhf_reader.NHFDataset = None
    ch_deflection_vea: nhf_reader.NHFDataset = None
    ch_deflection_retract: nhf_reader.NHFDataset = None
    ch_z_position_advance: nhf_reader.NHFDataset = None
    ch_z_position_wait: nhf_reader.NHFDataset = None
    ch_z_position_vea: nhf_reader.NHFDataset = None
    ch_z_position_retract: nhf_reader.NHFDataset = None
    ch_time_advance: nhf_reader.NHFDataset = None
    ch_time_wait: nhf_reader.NHFDataset = None
    ch_time_vea: nhf_reader.NHFDataset = None
    ch_time_retract: nhf_reader.NHFDataset = None
    ch_meta_advance: nhf_reader.NHFDataset = None
    ch_meta_wait: nhf_reader.NHFDataset = None
    ch_meta_vea: nhf_reader.NHFDataset = None
    ch_meta_retract: nhf_reader.NHFDataset = None  ##all channels

    ch_reference_vea: nhf_reader.NHFDataset = None

    calc_indentation_advance: np.ndarray = None
    calc_indentation_wait: np.ndarray = None
    calc_indentation_vea: np.ndarray = None
    calc_indentation_retract: np.ndarray = None
    
    calc_force_advance: np.ndarray = None
    calc_force_wait: np.ndarray = None
    calc_force_vea: np.ndarray = None
    calc_force_retract: np.ndarray = None

    data_offsets_advance: np.ndarray = None
    data_offsets_wait: np.ndarray = None
    data_offsets_vea: np.ndarray = None
    data_offsets_retract: np.ndarray = None
    datapoints_acquired_advance: np.ndarray = None
    datapoints_acquired_wait: np.ndarray = None
    datapoints_acquired_vea: np.ndarray = None
    datapoints_acquired_retract: np.ndarray = None

    vea_configuration: dict = None

class Measurement_Info:
    deflection_sensitivity: float
    spring_constant: float
    offset_x: float
    offset_y: float
    size_x: float
    size_y: float
    points_x: float
    points_y: float
    rotation: float

class Demod_Data:
    list_frequency: np.ndarray = None
    list_offset: np.ndarray = None
    list_deflection_amp: np.ndarray = None
    list_deflection_phase: np.ndarray = None
    list_deflection_dc: np.ndarray = None
    list_position_z_amp: np.ndarray = None
    list_position_z_phase: np.ndarray = None
    list_position_z_dc: np.ndarray = None
    list_indentation_amp: np.ndarray = None
    list_indentation_phase: np.ndarray = None
    list_indentation_dc: np.ndarray = None
    list_reference_amp: np.ndarray = None
    list_reference_phase: np.ndarray = None
    list_reference_dc: np.ndarray = None

class Result_List:
    list_frequency: np.ndarray = None
    list_modulus_storage: np.ndarray = None
    list_modulus_loss: np.ndarray = None
    list_loss_tangent: np.ndarray = None
    static_contact_point: float
    static_modulus: float
    static_gamma: float
    static_snap: float
    static_adhesion: float

class Result_Maps:
    def __init__(self, z_size: int, y_size: int, x_size: int):
        # Initialize a 2D array of Result_List objects
        self.map = np.empty((z_size, y_size, x_size), dtype=object)
    def set_result(self, z: int, y: int, x: int, result):
        """Sets a specific (x, y) coordinate to a given Result_List instance."""
        self.map[z, y, x] = result
    def get_all_results(self):
        return [self.map[z, :, :] for z in range(self.map.shape[0])]

reference_data = Measurement_Data()  ## probably from calibration
sample_data = Measurement_Data()## sample measurement
sample_info = Measurement_Info()
reference_demod = Demod_Data()
sample_demod = Demod_Data()
vea_results = Result_List()
result_maps = None
wait_exists = True




### DATA HANDLING
# Compares the VEA settings (Reference Data vs. Measurement Data)
def compare_datasets():
    global reference_data
    global sample_data
    properties = ["start_frequency","end_frequency","frequency_datapoints","sweep_type","sweep_direction","sines_number","sines_pnts"]
    for prop in properties:
        if reference_data.vea_configuration["property"][prop] != sample_data.vea_configuration["property"][prop]:
            print(f"Error: '{prop}' mismatch between Calibration and Sample Measurement") ## change 2 
            return False
    return True 

# Reads offsets and number of datapoints for File-Loading
def get_offset_datapoints(segment: nhf_reader.NHFSegment, channel: nhf_reader.NHFDataset):
    try:
        ch_offset = segment.read_channel('channel_data_offsets')
        offset = ch_offset.dataset
        ch_datapoints = segment.read_channel('number_of_datapoints_acquired')
        datapoints = ch_datapoints.dataset
    except Exception as e:
        print(f"No offset and datapoint channel found. Probably used Nanosurf Studio Version 9 or higher: {e}") 
        try:
            block_size_id = channel.attribute["dataset_block_size_source"]
            datap = segment.find_dataset_by_attribute_value("dataset_block_size_id", block_size_id) 
            #looks like [100, 120, 80, 150],each value = the block length in datapoints
            offset = np.cumsum(datap)
            # 这里offset是每个block的结束的绝对位置 [100, 220, 300, 450]
            offset  = np.insert(offset, 0, 0) #[0, 100, 220, 300, 450]
            datapoints = np.diff(offset)  ## Actual running part

        except Exception as e:
            print(f"Warning: No data with number of acquired datapoints found: {e}")
    return offset, datapoints

# Returns all Deflection Values in m
def fix_deflection_unit(ch_defl: nhf_reader.NHFDataset, measurement: nhf_reader.NHFMeasurement): #change 3 type annotation
    sensitivity, stiffness = read_file_calibration(measurement)
    if ch_defl.unit == 'V':
        ch_defl.dataset = np.asarray(ch_defl.dataset, dtype=float) * sensitivity
    elif ch_defl.unit == 'm':
        pass  # Already in metres using this file's own calibration.
    elif ch_defl.unit == 'N':
        ch_defl.dataset = np.asarray(ch_defl.dataset, dtype=float) / stiffness
    else:
        raise ValueError(f"{ch_defl.name}: unsupported deflection unit {ch_defl.unit!r}")


def read_file_calibration(measurement):
    sensitivity = float(measurement.attribute['spm_probe_calibration_deflection_sensitivity'])
    stiffness = float(measurement.attribute['spm_probe_calibration_spring_constant'])
    if not (np.isfinite(sensitivity) and sensitivity > 0 and
            np.isfinite(stiffness) and stiffness > 0):
        raise ValueError("File calibration must contain finite, positive sensitivity and spring constant")
    return sensitivity, stiffness


# Detect changes in Excitation Frequency from Meta-Data
def find_frequency_changes(data: Measurement_Data, point):
    start_index = int(data.data_offsets_vea[point])
    end_index = int(start_index + data.datapoints_acquired_vea[point]-2)  #-1
    diff = np.diff(data.ch_meta_vea.dataset[start_index:end_index])
    change_indices = np.where(diff != 0)[0]
    change_indices = np.append(change_indices, end_index)
    return change_indices + 1

# Get Data from NHF File
def import_data(source_file: pathlib.Path, type) -> bool:  ##type: calibration or measurement

    ### Open file instance ###
    nhf_file = nhf_reader.NHFFileReader(verbose=True)
    if not nhf_file.read(source_file):
        print("Could not read file")
        exit()
    if nhf_file.version() < (2, 1):
        print(f"Unknown file version: {nhf_file.version()}")
        exit()

    ### Open group instances in file ###
    measurement_name = nhf_file.measurement_name(0)
    measurement = nhf_file.measurement[measurement_name]

    ### Read this file's calibration; never reuse another file's values. ###
    sensitivity, stiffness = read_file_calibration(measurement)
    if type == "Measurement":
        data = sample_data
    elif type == "Calibration":
        data = reference_data
    else:
        raise ValueError(f"Unknown data type: {type}")
    data.deflection_sensitivity = sensitivity
    data.spring_constant = stiffness
    print(f"[{type}] {source_file.name}: sensitivity={sensitivity} m/V, k={stiffness} N/m")

    if type == "Measurement":
        sample_info.deflection_sensitivity = sensitivity
        sample_info.spring_constant = stiffness
        sample_info.offset_x = measurement.attribute['scanner_offset_x']
        sample_info.offset_y = measurement.attribute['scanner_offset_y']
        sample_info.size_x = measurement.attribute['rect_axis_range'][0]
        sample_info.size_y = measurement.attribute['rect_axis_range'][1]
        sample_info.points_x = measurement.attribute['rect_axis_size'][0]
        sample_info.points_y = measurement.attribute['rect_axis_size'][1]
        sample_info.rotation = measurement.attribute['rect_rotation']


##segment list
    segment_name_advance = 'Advance to Setpoint 1'
    segment_name_wait = 'Wait 1'
    segment_name_vea = 'VEA Sweep 1'
    segment_name_retract = 'Retract 1'

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to an in-memory buffer

    ### Load Data
    try:
        segment_vea = measurement.segment[segment_name_vea]
        data.ch_time_vea = segment_vea.read_channel('Time')
        data.ch_meta_vea = segment_vea.read_channel('Sampler Meta Channel')
        data.ch_deflection_vea = segment_vea.read_channel('Deflection')
        data.ch_z_position_vea = segment_vea.read_channel('Position Z')
        fix_deflection_unit(data.ch_deflection_vea, measurement)
        
    
        data.data_offsets_vea, data.datapoints_acquired_vea = get_offset_datapoints(segment_vea, data.ch_time_vea)
        data.vea_configuration = json.loads(segment_vea.attribute['segment_configuration'])
        if use_reference:
            data.ch_reference_vea = segment_vea.read_channel('Analyzer 2 Reference') ##reference
        
    except KeyError:
        sys.stdout = original_stdout
        sys.exit("Error: VEA-Segment missing in '{type}' Data")
    
    if type == "Measurement":
        try:
            segment_advance = measurement.segment[segment_name_advance]
            data.ch_time_advance = segment_advance.read_channel('Time')
            data.ch_meta_advance = segment_advance.read_channel('Sampler Meta Channel')
            data.ch_deflection_advance = segment_advance.read_channel('Deflection')
            data.ch_z_position_advance = segment_advance.read_channel('Position Z')
            fix_deflection_unit(data.ch_deflection_advance, measurement)
            data.data_offsets_advance, data.datapoints_acquired_advance = get_offset_datapoints(segment_advance, data.ch_time_advance)
        except KeyError:
            sys.stdout = original_stdout
            sys.exit("Error: Advance-Segment is missing in Measurement Data")
        try:
            segment_retract = measurement.segment[segment_name_retract]
            data.ch_time_retract = segment_retract.read_channel('Time')
            data.ch_meta_retract = segment_retract.read_channel('Sampler Meta Channel')
            data.ch_deflection_retract = segment_retract.read_channel('Deflection')
            data.ch_z_position_retract = segment_retract.read_channel('Position Z')
            fix_deflection_unit(data.ch_deflection_retract, measurement)
            data.data_offsets_retract, data.datapoints_acquired_retract = get_offset_datapoints(segment_retract, data.ch_time_retract)
        except KeyError:
            sys.stdout = original_stdout
            sys.exit("Error: Retract-Segment is missing in Measurement Data")
        try:
            segment_wait = measurement.segment[segment_name_wait]
            data.ch_time_wait = segment_wait.read_channel('Time')
            data.ch_meta_wait = segment_wait.read_channel('Sampler Meta Channel')
            data.ch_deflection_wait = segment_wait.read_channel('Deflection')
            data.ch_z_position_wait = segment_wait.read_channel('Position Z')
            fix_deflection_unit(data.ch_deflection_wait, measurement)
            data.data_offsets_wait, data.datapoints_acquired_wait = get_offset_datapoints(segment_wait, data.ch_time_wait)
            wait_exists = True
        except KeyError:
            wait_exists = False

    sys.stdout = original_stdout

    return True

# Create Plots
def plot_data(x1data, x2data, x3data, y1data, y2data, y3data, plot_type, info):
    
    if plot_type == "ForceSpec":
        print("Force-Spectroscopy:")
        fig1, ax1 = plt.subplots()
        ax1.plot(x1data, y1data)
        ax1.plot(x2data, y2data)
        ax1.plot(x3data, y3data, linestyle='--')
        ax1.set_xlabel('Indentation (m)')
        ax1.set_ylabel('Force (N)')
         
         
    
    elif plot_type == "Transient":
        print(f"VEA Transient at {info} Hz")
        fig1, ax1 = plt.subplots()
        #ax1.plot(x1data, y1data*1e9, 'b-', label='Force')
        ax1.plot(x1data, y1data, 'b-', label='Force')  ## sample force / measured force on sample (deflection * spring constant)
        #ax1.plot(x3data, y3data, 'g-', label='Force') ##Reference force
        ax1.set_xlabel('Time (s)')
        #ax1.set_ylabel(r'$F_{osc}$ (nN)', color='b')
        ax1.set_ylabel('F (N)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        #ax2.plot(x1data, y2data*1e6, 'r-', label='Indentation')
        ax2.plot(x1data, y2data, 'r-', label='Indentation') ##indentation
        #ax2.set_ylabel('Indentation $δ_{osc}$ (μm)', color='r')
        ax2.set_ylabel('Indentation (m)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)       
    
     
            
  

    elif plot_type == "Moduli":
        print("VEA Elastic Moduli:")
        fig1, ax1 = plt.subplots()
        ax1.plot(x1data, y1data, 'b-', label='Storage')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage Modulus (Pa)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.plot(x1data, y2data, 'r-', label='Loss')

    elif plot_type == "Moduli_Tangent":
        print("VEA Elastic Moduli:")
        fig1, ax1 = plt.subplots()
        ax1.plot(x1data, y1data, 'b-', label='Storage')
        ax1.plot(x1data, y3data, 'b--', label='Loss')
        # ax1.plot([0.25], [26.6e3], 'b', marker='o', label='Spec')
        # ax1.plot([10, 500], [50e3, 50e3], color='black', linestyle='--', label='Ref')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage Modulus (Pa)', color='b')
        # ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xscale('log')
        ax2 = ax1.twinx()
        ax2.plot(x1data, y2data, 'r-', label='Loss')
        ax2.set_ylabel('Loss Tangent', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
         
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right') 
        
    elif plot_type == "VEA cycle":
        print("VEA: Z position vs Deflection")
        fig1, ax1 = plt.subplots()
        # x1data: z_position_vea (m)
        # y1data: deflection_vea (m) -- already unit-fixed by fix_deflection_unit(...)
        ax1.plot(x1data, y1data, '-', label='vea_Deflection vs vea_Z')
        ax1.set_xlabel('Z Position (m)')
        ax1.set_ylabel('Deflection (V)')
        ax1.legend(loc='best')
        
    filename = f"ForceSpec_{info}.png"
    save_path = Path.cwd() / filename
    plt.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.show()

# Get XY-coordinates for Snake-Grid
def get_coordinates(point, width):
    Y = point // width
    X = point % width if Y % 2 == 0 else width - 1 - (point % width)
    return int(X), int(Y)



### DEMODULATION & MODULUS CALCULATIONS
# Reconstruction of the Frequency-Values
def reconstruct_frequencies(point):
    reference_demod.list_offset = find_frequency_changes(reference_data, 0)
    frequency_list = np.empty(0)

    # Read some Settings:
    start_frequency = float(reference_data.vea_configuration["property"]["start_frequency"]["value"])
    end_frequency = float(reference_data.vea_configuration["property"]["end_frequency"]["value"])
    frequency_datapoints = int(reference_data.vea_configuration["property"]["frequency_datapoints"]["value"])
    sines_pnts = int(reference_data.vea_configuration["property"]["sines_pnts"]["value"])
    sines_number = int(reference_data.vea_configuration["property"]["sines_number"]["value"])

    sweep_direction = int(reference_data.vea_configuration["property"]["sweep_direction"]["value"])
    sweep_type = int(reference_data.vea_configuration["property"]["sweep_type"]["value"])

    # Reconstruct Frequency points....
    if sweep_type == 0:
        step_frequency = (end_frequency - start_frequency) / (frequency_datapoints - 1)
    elif sweep_type == 1:
        step_frequency = (math.log(end_frequency) - math.log(start_frequency)) / (frequency_datapoints - 1)

    for i in range(frequency_datapoints):
        if sweep_type == 0:
            if sweep_direction == 0:
                current_frequency = start_frequency + i * step_frequency   
            elif sweep_direction == 1:
                current_frequency = end_frequency - i * step_frequency
        elif sweep_type == 1:
            if sweep_direction == 0:
                current_frequency = math.exp(math.log(start_frequency) + i * step_frequency)
            elif sweep_direction == 1:
                current_frequency = math.exp(math.log(end_frequency) - i * step_frequency)

        frequency_list = np.append(frequency_list, current_frequency)

    return frequency_list, sines_pnts, sines_number

# Create a low-pass Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the filter to a signal
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Define Sine-Fit
def fit_sine(t_wave, amplitude, frequency, phase, offset) -> float:
    return amplitude*np.sin(2*np.pi*t_wave*frequency+phase-1)+offset

def residuals_sine(params, x, y):
    amplitude, frequency, phase, offset = params
    return y - fit_sine(x, amplitude, frequency, phase, offset)

# Demodulation of a given signal
def demodulate_signal(ydata_vea, tdata_vea, frequencies, offsets, sines_pnts, sines_number):
    amp_list = np.empty(0)
    phase_list = np.empty(0)
    dc_list = np.empty(0)

    for i in range(len(frequencies)):
        
        # Calculate Index Ranges
        current_frequency = frequencies[i]
        start_index = offsets[i]
        end_index = offsets[i+1]

        tdata = tdata_vea[start_index:end_index]
        ydata = ydata_vea[start_index:end_index]

        # Get waves for Fitting
        tdata = tdata - tdata[0]

        # Apply Bessel-LPF to Raw Data:
        if bessel_on_raw:
            sampling_rate = (sines_pnts * current_frequency)
            ydata = lowpass_filter(ydata, current_frequency*2, sampling_rate)
            
        start_fit = int((end_index-start_index) * 0.5 * (1-fit_portion))
        end_fit = int((end_index-start_index) - start_fit)

        # Sine-Fitting Position Z:
        ###y(t)=Asin(2πft+ϕ)+offset A--amplitude f--frequency ϕ--phase offset--dc
        amp_guess = abs(np.max(ydata) - np.min(ydata)) / 2.0
        offset_guess = (np.max(ydata) + np.min(ydata)) / 2.0
        p0 = np.array([amp_guess, current_frequency, 0.9*np.pi, offset_guess])
        bounds = ([0, 0.999*current_frequency, 0, -np.inf], [np.inf, 1.001*current_frequency, 2*np.pi, np.inf])

        result = opt.least_squares(residuals_sine, p0, bounds=bounds, args=(tdata[start_fit:end_fit], ydata[start_fit:end_fit]), gtol=2.23e-16, xtol=2.23e-16)
        popt = result.x

        amp_list = np.append(amp_list, popt[0])
        phase_list = np.append(phase_list, popt[2])
        dc_list = np.append(dc_list, popt[3])

    phase_list = np.unwrap(phase_list)

    if savitzky_golay_on_demod:
        amp_list = savgol_filter(amp_list,   int(np.floor(len(frequencies)/3)), 3)
        phase_list = savgol_filter(phase_list,   int(np.floor(len(frequencies)/3)), 3)
    
    return amp_list, phase_list, dc_list

# Calculate the complex Modulus
def evaluate_moduli(point, measu_stem, measu_dir):
    offsets = find_frequency_changes(reference_data, 0)
    offsets_ref = offsets
    frequencies, sines_pnts, sines_number = reconstruct_frequencies(point)

    sample_start_index = int(sample_data.data_offsets_vea[point])
    sample_end_index = int(sample_data.datapoints_acquired_vea[point]) + sample_start_index
    reference_start_index = int(reference_data.data_offsets_vea[0])
    reference_end_index = int(reference_data.datapoints_acquired_vea[0]) + reference_start_index

    # Demodulate Reference Measurement Data
    reference_time = reference_data.ch_time_vea.dataset[reference_start_index:reference_end_index]
    reference_deflection = reference_data.ch_deflection_vea.dataset[reference_start_index:reference_end_index]
    reference_position = reference_data.ch_z_position_vea.dataset[reference_start_index:reference_end_index]
    reference_data.calc_indentation_vea = -(reference_position + reference_deflection)  # Calculate indentation 
    reference_indentation = reference_data.calc_indentation_vea

    if point == 0:
        reference_demod.list_deflection_amp, reference_demod.list_deflection_phase, reference_demod.list_deflection_dc = demodulate_signal(
            reference_deflection, reference_time, frequencies, offsets, sines_pnts, sines_number
        )
        reference_demod.list_indentation_amp, reference_demod.list_indentation_phase, reference_demod.list_indentation_dc = demodulate_signal(
            reference_indentation, reference_time, frequencies, offsets, sines_pnts, sines_number
        )
        reference_demod.list_position_z_amp, reference_demod.list_position_z_phase, reference_demod.list_position_z_dc = demodulate_signal(
            reference_position, reference_time, frequencies, offsets, sines_pnts, sines_number
        )
        reference_demod.list_frequency = frequencies
        reference_demod.list_offset = offsets

        if use_reference:
            reference_reference = reference_data.ch_reference_vea.dataset[reference_start_index:reference_end_index]
            reference_demod.list_reference_amp, reference_demod.list_reference_phase, reference_demod.list_reference_dc = demodulate_signal(
                reference_reference, reference_time, frequencies, offsets, sines_pnts, sines_number
            )    

    # Demodulate Sample Measurement Data
    offsets = find_frequency_changes(sample_data, point)
    sample_time = sample_data.ch_time_vea.dataset[sample_start_index:sample_end_index]
    sample_deflection = sample_data.ch_deflection_vea.dataset[sample_start_index:sample_end_index]
    sample_position = sample_data.ch_z_position_vea.dataset[sample_start_index:sample_end_index]
    sample_indentation = sample_data.calc_indentation_vea

    sample_demod.list_deflection_amp, sample_demod.list_deflection_phase, sample_demod.list_deflection_dc = demodulate_signal(
        sample_deflection, sample_time, frequencies, offsets, sines_pnts, sines_number
    )
    sample_demod.list_indentation_amp, sample_demod.list_indentation_phase, sample_demod.list_indentation_dc = demodulate_signal(
        sample_indentation, sample_time, frequencies, offsets, sines_pnts, sines_number
    )
    sample_demod.list_position_z_amp, sample_demod.list_position_z_phase, sample_demod.list_position_z_dc = demodulate_signal(
        sample_position, sample_time, frequencies, offsets, sines_pnts, sines_number
    )
    
    sample_demod.list_frequency = frequencies
    sample_demod.list_offset = offsets

    if use_reference:
        sample_reference = sample_data.ch_reference_vea.dataset[sample_start_index:sample_end_index]
        sample_demod.list_reference_amp, sample_demod.list_reference_phase, sample_demod.list_reference_dc = demodulate_signal(
            sample_reference, sample_time, frequencies, offsets, sines_pnts, sines_number
        )

    if 0<= plot_transient <= len(frequencies)-1:
        plot_data(sample_time[offsets[plot_transient]:offsets[plot_transient+1]] - np.min(sample_time[offsets[plot_transient]:offsets[plot_transient+1]]), 
                  None, 
                  reference_time[offsets_ref[plot_transient]:offsets_ref[plot_transient+1]] - np.min(reference_time[offsets_ref[plot_transient]:offsets_ref[plot_transient+1]]),
                  sample_deflection[offsets[plot_transient]:offsets[plot_transient+1]] * sample_data.spring_constant,
                  sample_indentation[offsets[plot_transient]:offsets[plot_transient+1]],
                  reference_deflection[offsets_ref[plot_transient]:offsets_ref[plot_transient+1]] * reference_data.spring_constant,
                  "Transient", frequencies[plot_transient])
        
    # --- NEW: Plot Z vs Deflection (only once or on selected point) ---
    if point == 0:   # 只画map中的第一个点
        try:
            print("[DEBUG] Plotting Z vs Deflection for point 0")
     
        except Exception as e:
            print("[ERROR] Z_vs_Deflection plotting failed:", e)    

    # Calculate VEA-Results
    vea_results.list_frequency = frequencies

    if use_reference:
        reference_indentation_complex = reference_demod.list_indentation_amp * np.exp(1j * (reference_demod.list_indentation_phase - reference_demod.list_reference_phase))
        reference_deflection_complex = reference_demod.list_deflection_amp * np.exp(1j * (reference_demod.list_deflection_phase - reference_demod.list_reference_phase))
        sample_indentation_complex = sample_demod.list_indentation_amp * np.exp(1j * (sample_demod.list_indentation_phase - sample_demod.list_reference_phase)) ##change 4
        sample_deflection_complex = sample_demod.list_deflection_amp * np.exp(1j * (sample_demod.list_deflection_phase - sample_demod.list_reference_phase))
    else:
        reference_indentation_complex = reference_demod.list_indentation_amp * np.exp(1j * reference_demod.list_indentation_phase)
        reference_deflection_complex = reference_demod.list_deflection_amp * np.exp(1j * reference_demod.list_deflection_phase)
        sample_indentation_complex = sample_demod.list_indentation_amp * np.exp(1j * sample_demod.list_indentation_phase)
        sample_deflection_complex = sample_demod.list_deflection_amp * np.exp(1j * sample_demod.list_deflection_phase)

    sample_response_complex = sample_deflection_complex / sample_indentation_complex
    # sample_response_complex = (sample_deflection_complex - reference_deflection_complex) / sample_indentation_complex

    # Correction for Hydrodynamic Drag
    if correct_drag:
        # factor below contains k_sample: convert reference response to force
        # with k_reference before subtraction (reduces to original for equal k).
        H_func = (reference_data.spring_constant / sample_data.spring_constant) * reference_deflection_complex / reference_indentation_complex
        # H_func = (reference_demod.list_deflection_amp / reference_demod.list_indentation_amp) * np.exp(-reference_demod.list_deflection_phase)
        sample_response_complex -= H_func

    # Calculate Geometry Factors for Contact Mechanics
    if analysis_model == "Hertz" or analysis_model == "DMT_Sphere":
        factor = ( (1 - poisson_ratio) * sample_data.spring_constant) / (4 * np.sqrt(tip_radius) * np.sqrt(sample_demod.list_indentation_dc))
    elif analysis_model == "Sneddon" or analysis_model == "DMT_Cone":
        factor = ( (1 - poisson_ratio) * sample_data.spring_constant * np.pi) / (8 * np.tan(np.deg2rad(cone_half_angle)) * sample_demod.list_indentation_dc)
    elif analysis_model == "Pyramid":
        factor = ( (1 - poisson_ratio) * sample_data.spring_constant) / (2 * np.sqrt(2) * np.tan(np.deg2rad(cone_half_angle)) * sample_demod.list_indentation_dc)

    # Calculate Moduli
    if sample_data.vea_configuration["property"]["output_id"]["value"] == "1":
    # if False:
        print("Using CleanDrive Excitation Method")
        G_complex_shear = factor * ( (reference_data.spring_constant * reference_deflection_complex /
                                     (sample_data.spring_constant * sample_deflection_complex)) - 1)
    else:
        print("Using Piezo Excitation Method")
        G_complex_shear = factor * sample_response_complex

    E_complex_elastic = 2 * G_complex_shear * (1 + poisson_ratio)
    vea_results.list_modulus_storage = np.real(E_complex_elastic)
    vea_results.list_modulus_loss = np.imag(E_complex_elastic)
    vea_results.list_loss_tangent = (vea_results.list_modulus_loss / vea_results.list_modulus_storage)

        # ---- Save results into a txt file ----
    output_filename = measu_dir / f"{measu_stem}_vea.txt"
    header = "frequency\tmodulus_storage\tloss_tangent\tmodulus_loss"

    data_to_save = np.column_stack((
        frequencies,
        vea_results.list_modulus_storage,
        vea_results.list_loss_tangent,
        vea_results.list_modulus_loss
    ))

    np.savetxt(output_filename, data_to_save, header=header, comments="", fmt="%.6e")
    print(f"[INFO] Results saved to {output_filename}")
    
    if plot_moduli or point==2 or point==130:
        # plot_data(frequencies, None, None, vea_results.list_modulus_storage, vea_results.list_modulus_loss, None, "Moduli", "")
        plot_data(frequencies, None, None, vea_results.list_modulus_storage, vea_results.list_loss_tangent, vea_results.list_modulus_loss, "Moduli_Tangent", "")
    
    
    return True



### FORCE-DISTANCE EVALUATION
# Define Fit-Functions
def fit_linear(x: np.ndarray, slope: float, offset: float) -> np.ndarray:
    return slope*x+offset

def fit_hertz(x, e_eff, x0):
    a_sphere = (4/3)/(1-poisson_ratio**2)*np.sqrt(tip_radius)*1e9
    #return np.where(x > x0, a_sphere * e_eff * (x-x0)**1.5, 0)
    return a_sphere * e_eff * np.power(np.maximum(x - x0, 0), 1.5) ##change  7

def fit_sneddon(x, e_eff, x0):
    a_cone = (2/np.pi)/(1-poisson_ratio**2)*np.tan(np.deg2rad(cone_half_angle))*1e9
    return np.where(x > x0, a_cone * e_eff * (x-x0)**2, 0)

def fit_pyramid(x, e_eff, x0):
    a_cone = (1/np.sqrt(2))/(1-poisson_ratio**2)*np.tan(np.deg2rad(cone_half_angle))*1e9
    return np.where(x > x0, a_cone * e_eff * (x-x0)**2, 0)

def fit_dmt_sphere(x, e_eff, x0, gamma):
    a_sphere = (4/3)/(1-poisson_ratio**2)*np.sqrt(tip_radius)*1e9
    a_adh = 2*np.pi*tip_radius*1e9
    return np.where(x > x0, a_sphere * e_eff * (x-x0)**1.5 - a_adh * gamma, 0)

def fit_dmt_cone(x, e_eff, x0, gamma):
    a_cone = 1/(1-poisson_ratio**2)*np.tan(np.deg2rad(cone_half_angle))*1e9
    a_adh = 1e9*2*np.pi/np.tan(np.deg2rad(cone_half_angle))
    return np.where(x > x0, a_cone * e_eff * (x-x0)**2 - a_adh * gamma * (x-x0), 0)

# Define Residuals-Functions
def residuals_hertz(params, x, y):
    e_eff, x0 = params
    if ignore_adhesion:
        return np.where(y > 0, y - fit_hertz(x, e_eff, x0), 0)
    else:
        return y - fit_hertz(x, e_eff, x0)

def residuals_sneddon(params, x, y):
    e_eff, x0 = params
    if ignore_adhesion:
        return np.where(y > 0, y - fit_sneddon(x, e_eff, x0), 0)
    else:
        return y - fit_sneddon(x, e_eff, x0)
    
def residuals_pyramid(params, x, y):
    e_eff, x0 = params
    if ignore_adhesion:
        return np.where(y > 0, y - fit_pyramid(x, e_eff, x0), 0)
    else:
        return y - fit_pyramid(x, e_eff, x0)

def residuals_dmt_sphere(params, x, y):
    e_eff, x0, gamma = params
    if ignore_baseline:
        return np.where(x > x0, y - fit_dmt_sphere(x, e_eff, x0, gamma), 0)
    else:
        return y - fit_dmt_sphere(x, e_eff, x0, gamma)
    
def residuals_dmt_cone(params, x, y):
    e_eff, x0, gamma = params
    if ignore_baseline:
        return np.where(x > x0, y - fit_dmt_cone(x, e_eff, x0, gamma), 0)
    else:
        return y - fit_dmt_cone(x, e_eff, x0, gamma)

# FD-Curve Evaluation
def evaluate_spec(point):

    # Define Datasets...
    start_index_advance = int(sample_data.data_offsets_advance[point])
    start_index_wait = int(sample_data.data_offsets_wait[point])
    start_index_vea = int(sample_data.data_offsets_vea[point])
    start_index_retract = int(sample_data.data_offsets_retract[point])
    end_index_advance = start_index_advance + int(sample_data.datapoints_acquired_advance[point])
    end_index_wait = start_index_wait + int(sample_data.datapoints_acquired_wait[point])
    end_index_vea = start_index_vea + int(sample_data.datapoints_acquired_vea[point])
    end_index_retract = start_index_retract + int(sample_data.datapoints_acquired_retract[point])

    z_position_advance = sample_data.ch_z_position_advance.dataset[start_index_advance:end_index_advance]
    deflection_advance = sample_data.ch_deflection_advance.dataset[start_index_advance:end_index_advance]
    z_position_wait = sample_data.ch_z_position_wait.dataset[start_index_wait:end_index_wait]
    deflection_wait = sample_data.ch_deflection_wait.dataset[start_index_wait:end_index_wait]
    z_position_vea = sample_data.ch_z_position_vea.dataset[start_index_vea:end_index_vea]
    deflection_vea = sample_data.ch_deflection_vea.dataset[start_index_vea:end_index_vea]
    z_position_retract = sample_data.ch_z_position_retract.dataset[start_index_retract:end_index_retract]
    deflection_retract = sample_data.ch_deflection_retract.dataset[start_index_retract:end_index_retract]

    # Baseline Correction
    baseline_start_index = int(len(z_position_advance) * baseline_start)
    baseline_end_index = int(len(z_position_advance) * baseline_end)

    try:
        p0 = np.array([0,0])
        popt_baseline_fit, *_ = opt.curve_fit(fit_linear, z_position_advance[baseline_start_index:baseline_end_index], deflection_advance[baseline_start_index:baseline_end_index], p0)
        tilt = fit_linear(z_position_advance, *popt_baseline_fit)
        deflection_advance = deflection_advance - tilt
        if wait_exists:
            tilt = fit_linear(z_position_wait, *popt_baseline_fit)
            deflection_wait = deflection_wait - tilt
        tilt = fit_linear(z_position_vea, *popt_baseline_fit)
        deflection_vea = deflection_vea - tilt
        tilt = fit_linear(z_position_retract, *popt_baseline_fit)
        deflection_retract = deflection_retract - tilt
    except Exception as e:
        print(f"Error: Baseline Correction Failed: {e}")
        return False
    
    # Create Indentation & Force...
    sample_data.calc_indentation_advance = -(z_position_advance + deflection_advance)
    sample_data.calc_force_advance = deflection_advance * sample_data.spring_constant
    if wait_exists:
        sample_data.calc_indentation_wait = -(z_position_wait + deflection_wait)
        sample_data.calc_force_wait = deflection_wait * sample_data.spring_constant
    sample_data.calc_indentation_vea = -(z_position_vea + deflection_vea)
    sample_data.calc_force_vea = deflection_vea * sample_data.spring_constant
    sample_data.calc_indentation_retract = -(z_position_retract + deflection_retract)
    sample_data.calc_force_retract = deflection_retract * sample_data.spring_constant

    indentation_advance = sample_data.calc_indentation_advance
    force_advance = sample_data.calc_force_advance
    indentation_retract = sample_data.calc_indentation_retract
    force_retract = sample_data.calc_force_retract

    # Prepare Fitting: Least_Squares PMA approach :)
    if analysis_model == "DMT_Sphere" or analysis_model == "DMT_Cone":
        p0 = np.array([1e8, np.max(indentation_advance)-10e-9, 0.1])
        bounds = ([1e3, np.min(indentation_advance), 0], [1e11, np.max(indentation_advance), 100])
    else:
        p0 = np.array([1e0, indentation_advance[baseline_end_index]])
        bounds = ([1e0, np.min(indentation_advance)], [1e7, np.max(indentation_advance)])  ##change 5

    if analysis_model == "Hertz":
        fit_func = residuals_hertz
        calc_func = fit_hertz
    elif analysis_model == "Sneddon":
        fit_func = residuals_sneddon
        calc_func = fit_sneddon
    elif analysis_model == "Pyramid":
        fit_func = residuals_pyramid
        calc_func = fit_pyramid
    elif analysis_model == "DMT_Sphere":
        fit_func = residuals_dmt_sphere
        calc_func = fit_dmt_sphere
    elif analysis_model == "DMT_Cone":
        fit_func = residuals_dmt_cone
        calc_func = fit_dmt_cone

    try:
        if fit_direction == "Advance":
            result = opt.least_squares(fit_func, p0, bounds=bounds, args=(indentation_advance, force_advance*1e9), gtol=2.23e-16, xtol=2.23e-16)
        elif fit_direction == "Retract":
            result = opt.least_squares(fit_func, p0, bounds=bounds, args=(indentation_retract, force_retract*1e9), gtol=2.23e-16, xtol=2.23e-16)
        popt = result.x
    except Exception as e:
        print(f"Error: FD Fitting Failed: {e}")
        return False
    
    vea_results.static_modulus = popt[0]
    vea_results.static_contact_point = popt[1]
    if analysis_model == "DMT_Sphere" or analysis_model == "DMT_Cone":
        vea_results.static_gamma = popt[2]
    else:
        vea_results.static_gamma = 0.0
    vea_results.static_snap = np.min(force_advance)
    vea_results.static_adhesion = np.min(force_retract)

    sample_data.calc_indentation_advance -= popt[1]  
    sample_data.calc_indentation_wait -= popt[1]
    sample_data.calc_indentation_vea -= popt[1]
    sample_data.calc_indentation_retract -= popt[1]

    print(f"Elastic Modulus = {popt[0]} Pa") ##change 6
    print(f"Contact Point = {popt[1]} m")

    if plot_spec or point==2 or point==130:
        popt[1]=0.0
        plot_data(indentation_advance, indentation_retract, indentation_advance, force_advance, force_retract, calc_func(indentation_advance, *popt)*1e-9, "ForceSpec", "")

    return True



### THE MAIN


def select_multiple_files(title: str):
    """
    Open one dialog to select multiple files at once.
    Returns list[Path].
    """
    root = tk.Tk()
    root.withdraw()  # hide the root window
    paths = filedialog.askopenfilenames(title=title)  # multi-select
    root.update()
    root.destroy()
    if not paths:
        sys.exit(f"Error: No files selected for '{title}'")
    return [Path(p) for p in paths]


def filename_pair_key(path: Path, suffix: str):
    path = Path(path)
    if path.suffix.lower() != ".nhf" or not path.stem.endswith(suffix):
        raise ValueError(f"Expected <key>{suffix}.nhf, got: {path.name}")
    key = path.stem[:-len(suffix)]
    if not key:
        raise ValueError(f"Empty pairing key: {path.name}")
    return key


def match_file_pairs(calib_files, meas_files):
    def index_files(paths, suffix):
        indexed = {}
        for path in paths:
            path = Path(path)
            key = filename_pair_key(path, suffix)
            if key in indexed:
                raise ValueError(f"Duplicate pairing key {key!r}: {indexed[key]} and {path}")
            indexed[key] = path
        return indexed

    calibrations = index_files(calib_files, CALIBRATION_SUFFIX)
    measurements = index_files(meas_files, MEASUREMENT_SUFFIX)
    missing_calib = sorted(measurements.keys() - calibrations.keys())
    missing_meas = sorted(calibrations.keys() - measurements.keys())
    if missing_calib or missing_meas:
        raise ValueError(f"Missing calibration for: {missing_calib}; "
                         f"missing measurement for: {missing_meas}")
    return [(calibrations[key], measurements[key]) for key in sorted(calibrations)]


def process_pair(calib_path: Path, meas_path: Path, strict: bool = False):
    # Also validate direct callers and isolate state between pairs.
    match_file_pairs([calib_path], [meas_path])
    global reference_data, sample_data, sample_info, reference_demod, sample_demod, vea_results
    reference_data = Measurement_Data()
    sample_data = Measurement_Data()
    sample_info = Measurement_Info()
    reference_demod = Demod_Data()
    sample_demod = Demod_Data()
    vea_results = Result_List()
    print("\n=== Processing pair ===")
    print(f"[Calibration] {calib_path}")
    done = import_data(calib_path, "Calibration")
    if not done:
        print("[FATAL] Failed to import Calibration — skip this pair.")
        return False

    # 记录当前（校准）参数，帮助你对比
    try:
        print(f"[DEBUG] After Calibration: "
              f"defl_sens={reference_data.deflection_sensitivity}, k={reference_data.spring_constant}")
    except Exception:
        pass

    print(f"[Measurement] {meas_path}")
    done = import_data(meas_path, "Measurement")
    if not done:
        print("[FATAL] Failed to import Measurement — skip this pair.")
        return False

    # ------- 关键自检：measurement 的栅格与频点 -------
    # sample_info 必须是“测量”对应的；这里强制检查
    try:
        px = int(sample_info.points_x)
        py = int(sample_info.points_y)
    except Exception as e:
        print(f"[FATAL] sample_info missing or invalid after Measurement: {e}")
        return False

    n_pts = px * py
    print(f"[DEBUG] Grid from Measurement: points_x={px}, points_y={py}, total={n_pts}")
    if n_pts <= 0:
        print("[FATAL] No points to process (n_pts <= 0). "
              "Check that the selected Measurement has a valid grid/segment.")
        return False

    try:
        n_fd = int(reference_data.vea_configuration['property']['frequency_datapoints']['value'])
        print(f"[DEBUG] frequency_datapoints={n_fd}")
    except Exception as e:
        print(f"[FATAL] Missing frequency_datapoints in reference_data: {e}")
        return False

    measu_path = Path(meas_path)
    measu_stem = measu_path.stem
    measu_dir  = measu_path.parent

    num_stacks  = 3 * n_fd
    result_maps = Result_Maps(num_stacks + 5, py, px)
    data_labels = ['Contact Point','Youngs Modulus','DMT Gamma','Force of Snap-In','Force of Adhesion']
    data_units  = ['m','Pa','arb','N','N']

    ok_pts = 0
    fail_pts = []
    did_compare = False

    for point in range(n_pts):
        try:
            print(f"[DEBUG] -> point {point}")

            d1 = evaluate_spec(point)
            if not d1:
                msg = f"point {point}: evaluate_spec failed"
                if strict:
                    print("[ERROR] " + msg + " — skip pair.")
                    return False
                print("[WARN] " + msg + " — continue.")
                fail_pts.append((point, "evaluate_spec"))
                continue

            if not did_compare:
                cmp_ok = compare_datasets()
                if not cmp_ok:
                    print("[FATAL] VEA settings mismatch — skip pair.")
                    return False
                did_compare = True

            d2 = evaluate_moduli(point, measu_stem, measu_dir)
            if not d2:
                msg = f"point {point}: evaluate_moduli failed"
                if strict:
                    print("[ERROR] " + msg + " — skip pair.")
                    return False
                print("[WARN] " + msg + " — continue.")
                fail_pts.append((point, "evaluate_moduli"))
                continue

            x, y = get_coordinates(point, px)
            # Static
            result_maps.set_result(0, y, x, vea_results.static_contact_point)
            result_maps.set_result(1, y, x, vea_results.static_modulus)
            result_maps.set_result(2, y, x, vea_results.static_gamma)
            result_maps.set_result(3, y, x, vea_results.static_snap)
            result_maps.set_result(4, y, x, vea_results.static_adhesion)

            # Dynamic
            freqs = sample_demod.list_frequency
            for i in range(len(freqs)):
                result_maps.set_result(5 + i, y, x, vea_results.list_modulus_storage[i])
                if point == 0:
                    data_labels.append(f"E Store {np.round(vea_results.list_frequency[i])} Hz")
                    data_units.append('Pa')
            for i in range(len(freqs)):
                result_maps.set_result(len(freqs) + 5 + i, y, x, vea_results.list_modulus_loss[i])
                if point == 0:
                    data_labels.append(f"E Loss {np.round(vea_results.list_frequency[i])} Hz")
                    data_units.append('Pa')
            for i in range(len(freqs)):
                result_maps.set_result(2*len(freqs) + 5 + i, y, x, vea_results.list_loss_tangent[i])
                if point == 0:
                    data_labels.append(f"Loss Tangent {np.round(vea_results.list_frequency[i])} Hz")
                    data_units.append('arb')

            ok_pts += 1

        except Exception as e:
            print(f"[WARN] point {point}: {type(e).__name__}: {e} — continue.")
            traceback.print_exc()
            fail_pts.append((point, "exception"))
            continue

    # ------- 结算 -------
    print(f"[SUMMARY] {ok_pts}/{n_pts} points processed; failures={len(fail_pts)}.")
    if ok_pts == 0:
        print("[FATAL] No points succeeded — this pair is considered FAILED.")
        return False

    print("Pair finished successfully.")
    return True


if __name__ == "__main__":
    print("Select one or more calibration files")
    calib_files = select_multiple_files("Select Calibration Measurement(s)")

    print("Select one or more measurement files")
    meas_files = select_multiple_files("Select Sample Measurement(s)")

    try:
        pairs = match_file_pairs(calib_files, meas_files)
    except ValueError as exc:
        sys.exit(f"[FATAL] Filename pairing failed: {exc}")

    print("Verified filename pairs:")
    for calib_path, meas_path in pairs:
        print(f"  {calib_path.name}  ->  {meas_path.name}")

    for idx, (calib_path, meas_path) in enumerate(pairs, start=1):
        print(f"\nProcessing Pair #{idx}:")
        try:
            ok = process_pair(calib_path, meas_path)
        except Exception as exc:
            print(f"[ERROR] Pair #{idx}: {exc}")
            traceback.print_exc()
            ok = False
        if not ok:
            print(f"Pair #{idx} failed. Continuing to next pair.")

    # export_gwy()
    """
    gwy_export.savedata_gwy("test.gwy",
        size_info=gwy_export.GwySizeInfo(x_range=sample_info.size_x, y_range=sample_info.size_y, unit_xy="m"),
        data_sets=result_maps.get_all_results(),data_labels=data_labels,data_units=data_units)
    """
    print("Done :)")



# %% Proceed here for further evaluation steps


