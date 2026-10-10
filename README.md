# AFM-Cell-Mechanics-Analysis

This repository contains analysis code for AFM-based cell mechanics analysis used in the manuscript:

**Sequential multiscale mechanics reveals two dominant response modes across timescales in living cells**

The code includes analysis for indentation mechanics, relaxation, rheology, adhesion, tether force, and related data processing developed during this project. 


AFM force-curve data were collected using a Nanosurf DriveAFM with Nanosurf software version 13.6 and saved in `.nhf` format.

## 1. Young's Modulus, Relaxation Time, and Adhesion

Run the corresponding Python analysis script and select the following sample file when prompted:

`mcf7 ddmim ctrl_5_2.nhf`

## 2. Rheology Analysis

### Complex Modulus

Run:

`Rheology_complex modulus_fig2d.py`

When prompted:

- Select `mcf7 ddmim ctrl_5_1.nhf` as the calibration file.
- Select `mcf7 ddmim ctrl_5_2.nhf` as the measurement file.
- Save the output as `mcf7 ddmim ctrl_5_2_vea.txt`.

### SPL α and DPL β Fitting

Run:

`SPL_alpha_then_DPL_beta_fitting_fig2e.py`

Select:

`mcf7 ddmim ctrl_5_2_vea.txt`

Save the results in the `SPL_alpha_DPL_beta` folder.

### Loss Tangent

Run:

`loss tangent_fig 2f.py`

Select:

`mcf7 ddmim ctrl_5_2_vea.txt`
