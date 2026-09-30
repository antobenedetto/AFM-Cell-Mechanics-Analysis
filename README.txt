AFM experimental data force curves collected from Nanosurf software 13.6. (.nhf file)

(1) For young's modulus, relaxation time and adhesion
	--run the python script
	--select sample data "mcf7 ddmim ctrl_5_2.nhf"

(2) For rheology
	--run the Rheology_complex modulus_fig2d script
	--select calibration file "mcf7 ddmim ctrl_5_1.nhf"
	--select measurement file "mcf7 ddmim ctrl_5_2.nhf"
	--save result "mcf7 ddmim ctrl_5_2_vea.txt"

	--run the SPL_alpha_then_DPL_beta_fitting_fig2e script
	--select  "mcf7 ddmim ctrl_5_2_vea.txt"
	--save result in folder SPL_alpha_DPL_beta


	--run loss tangent_fig 2f script
	--select  "mcf7 ddmim ctrl_5_2_vea.txt"
	





