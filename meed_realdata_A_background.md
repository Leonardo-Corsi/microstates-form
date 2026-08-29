Problem A: MEED as an advancement from Zanesco-style GFP/TD analysis
MEED adds an interpretable layer to the GFP, DISS/TD, and microstate-fit logic.
Zanesco’s base result is that high-GFP samples tend to be more topographically stable and better resemble microstate configurations, whereas low-GFP samples are heterogeneous and can show a wide range of DISS/TD values. GFP is therefore useful, but incomplete. MEED extends this logic in three ways.
1.	it provides a pre-fitting field-conformity measure, (\rho^2), which asks whether the instantaneous normalized scalp map is well represented by the fixed MEED scaffold before assigning it to A/B/C/D.
2.	it places each instantaneous map in an interpretable ((\theta,\phi,\rho)) coordinate system, allowing GFP, TD, correlation, and uncertainty to be visualized directly in the MEED plane.
3.	it decomposes Zanesco-style TD into a MEED-explained component and a residual component. TD itself remains on the original (0) to (2) scale. The squared TD is used only because the MEED projection and its residual are orthogonal, so the decomposition is squared-additive.
For each recording, the required first-layer output is:
•	< Subject >_samplewise_characterization.csv, with columns:

sub, time, GFP, is_peak, TD, TD_MEED, TD_res, TD2, TD2_MEED, TD2_res, F_MEED_TD, TD_decomposition_closure_error,
d_x, d_y, d_z, theta, phi, rho, rho2, psiD,
corr_meta_A, corr_meta_B, corr_meta_C, corr_meta_D,
corr_sub_A, corr_sub_B, corr_sub_C, corr_sub_D,
psi_meta_A, psi_meta_B, psi_meta_C, psi_meta_D,
psi_sub_A, psi_sub_B, psi_sub_C, psi_sub_D,
phi_meta_A, phi_meta_B, phi_meta_C, phi_meta_D,
phi_sub_A, phi_sub_B, phi_sub_C, phi_sub_D,
theta_meta_A, theta_meta_B, theta_meta_C, theta_meta_D,
theta_sub_A, theta_sub_B, theta_sub_C, theta_sub_D.

is_peak should identify the exact GFP peaks used by the clustering procedure. If the exact pycrostates internal peak mask is not available, the column should be renamed to local_GFP_peak and treated only as an approximation; TD is Zanesco-style backward topographic dissimilarity/DISS, computed after average referencing and GFP normalization, with theoretical range (0) to (2); TD_MEED is the part of TD explained by adjacent changes in the fixed MEED coordinate vector (q_t=(d_x,d_y,d_z)); TD_res is the residual part of TD outside the fixed MEED scaffold; F_MEED_TD = TD2_MEED / TD2 is the fraction of squared TD explained by MEED-coordinate change; TD_decomposition_closure_error = TD2 - TD2_MEED - TD2_res is a numerical diagnostic and should be close to zero; d_x, d_y, and d_z are Cartesian MEED coordinates, i.e. the unnormalized (q_t) components. These must be used for TD decomposition; theta, phi, and rho are the spherical MEED coordinates. rho2 is (\rho^2); psiD is the angular displacement between subsequent MEED unit directions. It is useful as a directional movement index, but it is not the same as TD_MEED, because it ignores changes in (\rho); corr_* columns are absolute spatial correlations with meta or subject templates; psi_* columns are angular distances to meta or subject templates in MEED direction space; theta_* and phi_* columns are component-wise angular distances to each template.

•	< Subject >_sensor_maps.csv, with columns sub, k, label, channel, value, storing subject solutions from (k=3) to (k=8) in sensor space.

•	< Subject >_dipole_maps.csv with columns sub, k, label, component, value, storing the same subject solutions transformed into MEED coordinates. component should include at least d_x, d_y, d_z, theta, phi, rho, and rho2.

Item 1: Zanesco plots, rho as pre-fitting microstateishness, and TD decomposition
Purpose: replicate the Zanesco-style GFP/TD logic and show what MEED adds.
Required visual outputs for the final paper:
•	Figure 1A: Zanesco replication panel. Hexbin plots, each shown both lin-lin and log-log: TD vs GFP, rho vs GFP, rho2 vs GFP, psiD vs GFP, psiD vs TD.

This shows whether the recording reproduces the expected GFP/TD structure and whether MEED angular displacement behaves similarly to TD.

•	Figure 1B: GFP-peak overlay panel. Same relationships as Figure 1A, but with all samples shown as background density and GFP-peak samples overlaid as points. This tests whether clustering-selected GFP peaks occupy the expected high-GFP, low-TD, high-rho region.

•	Figure 1C: TD decomposition panel. Core plots: TD_MEED vs TD, TD_res vs TD, F_MEED_TD vs GFP, F_MEED_TD vs rho2; distribution of F_MEED_TD; distribution of TD_decomposition_closure_error.
This is the key figure for showing how much of Zanesco’s TD is explained by MEED-coordinate change.
Required tabular outputs:
Table 1: Subject-level summary of Zanesco and MEED-TD quantities.
For each recording: median GFP, median TD, median TD_MEED, median TD_res, median F_MEED_TD, IQR of F_MEED_TD, correlation between TD and psiD, correlation between TD and TD_MEED, correlation between TD and TD_res, median absolute TD_decomposition_closure_error.
Interpretation:
GFP says whether the field is strong; TD says whether the normalized scalp map changes; rho-square says whether the instantaneous map is geometrically compatible with the fixed MEED scaffold; psiD says whether the MEED direction changes; TD_MEED says how much adjacent topographic change is explained by changes in the full MEED coordinate vector, including rho; TD_res says how much adjacent topographic change remains outside the fixed MEED scaffold.
The central claim is not that MEED replaces TD. The claim is that MEED decomposes TD into an interpretable dipole-coordinate component and a residual component.
