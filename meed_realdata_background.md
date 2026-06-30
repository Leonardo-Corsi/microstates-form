# MEED real-data application: current work-in-progress scope

## Current scope

The real-data application has been reset to a minimal first layer.

The validated method notebook remains separate. `MEED-method.ipynb` contains the MEED derivation, map-space diagnostics, fixed-versus-full dipole comparison, reconstruction analyses, leave-one-electrode sensitivity, and synthetic/null simulations. Those simulations already show the nominal behavior that matters for the method: fixed MEED is constrained, rejects iid or high-order random topographies, and accepts low-order smooth topographies that resemble the field family where the metamaps live.

The current real-data work does not touch that notebook.

The new files are:

- `MEED-realdata.ipynb`
- `utils_realdata.py`

The goal is not yet to decide cleaning thresholds, smoothing order, or sequence metrics. The goal is only to verify that MEED can be applied cleanly to real already-cleaned EDFs.

---

## Input data convention

The current export format is:

- cleaned EDF files
- optional same-stem CSV sidecars
- EDF channel types already mark only real EEG channels as `eeg`

The real-data notebook therefore uses:

```python
input_files = list(DATA_ROOT.glob("*.*"))
edf_files = [p for p in input_files if p.suffix.lower() == ".edf"]
csv_files = [p for p in input_files if p.suffix.lower() == ".csv"]
```

For now, CSV contents are not used for group assignment. Two provisional groups are defined as:

```python
group_0 = edf_files[0::2]
group_1 = edf_files[1::2]
```

The helper `standard_names_picker()` only standardizes labels, sets a montage when possible, and keeps EEG channels. It does not try to infer non-EEG channels because this has already been handled in the EDF export.

---

## Existing MEED loading remains unchanged

Meta-microstates are loaded through the existing method utility:

```python
ds = u.load_microstates(JSON_PATH, K)
B, ch_names, labels, info = ds["B"], ds["ch_names"], ds["labels"], ds["info"]
```

This is intentionally unchanged because the method notebook already handles metamap loading, ordering, normalization, and visualization correctly.

The real-data code builds a `meta_level` object from these maps:

```python
meta_level = ur.build_template_level("meta", B, labels, info)
```

A template level contains normalized maps, labels, MNE info, MEED geometry, MEED template directions, MEED angular coordinates, and MEED projection diagnostics.

---

## First real-data objects of interest

For each raw object, the notebook should be able to extract the following.

### 1. Subject maps and subject MEED representation

A subject-level microstate solution is fitted with pycrostates and reordered to the meta maps by absolute topographic correlation.

### 2. Meta maps and meta MEED representation

For each template map:

$$
q_m = S_{\mathrm{orth}}^\top \tilde b_m
$$

$$
q_m = \rho_m u_m
$$

$$
D_m=(\theta_m,\phi_m)
$$

where $\rho_m^2$ is a projection diagnostic and $D_m$ is the retained MEED coordinate.

### 3. Optional group maps and group MEED representation

Group maps can be fitted by concatenating aligned EDFs within each provisional group:

```python
group_0 = edf_files[0::2]
group_1 = edf_files[1::2]
```

### 4. Correlation of each sample with subject and meta maps

For each sample $x_t$, compute:

$$
\tilde x_t = \frac{x_t-\bar{x}_t\mathbf{1}}{\|x_t-\bar{x}_t\mathbf{1}\|}
$$

Then compute signed, absolute, and squared topographic correlations to each template:

$$
c_m(t)=\tilde x_t^\top \tilde b_m
$$

$$
|c_m(t)|
$$

$$
c_m(t)^2
$$

The default interpretation is polarity-insensitive, so the main quantities are $|c_m(t)|$ and $c_m(t)^2$.

### 5. MEED coordinates of each data sample

Each data sample is projected directly into MEED before using templates:

$$
q_t = S_{\mathrm{orth}}^\top \tilde x_t
$$

$$
\rho_t = \|q_t\|
$$

$$
u_t = q_t/\rho_t
$$

$$
D_t=(\theta_t,\phi_t)
$$

Here $\rho_t^2$ says how well the instantaneous topography survives the fixed MEED scaffold. This is independent from whether it is close to any specific microstate template.

### 6. Angular distance from data MEED to each template MEED

For each data direction $u_t$ and template direction $u_m$, compute:

$$
d_{\pm}(u_t,u_m)=\arccos(|u_t^\top u_m|)
$$

This is computed separately for meta, subject, and optional group templates.

---

## Current stopping point

The current notebook stops after file discovery, one-raw loading and channel alignment, meta-map loading, subject-map fitting, optional group-map fitting, template-level MEED summaries, samplewise data MEED features, samplewise correlations to templates, and samplewise angular distances to templates.

It deliberately does not yet implement MEED cleaning thresholds, smoothing-order comparisons, Pascual-Marqui smoothing experiments, A/B/C/D conditions, sequence-metric convergence, or clinical/group-level inference. Those belong to the next layer only after this first real-data extraction layer is verified.
---

## Section 3 and 4 update: explicit real-data MEED characterization

The current notebook now treats the first real-data layer as an object-extraction problem, not yet as a cleaning or sequence-metric problem.

In Section 3, each computational passage should be followed by an explicit display:

1. load and align one raw object
2. fit subject-specific pycrostates maps
3. display the subject maps
4. display normalized maps across all available template levels: meta, subject, and optional group levels
5. display the MEED coordinate table for all levels
6. plot MEED coordinates annotated by label, $\rho$, and $R^2=\rho^2$
7. display the subject-vs-meta matching table
8. plot sensor-space correlation and MEED-angle comparison matrices

The MEED coordinate plot must not only show $(\theta,\phi)$. It must also show whether a point is well described by fixed MEED. A map can visually land near another map in the angular plane while still having low $\rho$ or low $R^2$, meaning that the fixed-dipole scaffold is only weakly explaining that map. This is especially important for collapsed or non-lateralized subject-specific maps.

In Section 4, the per-sample table should contain the practical real-data features:

```text
sub
ses
time
GFP
TD
C_Ameta, C_Bmeta, ...
C_Asub, C_Bsub, ...
theta
phi
rho
theta_Ameta, phi_Ameta, angle_Ameta, ...
theta_Asub, phi_Asub, angle_Asub, ...
```

The compact correlation columns use polarity-insensitive absolute correlation:

$$
C_{m}(t)=|\tilde x_t^\top\tilde b_m|
$$

and the squared columns, when needed, are:

$$
C^2_m(t)
$$

The data MEED projection is template-free:

$$
q_t=S_{\mathrm{orth}}^\top\tilde x_t
$$

$$
\rho_t=\|q_t\|
$$

$$
u_t=q_t/\rho_t
$$

$$
D_t=(\theta_t,\phi_t)
$$

The angular distance to each template is computed after polarity alignment:

$$
d_{\pm}(u_t,u_m)=\arccos(|u_t^\top u_m|)
$$

For visualization, the notebook now generates two subject-level plots.

The first is a long time-series panel:

1. GFP as a grey filled trace, with topographic dissimilarity in Okabe-Ito reddish-purple and composite MEED angular velocity in Okabe-Ito green
2. correlations with meta maps as thick transparent lines and correlations with subject maps as thin opaque lines using the same state colors
3. data $\theta$, $\phi$, and $\rho$
4. component-wise $\theta$ distances to meta and subject templates
5. component-wise $\phi$ distances to meta and subject templates
6. composite angular distances to meta and subject templates

The second removes time and uses pairwise hexbin plots between:

- GFP
- TD
- composite MEED angular velocity
- minimum composite angle to meta maps
- minimum composite angle to subject maps
- $\rho$

Sections 5 and 6 remain unchanged for now. The optional group-level fitting and all-recording loop are not part of the current edits.
