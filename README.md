# FORM microstate analysis

This repository contains the code accompanying the FORM manuscript. FORM is a
fixed three-dimensional representation of scalp electric fields: it projects
each average-referenced EEG topography onto the three-dimensional sensor
geometry, separating the part explained by the fixed FORM subspace from the
residual field. The method stage documents this construction, its relation to
literature templates, its orientation views, and a controlled spatial-spectrum
simulation. The LEMON analysis then tests the representation on continuous
resting EEG. It estimates recording-level and pooled conventional microstate
maps, performs instantaneous pooled-group backfitting, calculates conventional
GEV, and evaluates FORM temporal decomposition, conformity, and finite
dictionary behaviour. Outputs are vector figures, unrounded tables, and a
final manuscript package. The code is intentionally a direct sequence of
Python scripts rather than a package or workflow framework.

## Usage
Create one working directory containing sibling `LEMON`, `microstate-form`, and
`results` directories. Install the packages in `requirements.txt`, and ensure
that `wget` is available. These commands download the preprocessed EEG and
the behavioural/subject metadata into the layout expected by the preparation
script. The preparation follows the LEMON analysis described in

 Zanesco, A.P. EEG Electric Field Topography is Stable
During Moments of High Field Strength. *Brain Topogr* **33**, 450-460 (2020).
[https://doi.org/10.1007/s10548-020-00780-7](https://doi.org/10.1007/s10548-020-00780-7)

To download with wget:
```bash
mkdir form-work && cd form-work
wget -r -np -nH --cut-dirs=4 -P LEMON https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/
wget -r -np -nH --cut-dirs=3 -P LEMON https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/Behavioural_Data_MPILMBB_LEMON/
git clone https://github.com/Leonardo-Corsi/microstate-form.git
cd microstate-form
python -m pip install -r requirements.txt
```

Run the analysis in this order:

- `python form_lemon_dataset.py` - standardizes downloaded EEG and creates vetted FIF inputs.
- `python form_method.py --spatial-spectrum --literature-mds --literature-conformity --orientation-assets` - produces data-free FORM method figures and tables.
- `python form_validation_a_subject_level.py` - extracts peaks, fits recording models, caches FORM quantities.
- `python form_validation_b_group_level.py` - builds the canonical pooled condition-insensitive A-E dictionary.
- `python form_validation_c_backfit.py` - backfits pooled maps and writes conventional GEV outputs.
- `python form_validation_d_descriptive.py` - produces FORM empirical analyses, tables, and figures.
- `python form_validation_f_figuresandtables.py --overwrite` - copies validated leaves into the manuscript package.

For a small check, pass `--only-id sub-XXXXXX` to A, C, or D; scoped runs use one worker and do not overwrite cohort-level Stage-C tables.