# Racial Fairness Audit of Bundle Branch Block Detection (MIMIC-IV-ECG)

Analysis code accompanying the manuscript:

**"Toward FHIR-Ready Algorithmic Fairness Auditing in Cardiovascular AI: A Racial
Subgroup Analysis of Bundle Branch Block Detection Using Real-World ECG Data"**
Chuelwon Lee, RAQA Team, HUINNO Co., Ltd.
Submitted to Healthcare Informatics Research (HIR).

## Overview

This repository contains the full pipeline used to:
1. Build a race/ethnicity-labeled bundle branch block (BBB) cohort from MIMIC-IV-ECG
   (`cohort_builder.py`)
2. Preprocess raw ECG waveforms (`preprocess.py`)
3. Define the model architecture, a modified 1D-ResNet-34 (`resnet1d.py`)
4. Train the model with a subject-level 70/10/20 split (`train.py`)
5. Evaluate overall and per-subgroup performance with bootstrap CIs and pairwise
   DeLong-variance significance testing (`evaluate.py`)

## Data access

This code operates on the MIMIC-IV-ECG database, which requires credentialed access
via PhysioNet (https://physionet.org) under its data use agreement. This repository
does not include any patient data. Users must obtain their own PhysioNet credentialing
and download access before running this pipeline.

## Pipeline

```bash
# 1. Build the cohort (requires MIMIC-IV-ECG record_list.csv, machine_measurements.csv,
#    and the linked MIMIC-IV hosp module's admissions.csv and patients.csv)
python cohort_builder.py \
    --record_list /path/to/record_list.csv \
    --machine_measurements /path/to/machine_measurements.csv \
    --admissions /path/to/admissions.csv \
    --patients /path/to/patients.csv \
    --out_dir ./cohort_out

# 2. Preprocess waveforms for the sampled cohort CSV
python preprocess.py \
    --cohort_csv /path/to/sampled_cohort.csv \
    --waveform_root /path/to/mimic-iv-ecg/1.0 \
    --out_dir ./preprocessed

# 3. Train (subject-level 70/10/20 split, modified 1D-ResNet-34)
python train.py --data_dir ./preprocessed --out_dir ./model_out

# 4. Evaluate (overall + per-subgroup AUROC/sensitivity/specificity, bootstrap CIs,
#    pairwise DeLong-variance z-tests)
python evaluate.py --data_dir ./preprocessed --model_dir ./model_out --out_dir ./eval_out
```

## Requirements

Python 3.13.7. See `requirements.txt` for pinned package versions used to generate
the results reported in the manuscript.

## Citation

If you use this code, please cite the manuscript above (citation details to be added
upon acceptance/publication).

## License

[TO DECIDE: choose a license, e.g. MIT, before making the repository public.]
