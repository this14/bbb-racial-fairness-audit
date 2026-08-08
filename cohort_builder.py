"""
cohort_builder.py

Builds the race/ethnicity-labeled BBB detection cohort entirely within MIMIC-IV,
by joining:
  - record_list.csv          (MIMIC-IV-ECG: subject_id, study_id, path)
  - machine_measurements.csv (MIMIC-IV-ECG: free-text report fields -> BBB label)
  - admissions.csv            (MIMIC-IV hosp module: race, per admission)
  - patients.csv               (MIMIC-IV hosp module: gender, anchor_age)

Usage:
    python cohort_builder.py \
        --record_list /path/to/record_list.csv \
        --machine_measurements /path/to/machine_measurements.csv \
        --admissions /path/to/admissions.csv \
        --patients /path/to/patients.csv \
        --out_dir ./cohort_out
"""
import argparse
import re
import os
import pandas as pd

BBB_PATTERN = re.compile(r'bundle branch block|\brbbb\b|\blbbb\b')
NEGATION_PATTERN = re.compile(r'\bno\b|\bwithout\b|not seen|excluded|r/o|rule out|cannot exclude')


def is_bbb(text: str) -> bool:
    for m in BBB_PATTERN.finditer(text):
        start = max(0, m.start() - 30)
        window = text[start:m.start()]
        if NEGATION_PATTERN.search(window):
            continue
        return True
    return False


def normalize_race(r):
    if pd.isna(r):
        return 'Unknown/Other'
    r = r.upper()
    if r.startswith('WHITE'):
        return 'White'
    if r.startswith('BLACK'):
        return 'Black'
    if r.startswith('ASIAN'):
        return 'Asian'
    if r.startswith('HISPANIC'):
        return 'Hispanic/Latino'
    if r in ('UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER'):
        return 'Unknown/Other'
    return 'Other'


def most_common_nonnull(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    return s.value_counts().idxmax()


def build_cohort(record_list_path, machine_measurements_path, admissions_path, patients_path, out_dir,
                  min_sqi=None, exclude_unknown=False):
    os.makedirs(out_dir, exist_ok=True)

    rl = pd.read_csv(record_list_path, dtype=str)
    mm = pd.read_csv(machine_measurements_path, dtype=str, low_memory=False)
    adm = pd.read_csv(admissions_path, dtype=str, usecols=['subject_id', 'race'])
    pts = pd.read_csv(patients_path, dtype=str, usecols=['subject_id', 'gender', 'anchor_age'])

    # --- BBB labeling from free-text report fields ---
    report_cols = [c for c in mm.columns if c.startswith('report_')]
    mm['full_report'] = mm[report_cols].fillna('').agg(' '.join, axis=1).str.lower().str.strip()
    mm['bbb'] = mm['full_report'].apply(is_bbb)

    # --- race per patient: most frequent non-null admission race ---
    race_per_subj = adm.groupby('subject_id')['race'].apply(most_common_nonnull).reset_index()
    race_per_subj['race_group'] = race_per_subj['race'].apply(normalize_race)

    # --- join everything at record level ---
    mm['subject_id'] = mm['subject_id'].astype(str)
    rl['subject_id'] = rl['subject_id'].astype(str)
    rl['study_id'] = rl['study_id'].astype(str)
    mm['study_id'] = mm['study_id'].astype(str)

    cohort = mm[['subject_id', 'study_id', 'bbb']].merge(
        rl[['subject_id', 'study_id', 'path']], on=['subject_id', 'study_id'], how='left'
    )
    cohort = cohort.merge(race_per_subj[['subject_id', 'race_group']], on='subject_id', how='left')
    cohort = cohort.merge(pts, on='subject_id', how='left')
    cohort['race_group'] = cohort['race_group'].fillna('Unknown/Other')

    if exclude_unknown:
        cohort = cohort[cohort['race_group'] != 'Unknown/Other'].reset_index(drop=True)

    if min_sqi is not None:
        print("WARNING: --min_sqi was provided but SQI filtering requires reading "
              "raw waveforms; this must be implemented in preprocess.py before training.")

    cohort_path = os.path.join(out_dir, 'final_cohort_record_level.csv')
    cohort.to_csv(cohort_path, index=False)

    print(f"Saved cohort: {cohort_path}  (N={len(cohort)} records, "
          f"{cohort['subject_id'].nunique()} unique subjects)")
    print("\nRace/ethnicity subgroup distribution (unique subjects):")
    print(cohort.drop_duplicates('subject_id')['race_group'].value_counts())
    print("\nBBB prevalence by subgroup (record-level):")
    summary = cohort.groupby('race_group')['bbb'].agg(['size', 'sum', 'mean'])
    summary.columns = ['n_records', 'n_bbb_positive', 'bbb_prevalence']
    print(summary)

    return cohort


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--record_list', required=True)
    ap.add_argument('--machine_measurements', required=True)
    ap.add_argument('--admissions', required=True)
    ap.add_argument('--patients', required=True)
    ap.add_argument('--out_dir', default='./cohort_out')
    ap.add_argument('--exclude_unknown', action='store_true',
                     help='Drop patients with no admissions.csv race match')
    args = ap.parse_args()
    build_cohort(args.record_list, args.machine_measurements, args.admissions, args.patients,
                 args.out_dir, exclude_unknown=args.exclude_unknown)
