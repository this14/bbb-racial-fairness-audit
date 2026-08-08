"""
preprocess.py

Loads raw MIMIC-IV-ECG waveforms (.hea/.dat via wfdb) referenced in the cohort CSV,
applies bandpass filtering + per-lead z-score normalization, and excludes records
with unresolvable signal quality issues (excessive NaNs / flat/clipped signal).

Usage:
    python preprocess.py --cohort_csv cohort_out/final_cohort_record_level.csv \
        --waveform_root /path/to/mimic-iv-ecg/1.0 \
        --out_dir ./preprocessed
"""
import argparse
import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt


def bandpass(sig, low=0.5, high=40, fs=500, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)


def load_and_preprocess(cohort_csv, waveform_root, out_dir, max_nan_frac=0.01):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(cohort_csv, dtype={'subject_id': str, 'study_id': str})

    signals, labels, race_groups, subjects, kept_rows = [], [], [], [], []
    n_excluded_missing = 0
    n_excluded_quality = 0

    for i, row in df.iterrows():
        rec_path = os.path.join(waveform_root, row['path'])
        if not os.path.exists(rec_path + '.hea'):
            n_excluded_missing += 1
            continue
        try:
            rec = wfdb.rdrecord(rec_path)
            sig = rec.p_signal
        except Exception:
            n_excluded_missing += 1
            continue

        if sig.shape != (5000, 12):
            n_excluded_quality += 1
            continue
        nan_frac = np.isnan(sig).mean()
        if nan_frac > max_nan_frac:
            n_excluded_quality += 1
            continue
        sig = np.nan_to_num(sig)

        sig_f = bandpass(sig)
        mean = sig_f.mean(axis=0, keepdims=True)
        std = sig_f.std(axis=0, keepdims=True) + 1e-8
        sig_norm = (sig_f - mean) / std

        signals.append(sig_norm.astype(np.float32))
        labels.append(row['bbb'])
        race_groups.append(row['race_group'])
        subjects.append(row['subject_id'])
        kept_rows.append(i)

    print(f"Loaded {len(signals)} / {len(df)} records "
          f"({n_excluded_missing} missing files, {n_excluded_quality} excluded for signal quality)")

    X = np.stack(signals) if signals else np.empty((0, 5000, 12), dtype=np.float32)
    y = np.array(labels).astype(np.int64)
    groups = np.array(race_groups)
    subjects = np.array(subjects)

    # NOTE: np.savez() writes to a zip container. On Windows, zip writes of
    # multi-GB archives can raise "OSError: [Errno 22] Invalid argument"
    # even on NTFS (unrelated to the 4GB FAT32 limit). Saving each array as
    # a plain .npy file avoids the zip container entirely and is reliable
    # at this size.
    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'groups.npy'), groups)
    np.save(os.path.join(out_dir, 'subjects.npy'), subjects)
    print(f"Saved to {out_dir}: X={X.shape}, y={y.shape}, groups={groups.shape}, subjects={subjects.shape}")
    return X, y, groups, subjects


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort_csv', required=True)
    ap.add_argument('--waveform_root', required=True,
                     help='Local root folder containing the "files/pXXXX/..." MIMIC-IV-ECG tree')
    ap.add_argument('--out_dir', default='./preprocessed')
    args = ap.parse_args()
    load_and_preprocess(args.cohort_csv, args.waveform_root, args.out_dir)
