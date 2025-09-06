#!/usr/bin/env python3
"""
Build an aggregated CC200 connectivity bundle from the best available sources, with QC.

Priority:
1) Per-subject Fisher-z FC: artifacts/multimodal_depression_gnn/fc/fc_fisherz_subject{ID}.npy
   - Converts z -> r via tanh(z), symmetrizes, zero diagonal, clamps to [-0.99, 0.99]
2) Per-subject CC200 timeseries: artifacts/atlas_aggregated_official/per_subject_cc200/*_timeseries_cc200.npy
   - Computes correlation per subject, symmetrizes, zero diagonal
3) Aggregated timeseries: artifacts/atlas_aggregated/atlas_timeseries_cc200.npy (assumed [subjects, T, 200])
   - Computes correlation per subject in order

Outputs in current directory:
- connectivity_matrices_cc200_fisherz.npy  (float32, [subjects, 200, 200])
- qc_connectivity_report.json              (basic QC summary)

Usage: python build_connectivity_bundle.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).parent


def load_subjects(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'subject_id' not in df.columns:
        # Try fallback columns
        if 'ID' in df.columns:
            df = df.rename(columns={'ID': 'subject_id'})
        else:
            raise RuntimeError('subjects CSV missing subject_id/ID')
    return df


def fisherz_dir(search_roots) -> Path | None:
    for base in search_roots:
        cand = base / 'artifacts' / 'multimodal_depression_gnn' / 'fc'
        if cand.exists():
            return cand
    return None


def per_subject_timeseries_dir(search_roots) -> Path | None:
    for base in search_roots:
        cand = base / 'artifacts' / 'atlas_aggregated_official' / 'per_subject_cc200'
        if cand.exists():
            return cand
    return None


def aggregated_timeseries_path(search_roots) -> Path | None:
    for base in search_roots:
        cand = base / 'artifacts' / 'atlas_aggregated' / 'atlas_timeseries_cc200.npy'
        if cand.exists():
            return cand
    return None


def symmetrize_zero_diag(mat: np.ndarray) -> np.ndarray:
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)
    return mat


def clamp(x: np.ndarray, lo=-0.99, hi=0.99) -> np.ndarray:
    return np.clip(x, lo, hi)


def main():
    search_roots = [ROOT, ROOT.parent]

    subj_csv = ROOT / 'subjects_with_motion_multimodal_COMPLETE.csv'
    if not subj_csv.exists():
        print('ERROR: subjects_with_motion_multimodal_COMPLETE.csv not found in package folder.')
        sys.exit(1)
    df = load_subjects(subj_csv)
    subject_ids = [str(x).strip() for x in df['subject_id'].tolist()]
    n = len(subject_ids)

    out_path = ROOT / 'connectivity_matrices_cc200_fisherz.npy'
    qc_path = ROOT / 'qc_connectivity_report.json'

    # Data sources
    fc_dir = fisherz_dir(search_roots)
    ts_dir = per_subject_timeseries_dir(search_roots)
    ts_agg = aggregated_timeseries_path(search_roots)

    conn = np.zeros((n, 200, 200), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)

    # 1) Try Fisher-z per subject
    if fc_dir is not None:
        print(f'Using Fisher-z per-subject directory: {fc_dir}')
        miss = 0
        for i, sid in enumerate(subject_ids):
            p = fc_dir / f'fc_fisherz_subject{sid}.npy'
            if p.exists():
                z = np.load(str(p))
                if z.shape != (200, 200):
                    miss += 1
                    continue
                r = np.tanh(z).astype(np.float32)
                r = symmetrize_zero_diag(r)
                r = clamp(r)
                conn[i] = r
                filled[i] = True
            else:
                miss += 1
        print(f'  Loaded Fisher-z for {filled.sum()}/{n} subjects (missing {miss})')

    # 2) Per-subject timeseries
    if not filled.all() and ts_dir is not None:
        print(f'Using per-subject timeseries: {ts_dir}')
        for i, sid in enumerate(subject_ids):
            if filled[i]:
                continue
            p = ts_dir / f'{sid}_timeseries_cc200.npy'
            if p.exists():
                ts = np.load(str(p))
                if ts.shape[0] < ts.shape[1]:
                    ts = ts  # [T, 200]
                else:
                    ts = ts.T
                if ts.shape[1] != 200:
                    continue
                c = np.corrcoef(ts.T).astype(np.float32)
                c = symmetrize_zero_diag(c)
                conn[i] = clamp(c)
                filled[i] = True

    # 3) Aggregated timeseries (assumed [subjects, T, 200])
    if not filled.all() and ts_agg is not None:
        print(f'Using aggregated timeseries: {ts_agg}')
        ts = np.load(str(ts_agg), mmap_mode='r')
        # Detect orientation
        if ts.ndim != 3:
            print(f'  WARNING: Unexpected aggregated timeseries shape {ts.shape}')
        else:
            S0, S1, S2 = ts.shape
            # Case A: [subjects, T, 200]
            if S0 >= n and S2 == 200:
                for i in range(n):
                    if filled[i]:
                        continue
                    if i >= ts.shape[0]:
                        break
                    tsi = ts[i]  # [T, 200]
                    if tsi.shape[0] < tsi.shape[1]:
                        pass
                    else:
                        tsi = tsi.T
                    c = np.corrcoef(tsi.T).astype(np.float32)
                    c = symmetrize_zero_diag(c)
                    conn[i] = clamp(c)
                    filled[i] = True
            # Case B: [T, 200, subjects]
            elif S1 == 200 and S2 >= n:
                for i in range(n):
                    if filled[i]:
                        continue
                    if i >= S2:
                        break
                    tsi = ts[:, :, i]  # [T, 200]
                    c = np.corrcoef(tsi.T).astype(np.float32)
                    c = symmetrize_zero_diag(c)
                    conn[i] = clamp(c)
                    filled[i] = True
            else:
                print(f'  WARNING: Unrecognized aggregated shape {ts.shape}; skipping aggregated fill')

    # Final QC and save
    n_filled = int(filled.sum())
    print(f'Filled {n_filled}/{n} connectivity matrices')
    if n_filled == 0:
        print('ERROR: Could not build any connectivity matrices. Check data paths.')
        sys.exit(2)

    # Save
    np.save(out_path, conn)
    print(f'Saved: {out_path} shape={conn.shape}')

    # QC report
    sample = conn[: min(100, n_filled)]
    report = {
        'n_subjects': n,
        'n_filled': n_filled,
        'shape': list(conn.shape),
        'mean': float(np.mean(sample)),
        'std': float(np.std(sample)),
        'abs_p90': float(np.percentile(np.abs(sample), 90)),
        'min': float(sample.min()),
        'max': float(sample.max()),
    }
    qc_path.write_text(json.dumps(report, indent=2))
    print('QC:', json.dumps(report))


if __name__ == '__main__':
    sys.exit(main())
