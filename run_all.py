#!/usr/bin/env python3
"""
Run-all Orchestrator for Package 9

Runs preflight checks, executes the full pipeline, and saves performance
and interpretability visuals (global + basic fallbacks) in a tidy
reports/ directory tree.
"""

import os
import json
from pathlib import Path
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PKG_DIR = Path(__file__).parent
ROOT = PKG_DIR
REPORTS = ROOT / 'reports'
PERF_DIR = REPORTS / 'performance'
INT_SUMMARY_DIR = REPORTS / 'interpretability'
INT_DIR = REPORTS / 'interpretability'
PP_DIR = REPORTS / 'per_patient'


def ensure_dirs():
    for d in [REPORTS, PERF_DIR / '80_20', PERF_DIR / '5fold', INT_DIR / 'rf' / '80_20', INT_DIR / 'rf' / '5fold',
              INT_DIR / 'gnn' / '80_20', INT_DIR / 'gnn' / '5fold', PP_DIR / '80_20', PP_DIR / '5fold']:
        d.mkdir(parents=True, exist_ok=True)


def run_preflight() -> int:
    try:
        from preflight_check import main as preflight_main
        return preflight_main()
    except Exception as e:
        print(f"Preflight error: {e}")
        return 1


def run_full_pipeline():
    try:
        from run_full_pipeline import main as run_main
        run_main()
    except Exception:
        print("Pipeline failed:")
        print(traceback.format_exc())
        raise


def load_incremental_results() -> list:
    # Prefer JSONL if present; fallback to JSON array
    inc_dir = ROOT / 'incremental_results'
    jsonl = inc_dir / 'incremental_results.jsonl'
    json_arr = inc_dir / 'incremental_results.json'
    if jsonl.exists():
        out = []
        try:
            with jsonl.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
            return out
        except Exception:
            pass
    if json_arr.exists():
        try:
            return json.loads(json_arr.read_text())
        except Exception:
            return []
    return []


def plot_metric_bar(ax, labels, values, title, ylabel):
    ax.bar(labels, values, color='#2463EB')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)


def save_performance_plots(results: list):
    # Aggregate best (or median) metrics per phase for 80/20 and 5fold
    rf_8020, rf_5, gnn_8020, gnn_5 = [], [], [], []
    for r in results:
        if r.get('status') != 'success':
            continue
        phase = r.get('phase')
        if phase == 'rf':
            v8020 = r.get('validation_80_20', {})
            v5 = r.get('validation_5fold_site_cv', {})
            if v8020:
                rf_8020.append(v8020)
            if v5 and v5.get('n_folds', 0) > 0:
                rf_5.append(v5)
        elif phase == 'gnn':
            v8020 = r.get('validation_80_20', {})
            v5 = r.get('validation_5fold_site_cv', {})
            if v8020:
                gnn_8020.append(v8020)
            if v5 and v5.get('status') == 'success':
                gnn_5.append(v5)

    def median_metric(items, key, default=0.0):
        vals = [float(x.get(key, default) or 0.0) for x in items if key in x]
        return float(np.median(vals)) if vals else 0.0

    # RF 80/20
    if rf_8020:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['HAMD R', 'HAMD R2', 'HAMA R', 'HAMA R2', 'AUC']
        vals = [
            median_metric(rf_8020, 'hamd_r'),
            median_metric(rf_8020, 'hamd_r2'),
            median_metric(rf_8020, 'hama_r'),
            median_metric(rf_8020, 'hama_r2'),
            median_metric(rf_8020, 'diagnosis_auc'),
        ]
        plot_metric_bar(ax, labels, vals, 'RF 80/20 Performance', 'Score')
        fig.tight_layout()
        fig.savefig(PERF_DIR / '80_20' / 'rf_metrics.png', dpi=150)
        plt.close(fig)

    # RF 5-fold
    if rf_5:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['HAMD R', 'HAMD R2', 'HAMA R', 'HAMA R2', 'AUC']
        vals = [
            median_metric(rf_5, 'hamd_r'),
            median_metric(rf_5, 'hamd_r2'),
            median_metric(rf_5, 'hama_r'),
            median_metric(rf_5, 'hama_r2'),
            median_metric(rf_5, 'diagnosis_auc'),
        ]
        plot_metric_bar(ax, labels, vals, 'RF 5-fold (site) Performance', 'Score')
        fig.tight_layout()
        fig.savefig(PERF_DIR / '5fold' / 'rf_metrics.png', dpi=150)
        plt.close(fig)

    # GNN 80/20
    if gnn_8020:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['HAMD R', 'HAMD R2', 'HAMA R', 'HAMA R2', 'AUC']
        vals = [
            median_metric(gnn_8020, 'hamd_correlation'),
            median_metric(gnn_8020, 'hamd_r2'),
            median_metric(gnn_8020, 'hama_correlation'),
            median_metric(gnn_8020, 'hama_r2'),
            median_metric(gnn_8020, 'diagnosis_auc'),
        ]
        plot_metric_bar(ax, labels, vals, 'GNN 80/20 Performance', 'Score')
        fig.tight_layout()
        fig.savefig(PERF_DIR / '80_20' / 'gnn_metrics.png', dpi=150)
        plt.close(fig)

    # GNN 5-fold
    if gnn_5:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['HAMD R', 'HAMD R2', 'AUC', 'ACC']
        vals = [
            median_metric(gnn_5, 'hamd_correlation'),
            median_metric(gnn_5, 'hamd_r2'),
            median_metric(gnn_5, 'diagnosis_auc'),
            median_metric(gnn_5, 'diagnosis_accuracy'),
        ]
        plot_metric_bar(ax, labels, vals, 'GNN 5-fold (site) Performance', 'Score')
        fig.tight_layout()
        fig.savefig(PERF_DIR / '5fold' / 'gnn_metrics.png', dpi=150)
        plt.close(fig)

    # Save a raw summary JSON for quick viewing
    (PERF_DIR / 'summary.json').write_text(json.dumps({
        'rf_8020': rf_8020,
        'rf_5fold': rf_5,
        'gnn_8020': gnn_8020,
        'gnn_5fold': gnn_5,
    }, indent=2))

    # Interpretability global summaries (ROI-level)
    try:
        def summarize_roi(base: Path, filename: str) -> tuple[list[str], list[float]]:
            if not base.exists():
                return [], []
            import numpy as np
            exp_dirs = [p for p in base.iterdir() if p.is_dir()]
            if not exp_dirs:
                return [], []
            aggregates = []
            for exp in exp_dirs:
                f = exp / filename
                if not f.exists():
                    continue
                arr = np.load(f, allow_pickle=False)
                if arr.ndim == 2:  # subjects x rois
                    aggregates.append(arr.mean(axis=0))
            if not aggregates:
                return [], []
            mean_roi = np.vstack(aggregates).mean(axis=0)
            # Top 20
            k = min(20, len(mean_roi))
            idx = np.argsort(mean_roi)[::-1][:k]
            labels = [f'ROI_{int(i)}' for i in idx]
            values = [float(mean_roi[i]) for i in idx]
            return labels, values

        INT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        # RF
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'rf' / '80_20' / 'per_patient', 'hamd_roi_importance.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'RF 80/20 - Top ROI (HAMD SHAP)', 'Importance')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'rf_80_20_top_roi.png', dpi=150); plt.close(fig)
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'rf' / '5fold' / 'per_patient', 'hamd_roi_importance.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'RF 5-fold - Top ROI (HAMD SHAP)', 'Importance')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'rf_5fold_top_roi.png', dpi=150); plt.close(fig)
        # GNN (prefer IG; fallback to gradients)
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'gnn' / '80_20' / 'per_patient', 'hamd_roi_integrated_gradients.npy')
        if not labels:
            labels, values = summarize_roi(ROOT / 'interpretability_results' / 'gnn' / '80_20' / 'per_patient', 'hamd_roi_gradients.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'GNN 80/20 - Top ROI (HAMD IG)', 'Attribution')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'gnn_80_20_top_roi.png', dpi=150); plt.close(fig)
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'gnn' / '5fold' / 'per_patient', 'hamd_roi_integrated_gradients.npy')
        if not labels:
            labels, values = summarize_roi(ROOT / 'interpretability_results' / 'gnn' / '5fold' / 'per_patient', 'hamd_roi_gradients.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'GNN 5-fold - Top ROI (HAMD IG)', 'Attribution')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'gnn_5fold_top_roi.png', dpi=150); plt.close(fig)
        # Ensemble
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'ensemble' / '80_20' / 'per_patient', 'hamd_roi_ensemble.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'Ensemble 80/20 - Top ROI (HAMD)', 'Attribution')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'ensemble_80_20_top_roi.png', dpi=150); plt.close(fig)
        labels, values = summarize_roi(ROOT / 'interpretability_results' / 'ensemble' / '5fold' / 'per_patient', 'hamd_roi_ensemble.npy')
        if labels:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_metric_bar(ax, labels, values, 'Ensemble 5-fold - Top ROI (HAMD)', 'Attribution')
            fig.tight_layout(); fig.savefig(INT_SUMMARY_DIR / 'ensemble_5fold_top_roi.png', dpi=150); plt.close(fig)
    except Exception as e:
        print(f"Interpretability global summaries skipped: {e}")
    # Edge percentile analysis for GNN
    try:
        def summarize_by_edge(validation_key: str, metric_key: str):
            buckets = {}
            for r in results:
                if r.get('phase') != 'gnn':
                    continue
                edge = r.get('edge_percentile')
                val = r.get(validation_key, {})
                if not isinstance(val, dict):
                    continue
                metric = val.get(metric_key)
                if metric is None:
                    continue
                buckets.setdefault(edge, []).append(float(metric))
            labels = []
            values = []
            for edge, arr in sorted(buckets.items(), key=lambda x: (x[0] is None, x[0])):
                if not arr:
                    continue
                labels.append('none' if edge is None else str(edge))
                values.append(float(np.median(arr)))
            return labels, values

        # 80/20 by edge (HAMD R2)
        labels, values = summarize_by_edge('validation_80_20', 'hamd_r2')
        if labels and values:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_metric_bar(ax, labels, values, 'GNN 80/20 - HAMD R2 by edge_percentile', 'HAMD R2')
            fig.tight_layout()
            fig.savefig(PERF_DIR / '80_20' / 'gnn_edge_hamd_r2.png', dpi=150)
            plt.close(fig)

        # 5-fold by edge (HAMD R2)
        labels5, values5 = summarize_by_edge('validation_5fold_site_cv', 'hamd_r2')
        if labels5 and values5:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_metric_bar(ax, labels5, values5, 'GNN 5-fold - HAMD R2 by edge_percentile', 'HAMD R2')
            fig.tight_layout()
            fig.savefig(PERF_DIR / '5fold' / 'gnn_edge_hamd_r2.png', dpi=150)
            plt.close(fig)

        # Persist a simple JSON summary
        (PERF_DIR / 'gnn_edges_summary.json').write_text(json.dumps({
            '80_20_hamd_r2_by_edge': dict(zip(labels, values)) if labels and values else {},
            '5fold_hamd_r2_by_edge': dict(zip(labels5, values5)) if labels5 and values5 else {}
        }, indent=2))
    except Exception as e:
        print(f"Edge percentile analysis skipped: {e}")


def try_run_interpretability(results: list):
    try:
        from comprehensive_interpretability_integration import ComprehensiveInterpretabilityFramework
        interp = ComprehensiveInterpretabilityFramework(output_dir=str(INT_DIR))

        # Global interpretability stubs using legacy inputs where needed
        # RF
        rf_8020 = [r for r in results if r.get('phase') == 'rf' and r.get('validation_80_20')]
        rf_5 = [r for r in results if r.get('phase') == 'rf' and r.get('validation_5fold_site_cv')]
        if rf_8020:
            interp.run_global_interpretability(
                validation_type='80_20', model_type='random_forest', feature_data={}, results=rf_8020[0]
            )
        if rf_5:
            interp.run_global_interpretability(
                validation_type='5fold_site_cv', model_type='random_forest', feature_data={}, results=rf_5[0]
            )

        # GNN (global)
        gnn_8020 = [r for r in results if r.get('phase') == 'gnn' and r.get('validation_80_20')]
        gnn_5 = [r for r in results if r.get('phase') == 'gnn' and r.get('validation_5fold_site_cv')]
        if gnn_8020:
            interp.run_global_interpretability(
                validation_type='80_20', model_type='gnn', feature_data={}, results=gnn_8020[0]
            )
        if gnn_5:
            interp.run_global_interpretability(
                validation_type='5fold_site_cv', model_type='gnn', feature_data={}, results=gnn_5[0]
            )

    except Exception as e:
        print(f"Interpretability generation failed or modules missing: {e}")
        # Still proceed; performance plots are saved


def main():
    ensure_dirs()
    print("[1/3] Preflight...")
    code = run_preflight()
    if code != 0:
        print(f"Preflight failed with code {code}")
        return code

    print("[2/3] Running full pipeline...")
    run_full_pipeline()

    print("[3/3] Generating reports...")
    results = load_incremental_results()
    if not results:
        print("No incremental results found; skipping report generation")
        return 0
    save_performance_plots(results)
    try_run_interpretability(results)
    print(f"Reports saved under: {REPORTS}")

    # Global interpretability summaries + model cards
    try:
        # Already plotted top ROI summaries inside save_performance_plots via INT_SUMMARY_DIR
        # Write lightweight model cards for first 200 successful experiments
        cards_dir = ROOT / 'artifacts' / 'model_cards'
        cards_dir.mkdir(parents=True, exist_ok=True)
        def write_card(r: dict):
            eid = r.get('experiment_id', 'unknown')
            phase = r.get('phase', 'unknown')
            path = cards_dir / f"{phase}_{eid}.md"
            lines = []
            lines.append(f"# Model Card: {phase} — {eid}")
            lines.append("")
            lines.append(f"- Phase: {phase}")
            lines.append(f"- Atlas: {r.get('atlas','')}")
            lines.append(f"- Features: {r.get('features','')}")
            lines.append(f"- Feature Type: {r.get('feature_type','')}")
            if phase == 'gnn':
                lines.append(f"- Edge Percentile: {r.get('edge_percentile','')}")
            lines.append("")
            v80 = r.get('validation_80_20', {})
            v5 = r.get('validation_5fold_site_cv', {})
            lines.append("## Metrics")
            if v80:
                lines.append(f"- 80/20: HAMD R2={v80.get('hamd_r2',0):.3f}, AUC={v80.get('diagnosis_auc',0):.3f}")
            if v5:
                lines.append(f"- 5-fold: HAMD R2={v5.get('hamd_r2',0):.3f}, AUC={v5.get('diagnosis_auc',0):.3f}")
            lines.append("")
            lines.append("## Artifacts")
            if phase == 'gnn':
                sm = r.get('saved_model_path')
                pp = r.get('per_patient_dir')
                if sm:
                    lines.append(f"- Model: {sm}")
                if pp:
                    lines.append(f"- GNN per-patient: {pp}")
            lines.append(f"- Reports: {REPORTS}")
            path.write_text("\n".join(lines))
        count = 0
        for r in results:
            if r.get('status') == 'success':
                write_card(r)
                count += 1
                if count >= 200:
                    break
        print(f"Model cards written to: {cards_dir}")
    except Exception as e:
        print(f"Model card generation skipped: {e}")
    
    # Optional: per-patient export for GNN 80/20 using default/best config
    try:
        print("Per-patient export (80/20): training short GNN and exporting predictions...")
        from ultimate_comprehensive_ablation_framework import UltimateComprehensiveAblationFramework, GNNConfig, create_site_stratified_train_test_split
        from sklearn.model_selection import GroupKFold
        fw = UltimateComprehensiveAblationFramework()
        fw.data_loader.load_all_data()
        # Use tuned edge percentile if present in results
        edge_pct = None
        for r in results:
            if r.get('phase') == 'gnn' and 'edge_percentile' in r:
                edge_pct = r['edge_percentile']
                break
        fw.data_loader.edge_percentile = edge_pct
        graphs = fw.data_loader.create_graphs_for_config(
            __import__('types').SimpleNamespace(atlas='cc200', features=['alff'], feature_type='graph', connectivity_type='fc')
        )
        split = create_site_stratified_train_test_split(fw.data_loader.subjects_df, test_size=0.2, random_state=42)
        train_graphs = [graphs[i] for i in split['train_indices']]
        test_graphs = [graphs[i] for i in split['test_indices']]
        cfg = GNNConfig(hidden_dim=128, num_heads=4, num_layers=3)
        import torch
        from torch_geometric.data import Batch
        from sklearn.metrics import r2_score
        model = __import__('ultimate_comprehensive_ablation_framework', fromlist=['HierarchicalGNN']).HierarchicalGNN(cfg, n_sites=len(set(fw.data_loader.subjects_df['site']))).to(fw.device)
        # class weight
        import numpy as np
        diags = np.array([g.diagnosis.item() for g in train_graphs])
        pos = max(1, int(diags.sum())); neg = max(1, len(diags)-pos)
        model.set_diag_pos_weight(torch.tensor(neg/pos, dtype=torch.float32, device=fw.device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        model.train()
        for epoch in range(60):
            for i in range(0, len(train_graphs), 8):
                batch_graphs = train_graphs[i:i+8]
                batch = Batch.from_data_list(batch_graphs).to(fw.device)
                opt.zero_grad()
                # make batch tensor
                nums = [g.residuals.size(0) for g in batch_graphs]
                bt = torch.tensor(sum(([j]*n for j,n in enumerate(nums)), []), dtype=torch.long, device=fw.device)
                out = model(batch.residuals, batch.schaefer_ids, batch.yeo_ids, batch.edge_index, batch.edge_attr, bt, batch.site_id)
                loss = model.compute_total_loss(out, batch.hamd, batch.hama, batch.diagnosis)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # Evaluate and export
        model.eval(); rows=[]
        import csv
        hamd_true=[]; hamd_pred=[]
        with torch.no_grad():
            for g in test_graphs:
                gg = g.to(fw.device)
                bt = torch.zeros(gg.residuals.shape[0], dtype=torch.long, device=fw.device)
                o = model(gg.residuals, gg.schaefer_ids, gg.yeo_ids, gg.edge_index, gg.edge_attr, bt, gg.site_id.unsqueeze(0))
                pred = float(o['hamd_mean'].cpu().item())
                diag_p = float(torch.sigmoid(o['diag_logit']).cpu().item())
                rows.append({'subject_id': getattr(g, 'subject_id', ''), 'site': g.site_name, 'hamd_true': float(g.hamd.cpu().item()), 'hamd_pred': pred, 'abs_error': abs(pred-float(g.hamd.cpu().item())), 'diag_true': int(g.diagnosis.cpu().item()), 'diag_prob': diag_p})
                hamd_true.append(float(g.hamd.cpu().item())); hamd_pred.append(pred)
        # Save CSV
        out_csv = PP_DIR / '80_20' / 'per_patient_predictions.csv'
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        # Scatter plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5));
        ax.scatter(hamd_true, hamd_pred, alpha=0.5); lim=[min(hamd_true+hamd_pred), max(hamd_true+hamd_pred)]; ax.plot(lim, lim, 'k--', lw=1);
        ax.set_title('GNN 80/20 HAMD True vs Pred'); ax.set_xlabel('True'); ax.set_ylabel('Pred'); fig.tight_layout(); fig.savefig(PP_DIR / '80_20' / 'hamd_true_vs_pred.png', dpi=150); plt.close(fig)
        # Top-K exports
        rows_sorted = sorted(rows, key=lambda r: r['abs_error'])
        import json
        (PP_DIR / '80_20' / 'topk_well_fit.json').write_text(json.dumps(rows_sorted[:20], indent=2))
        (PP_DIR / '80_20' / 'topk_high_error.json').write_text(json.dumps(rows_sorted[-20:], indent=2))
        print(f"Per-patient 80/20 CSV saved: {out_csv}")

        # Full per-patient export for 5-fold site CV
        print("Per-patient export (5-fold by site): training short GNN per fold and exporting predictions...")
        sites = list(fw.data_loader.subjects_df['site'])
        gkf = GroupKFold(n_splits=5)
        all_graphs = graphs  # reuse built graphs with tuned edge percentile
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(all_graphs, groups=sites)):
            fold_dir = PP_DIR / '5fold' / f'fold_{fold_idx+1}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_graphs = [all_graphs[i] for i in train_idx]
            val_graphs = [all_graphs[i] for i in val_idx]
            cfg = GNNConfig(hidden_dim=128, num_heads=4, num_layers=3)
            import torch
            from torch_geometric.data import Batch
            from sklearn.metrics import r2_score
            model = __import__('ultimate_comprehensive_ablation_framework', fromlist=['HierarchicalGNN']).HierarchicalGNN(cfg, n_sites=len(set(fw.data_loader.subjects_df['site']))).to(fw.device)
            # class weight per fold
            import numpy as np
            diags = np.array([g.diagnosis.item() for g in train_graphs])
            pos = max(1, int(diags.sum())); neg = max(1, len(diags)-pos)
            model.set_diag_pos_weight(torch.tensor(neg/pos, dtype=torch.float32, device=fw.device))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            model.train()
            for epoch in range(60):
                for i in range(0, len(train_graphs), 8):
                    batch_graphs = train_graphs[i:i+8]
                    batch = Batch.from_data_list(batch_graphs).to(fw.device)
                    opt.zero_grad()
                    nums = [g.residuals.size(0) for g in batch_graphs]
                    bt = torch.tensor(sum(([j]*n for j,n in enumerate(nums)), []), dtype=torch.long, device=fw.device)
                    out = model(batch.residuals, batch.schaefer_ids, batch.yeo_ids, batch.edge_index, batch.edge_attr, bt, batch.site_id)
                    loss = model.compute_total_loss(out, batch.hamd, batch.hama, batch.diagnosis)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            # Evaluate fold
            model.eval(); rows=[]
            import csv
            hamd_true=[]; hamd_pred=[]
            with torch.no_grad():
                for g in val_graphs:
                    gg = g.to(fw.device)
                    bt = torch.zeros(gg.residuals.shape[0], dtype=torch.long, device=fw.device)
                    o = model(gg.residuals, gg.schaefer_ids, gg.yeo_ids, gg.edge_index, gg.edge_attr, bt, gg.site_id.unsqueeze(0))
                    pred = float(o['hamd_mean'].cpu().item())
                    diag_p = float(torch.sigmoid(o['diag_logit']).cpu().item())
                    rows.append({'subject_id': getattr(g, 'subject_id', ''), 'site': g.site_name, 'hamd_true': float(g.hamd.cpu().item()), 'hamd_pred': pred, 'abs_error': abs(pred-float(g.hamd.cpu().item())), 'diag_true': int(g.diagnosis.cpu().item()), 'diag_prob': diag_p})
                    hamd_true.append(float(g.hamd.cpu().item())); hamd_pred.append(pred)
            # Save per-fold CSV and scatter
            out_csv_fold = fold_dir / 'per_patient_predictions.csv'
            with out_csv_fold.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            fig, ax = plt.subplots(figsize=(5,5));
            if hamd_true and hamd_pred:
                ax.scatter(hamd_true, hamd_pred, alpha=0.5); lim=[min(hamd_true+hamd_pred), max(hamd_true+hamd_pred)]; ax.plot(lim, lim, 'k--', lw=1)
            ax.set_title(f'GNN 5-fold Fold {fold_idx+1} HAMD True vs Pred'); ax.set_xlabel('True'); ax.set_ylabel('Pred'); fig.tight_layout(); fig.savefig(fold_dir / 'hamd_true_vs_pred.png', dpi=150); plt.close(fig)
        print("Per-patient 5-fold exports saved under reports/per_patient/5fold/")
    except Exception as e:
        print(f"Per-patient export skipped: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
