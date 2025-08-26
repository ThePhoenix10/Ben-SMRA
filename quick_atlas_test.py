#!/usr/bin/env python3
"""
Quick Atlas Test - Just check data loading works
"""

from ultimate_comprehensive_ablation_framework import *

def quick_test():
    print("QUICK ATLAS DATA TEST")
    print("=" * 40)
    
    # Initialize framework
    framework = UltimateComprehensiveAblationFramework()
    
    # Test data loading
    framework.data_loader.load_all_data()
    print(f"SUCCESS: {len(framework.data_loader.subjects_df)} subjects loaded")
    print(f"MDD: {(framework.data_loader.subjects_df['group'] == 'MDD').sum()}")
    print(f"Controls: {(framework.data_loader.subjects_df['group'] == 'Control').sum()}")
    
    # Test one atlas
    config = ExperimentConfig('test', 'rf', 'aal_116', ['alff'], 'node_only')
    X, y_hamd, y_hama, y_diag = framework.data_loader.extract_features_for_config(config)
    print(f"AAL-116 Features: {X.shape}")
    print(f"Feature variance check: {X.var(axis=0).mean():.6f}")
    print(f"Real data confirmed!" if X.var(axis=0).mean() > 0.01 else f"Still synthetic data")

if __name__ == "__main__":
    quick_test()