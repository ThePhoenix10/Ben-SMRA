#!/usr/bin/env python3
"""
RunPod Package 9 - Complete Pipeline Runner
===========================================

This script runs the complete 54,000 experiment pipeline with:
- GPU acceleration (2×H100)
- Incremental saving
- Resume functionality
- All visualizations and interpretability

Usage:
    python run_full_pipeline.py
"""

import sys
import time
from ultimate_comprehensive_ablation_framework import UltimateComprehensiveAblationFramework

def main():
    print("🚀 STARTING ULTIMATE COMPREHENSIVE ABLATION FRAMEWORK")
    print("=" * 80)
    print("📋 Features:")
    print("  - 54,000 total experiments across 45 atlas configurations")
    print("  - GPU acceleration with 2×H100 support")  
    print("  - Incremental saving (no data loss)")
    print("  - Resume functionality")
    print("  - Full interpretability and visualization")
    print("=" * 80)
    
    # Initialize framework
    framework = UltimateComprehensiveAblationFramework()
    
    # Load data
    print("📊 Loading comprehensive dataset...")
    framework.data_loader.load_all_data()
    
    # FORCE RESTART - clear broken results
    print("🔄 Clearing corrupted results and restarting fresh...")
    import os
    if os.path.exists('incremental_results/incremental_results.json'):
        os.rename('incremental_results/incremental_results.json', 'incremental_results/corrupted_backup.json')
        print("   📁 Moved corrupted results to backup")
    if os.path.exists('incremental_results/experiment_progress.json'):
        os.remove('incremental_results/experiment_progress.json')
        print("   🗑️ Cleared progress file")
    
    print("🆕 Starting completely fresh run")
    
    # Set start time
    framework.start_time = time.time()
    
    try:
        # Run the complete pipeline
        print("🎬 Starting complete ablation pipeline...")
        results = framework.run_ultimate_comprehensive_ablation()
        
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Total experiments: {len(results)}")
        print(f"📂 Results saved in: {framework.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        print(f"💾 Results saved up to interruption in: {framework.results_dir}")
        print("🔄 Run again to resume from where you left off")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print(f"💾 Partial results may be saved in: {framework.results_dir}")
        raise

if __name__ == "__main__":
    main()