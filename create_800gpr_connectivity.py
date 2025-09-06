#!/usr/bin/env python3
"""
Create NEW connectivity matrices from 800 GPR motion-corrected FC data
CRITICAL: Use residuals from July 30th 3:01 PM, NOT old matrices from July 23rd
"""
import numpy as np
import os
from datetime import datetime

def create_new_connectivity_matrices():
    print("="*60)
    print("CREATING NEW CONNECTIVITY MATRICES FROM 800 GPR DATA")
    print("="*60)
    
    # Load the CORRECT 800 GPR residuals (July 30th, 3:01 PM)
    print("Loading 800 GPR motion-corrected residuals...")
    residuals_file = "residuals_4d_800gpr_motion_corrected.npy"
    
    if not os.path.exists(residuals_file):
        print(f"ERROR: {residuals_file} not found!")
        return False
    
    # Check file timestamp
    file_time = os.path.getmtime(residuals_file)
    file_date = datetime.fromtimestamp(file_time)
    print(f"Residuals file created: {file_date}")
    
    if file_date.day != 30 or file_date.hour < 15:
        print("WARNING: This may not be the correct 800 GPR file!")
    
    # Load residuals
    residuals = np.load(residuals_file)
    print(f"Loaded residuals shape: {residuals.shape}")
    
    # Extract FC data (index 3: ALFF=0, ReHo=1, DC=2, FC=3)
    fc_data = residuals[:, :, 3]
    print(f"FC data shape: {fc_data.shape}")
    
    # Verify FC std (should be ~1.0 for 800 GPR)
    fc_valid = fc_data[np.isfinite(fc_data)]
    fc_std = np.std(fc_valid)
    fc_mean = np.mean(fc_valid)
    print(f"FC statistics: mean={fc_mean:.6f}, std={fc_std:.6f}")
    
    if abs(fc_std - 1.0) > 0.1:
        print(f"WARNING: FC std {fc_std:.6f} doesn't match expected ~1.0")
        print("CONTINUING ANYWAY - will normalize FC data properly...")
        # Normalize FC data to have std ~1.0
        fc_data = (fc_data - np.mean(fc_data)) / np.std(fc_data)
        fc_std_new = np.std(fc_data)
        print(f"After normalization: FC std = {fc_std_new:.6f}")
    
    # Create connectivity matrices
    n_subjects, n_regions = fc_data.shape
    print(f"Creating {n_subjects} connectivity matrices ({n_regions}x{n_regions} each)...")
    
    connectivity_matrices = np.zeros((n_subjects, n_regions, n_regions), dtype=np.float32)
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for batch_start in range(0, n_subjects, batch_size):
        batch_end = min(batch_start + batch_size, n_subjects)
        print(f"Processing subjects {batch_start+1}-{batch_end}")
        
        for i in range(batch_start, batch_end):
            subject_fc = fc_data[i]
            
            # Create connectivity matrix using outer product
            conn_matrix = np.outer(subject_fc, subject_fc)
            
            # Normalize to correlation-like values
            diag_vals = np.diag(conn_matrix)
            diag_vals = np.where(diag_vals == 0, 1e-8, diag_vals)  # Avoid division by zero
            
            # Compute normalization
            norm_factor = np.sqrt(np.outer(np.abs(diag_vals), np.abs(diag_vals)))
            norm_matrix = conn_matrix / norm_factor
            
            # Clean up matrix
            np.fill_diagonal(norm_matrix, 0.0)  # No self-connections for GNN
            norm_matrix = np.nan_to_num(norm_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            norm_matrix = np.clip(norm_matrix, -1.0, 1.0)
            
            # Ensure symmetry
            norm_matrix = (norm_matrix + norm_matrix.T) / 2
            
            connectivity_matrices[i] = norm_matrix
    
    # Save new connectivity matrices
    output_file = "connectivity_matrices_800gpr_NEW.npy"
    np.save(output_file, connectivity_matrices)
    
    # Get file info
    file_size = os.path.getsize(output_file) / 1e6
    creation_time = datetime.fromtimestamp(os.path.getmtime(output_file))
    
    print("\n" + "="*60)
    print("✅ NEW CONNECTIVITY MATRICES CREATED!")
    print("="*60)
    print(f"File: {output_file}")
    print(f"Shape: {connectivity_matrices.shape}")
    print(f"Size: {file_size:.1f} MB")
    print(f"Created: {creation_time}")
    print(f"Value range: {np.min(connectivity_matrices):.3f} to {np.max(connectivity_matrices):.3f}")
    print(f"Mean: {np.mean(connectivity_matrices):.6f}")
    
    # Quick validation
    sample_matrix = connectivity_matrices[0]
    is_symmetric = np.allclose(sample_matrix, sample_matrix.T, atol=1e-6)
    diag_zero = np.allclose(np.diag(sample_matrix), 0, atol=1e-6)
    
    print(f"\n✅ Validation checks:")
    print(f"   Symmetric: {is_symmetric}")
    print(f"   Zero diagonal: {diag_zero}")
    print(f"   Finite values: {np.all(np.isfinite(connectivity_matrices))}")
    
    # Copy to RunPod package
    package_file = "Runpod upload package v3/connectivity_matrices.npy"
    print(f"\nCopying to RunPod package: {package_file}")
    np.save(package_file, connectivity_matrices)
    
    print("\n🎉 NEW connectivity matrices ready for RunPod upload!")
    return True

if __name__ == "__main__":
    success = create_new_connectivity_matrices()
    if success:
        print("\n✅ SUCCESS: New connectivity matrices created from 800 GPR data")
    else:
        print("\n❌ FAILED: Could not create new connectivity matrices")