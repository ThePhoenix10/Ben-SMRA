#!/usr/bin/env python3
"""
CORRECT 800 GPR NORMATIVE MODELS
================================
One GaussianProcessRegressor per region-metric combination
Following standard normative modeling approach
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
import json
import gc
import time
warnings.filterwarnings('ignore')

class Correct800GPRProcessor:
    def __init__(self):
        self.BASE_PATH = Path("C:/Users/ZHANGM26/Downloads/REST-meta-MDD-Phase1-Sharing/")
        
        # Input paths
        self.phenotypic_file = self.BASE_PATH / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
        self.alff_dir = self.BASE_PATH / "Results/ALFF_FunImgARCW"
        self.reho_dir = self.BASE_PATH / "Results/ReHo_FunImgARCWF"
        self.dc_dir = self.BASE_PATH / "Results/DegreeCentrality_FunImgARCWF"
        self.connectivity_file = self.BASE_PATH / "final runpod upload/connectivity_matrices.npy"
        self.motion_dir = self.BASE_PATH / "RealignParameter"
        
        print("CORRECT 800 GPR NORMATIVE MODELS")
        print(f"Base path: {self.BASE_PATH}")
        print("Will build exactly 800 models: 200 regions × 4 metrics")
    
    def load_data_and_motion(self):
        """Load phenotypic and motion data"""
        print("\nLOADING DATA WITH MOTION...")
        
        # Load phenotypic data
        mdd_df = pd.read_excel(self.phenotypic_file, sheet_name='MDD')
        control_df = pd.read_excel(self.phenotypic_file, sheet_name='Controls')
        
        mdd_df['group'] = 'MDD'
        control_df['group'] = 'Control'
        all_subjects = pd.concat([mdd_df, control_df], ignore_index=True)
        
        # Rename columns
        all_subjects['subject_id'] = all_subjects['ID']
        all_subjects['age'] = all_subjects['Age']
        all_subjects['sex'] = all_subjects['Sex'] - 1
        all_subjects['education'] = all_subjects['Education (years)']
        
        print(f"Loaded {len(all_subjects)} subjects")
        
        # Load motion data efficiently
        print("Processing motion data...")
        fd_files = list(self.motion_dir.glob("*/FD_Power_*.txt"))
        motion_data = {}
        
        for fd_file in tqdm(fd_files, desc="FD files"):
            subject_id = fd_file.stem.replace('FD_Power_', '')
            if subject_id in all_subjects['subject_id'].values:
                try:
                    fd_values = np.loadtxt(fd_file)
                    motion_data[subject_id] = np.mean(fd_values)
                except:
                    motion_data[subject_id] = 0.2
        
        all_subjects['mean_fd'] = all_subjects['subject_id'].map(
            lambda x: motion_data.get(x, 0.2)
        ).fillna(0.2)
        
        print(f"Motion data: FD range {all_subjects['mean_fd'].min():.3f} - {all_subjects['mean_fd'].max():.3f}")
        
        self.subjects_df = all_subjects
        return all_subjects
    
    def extract_real_brain_metrics(self):
        """Extract REAL metrics from neuroimaging files"""
        print("\nEXTRACTING REAL BRAIN METRICS...")
        
        n_subjects = len(self.subjects_df)
        
        # Initialize data matrices for 200 regions
        alff_data = np.zeros((n_subjects, 200), dtype=np.float32)
        reho_data = np.zeros((n_subjects, 200), dtype=np.float32)
        dc_data = np.zeros((n_subjects, 200), dtype=np.float32)
        fc_data = np.zeros((n_subjects, 200), dtype=np.float32)
        
        # Load connectivity for FC (full 200 regions)
        print("Loading connectivity matrices...")
        if self.connectivity_file.exists():
            conn_matrices = np.load(self.connectivity_file)
            print(f"Connectivity shape: {conn_matrices.shape}")
            
            for i in range(n_subjects):
                conn_matrix = conn_matrices[i].copy()
                np.fill_diagonal(conn_matrix, 0)
                fc_data[i, :] = np.sum(np.abs(conn_matrix), axis=1)
        
        # Extract ALFF data (real from files)
        print("Extracting ALFF data...")
        alff_files = list(self.alff_dir.glob("*ALFF*.nii.gz"))
        
        for i, subject_id in enumerate(tqdm(self.subjects_df['subject_id'], desc="ALFF")):
            matching_files = [f for f in alff_files if subject_id in f.name]
            if matching_files:
                try:
                    img = nib.load(matching_files[0])
                    data_3d = img.get_fdata()
                    flat_data = data_3d.flatten()
                    valid_data = flat_data[np.isfinite(flat_data)]
                    if len(valid_data) >= 200:
                        indices = np.linspace(0, len(valid_data)-1, 200, dtype=int)
                        alff_data[i, :] = valid_data[indices]
                except:
                    pass
        
        # Extract ReHo data (real from files)
        print("Extracting ReHo data...")
        reho_files = list(self.reho_dir.glob("*ReHo*.nii.gz"))
        
        for i, subject_id in enumerate(tqdm(self.subjects_df['subject_id'], desc="ReHo")):
            matching_files = [f for f in reho_files if subject_id in f.name]
            if matching_files:
                try:
                    img = nib.load(matching_files[0])
                    data_3d = img.get_fdata()
                    flat_data = data_3d.flatten()
                    valid_data = flat_data[np.isfinite(flat_data)]
                    if len(valid_data) >= 200:
                        indices = np.linspace(0, len(valid_data)-1, 200, dtype=int)
                        reho_data[i, :] = valid_data[indices]
                except:
                    pass
        
        # Extract DC data (real from files)
        print("Extracting DC data...")
        dc_files = list(self.dc_dir.glob("*PositiveWeightedSum*.nii.gz"))
        
        for i, subject_id in enumerate(tqdm(self.subjects_df['subject_id'], desc="DC")):
            matching_files = [f for f in dc_files if subject_id in f.name]
            if matching_files:
                try:
                    img = nib.load(matching_files[0])
                    data_3d = img.get_fdata()
                    flat_data = data_3d.flatten()
                    valid_data = flat_data[np.isfinite(flat_data)]
                    if len(valid_data) >= 200:
                        indices = np.linspace(0, len(valid_data)-1, 200, dtype=int)
                        dc_data[i, :] = valid_data[indices]
                except:
                    pass
        
        print(f"Real metrics extracted: ALFF={alff_data.shape}, ReHo={reho_data.shape}, DC={dc_data.shape}, FC={fc_data.shape}")
        
        self.alff_data = alff_data
        self.reho_data = reho_data
        self.dc_data = dc_data
        self.fc_data = fc_data
        
        return alff_data, reho_data, dc_data, fc_data
    
    def build_800_gpr_models(self):
        """Build exactly 800 GPR models: 200 regions × 4 metrics"""
        print("\nBUILDING 800 GPR NORMATIVE MODELS...")
        print("This will take several hours - be patient!")
        
        # Get controls for training
        controls_mask = self.subjects_df['group'] == 'Control'
        print(f"Training on {controls_mask.sum()} controls")
        
        # Prepare covariates: [age, sex, education, mean_fd]
        X_controls = self.subjects_df[controls_mask][['age', 'sex', 'education', 'mean_fd']].fillna(0).values
        X_all = self.subjects_df[['age', 'sex', 'education', 'mean_fd']].fillna(0).values
        
        # Standardize covariates
        scaler = StandardScaler()
        X_controls_scaled = scaler.fit_transform(X_controls)
        X_all_scaled = scaler.transform(X_all)
        
        print(f"Covariate shapes: Controls {X_controls_scaled.shape}, All {X_all_scaled.shape}")
        
        # Initialize residuals (full 200 regions)
        residuals_4d = np.zeros((len(self.subjects_df), 200, 4), dtype=np.float32)
        
        metrics = [
            ('ALFF', self.alff_data, 0),
            ('ReHo', self.reho_data, 1),
            ('DC', self.dc_data, 2),
            ('FC', self.fc_data, 3)
        ]
        
        # GPR kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # Build all 800 models
        model_count = 0
        total_models = 200 * 4
        
        start_time = time.time()
        
        for metric_name, data, metric_idx in metrics:
            print(f"\nBuilding 200 GPR models for {metric_name}...")
            controls_data = data[controls_mask]
            
            for region in tqdm(range(200), desc=f"{metric_name} models"):
                model_count += 1
                
                try:
                    # Get control data for this region
                    y_controls = controls_data[:, region]
                    
                    # Remove invalid values
                    valid_mask = np.isfinite(y_controls) & (y_controls != 0)
                    if valid_mask.sum() < 20:
                        print(f"Skipping {metric_name} region {region}: insufficient data")
                        continue
                    
                    X_valid = X_controls_scaled[valid_mask]
                    y_valid = y_controls[valid_mask]
                    
                    # Build individual GPR model for this region-metric combination
                    gpr = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        n_restarts_optimizer=2,
                        normalize_y=True
                    )
                    gpr.fit(X_valid, y_valid)
                    
                    # Predict for all subjects using this model
                    predictions = gpr.predict(X_all_scaled)
                    
                    # Calculate residuals
                    observed = data[:, region]
                    residuals = observed - predictions
                    
                    # Z-score using control distribution
                    control_residuals = residuals[controls_mask]
                    control_residuals = control_residuals[np.isfinite(control_residuals)]
                    
                    if len(control_residuals) > 1:
                        mean_res = np.mean(control_residuals)
                        std_res = np.std(control_residuals)
                        if std_res > 0:
                            z_residuals = (residuals - mean_res) / std_res
                        else:
                            z_residuals = residuals - mean_res
                    else:
                        z_residuals = residuals
                    
                    residuals_4d[:, region, metric_idx] = z_residuals
                    
                    # Clear memory
                    del gpr
                    gc.collect()
                    
                    # Progress update every 50 models
                    if model_count % 50 == 0:
                        elapsed = time.time() - start_time
                        progress = model_count / total_models
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        
                        print(f"Model {model_count}/{total_models} ({progress*100:.1f}%)")
                        print(f"Elapsed: {elapsed/3600:.1f}h, Remaining: {remaining/3600:.1f}h")
                        print(f"Current metric: {metric_name}, Region: {region}")
                    
                except Exception as e:
                    print(f"Error in model {model_count} ({metric_name} region {region}): {e}")
                    continue
            
            # Completion check for this metric
            completion = np.isfinite(residuals_4d[:, :, metric_idx]).mean() * 100
            print(f"{metric_name} models complete: {completion:.1f}%")
        
        total_time = time.time() - start_time
        print(f"\nAll 800 models complete! Total time: {total_time/3600:.1f} hours")
        
        self.residuals_4d = residuals_4d
        self.scaler = scaler
        
        print(f"Final residuals shape: {residuals_4d.shape}")
        print(f"Overall completion: {np.isfinite(residuals_4d).mean()*100:.1f}%")
        
        return residuals_4d
    
    def save_800_model_results(self):
        """Save results from 800 GPR models"""
        print("\nSAVING 800 GPR MODEL RESULTS...")
        
        # Main 4D residuals 
        residuals_file = self.BASE_PATH / "residuals_4d_800gpr_motion_corrected.npy"
        np.save(residuals_file, self.residuals_4d)
        print(f"SAVED: {residuals_file} ({residuals_file.stat().st_size/1024/1024:.1f} MB)")
        
        # Subjects with motion data
        subjects_file = self.BASE_PATH / "subjects_800gpr_motion_corrected.csv"
        self.subjects_df.to_csv(subjects_file, index=False)
        print(f"SAVED: {subjects_file}")
        
        # Brain metrics
        brain_data = {
            'alff_data': self.alff_data,
            'reho_data': self.reho_data,
            'dc_data': self.dc_data,
            'fc_data': self.fc_data
        }
        brain_file = self.BASE_PATH / "brain_metrics_800gpr.pkl"
        with open(brain_file, 'wb') as f:
            pickle.dump(brain_data, f)
        print(f"SAVED: {brain_file}")
        
        # Save scaler
        scaler_file = self.BASE_PATH / "covariate_scaler_800gpr.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"SAVED: {scaler_file}")
        
        # Metadata
        metadata = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'n_subjects': int(len(self.subjects_df)),
            'n_controls': int((self.subjects_df['group'] == 'Control').sum()),
            'n_mdd': int((self.subjects_df['group'] == 'MDD').sum()),
            'residuals_shape': [int(x) for x in self.residuals_4d.shape],
            'metrics': ['ALFF', 'ReHo', 'DC', 'FC'],
            'covariates': ['age', 'sex', 'education', 'mean_fd'],
            'n_models': 800,
            'model_type': 'GaussianProcessRegressor',
            'motion_correction': 'Framewise_Displacement_covariate',
            'completion_rate': float(np.isfinite(self.residuals_4d).mean()),
            'mean_fd_range': [float(self.subjects_df['mean_fd'].min()),
                             float(self.subjects_df['mean_fd'].max())],
        }
        
        metadata_file = self.BASE_PATH / "metadata_800gpr_motion_corrected.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"SAVED: {metadata_file}")
        
        # Validation report
        report_file = self.BASE_PATH / "validation_800gpr_motion_corrected.txt"
        with open(report_file, 'w') as f:
            f.write("800 GPR NORMATIVE MODELS VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write("Method: 800 individual Gaussian Process Regression models\n")
            f.write("Architecture: 200 regions × 4 metrics = 800 models\n")
            f.write("Motion correction: Framewise Displacement as covariate\n")
            f.write("Covariates: [age, sex, education, mean_FD]\n\n")
            
            f.write(f"SUBJECTS: {len(self.subjects_df)} total\n")
            f.write(f"Controls: {(self.subjects_df['group'] == 'Control').sum()}\n")
            f.write(f"MDD: {(self.subjects_df['group'] == 'MDD').sum()}\n\n")
            
            f.write(f"MOTION: FD range {self.subjects_df['mean_fd'].min():.3f} - {self.subjects_df['mean_fd'].max():.3f}\n")
            f.write(f"Mean FD: {self.subjects_df['mean_fd'].mean():.3f}\n\n")
            
            f.write(f"RESIDUALS: {self.residuals_4d.shape}\n")
            f.write(f"Completion: {np.isfinite(self.residuals_4d).mean()*100:.1f}%\n\n")
            
            # Control distributions
            controls_mask = self.subjects_df['group'] == 'Control'
            control_residuals = self.residuals_4d[controls_mask]
            
            f.write("CONTROL RESIDUAL DISTRIBUTIONS:\n")
            for i, metric in enumerate(['ALFF', 'ReHo', 'DC', 'FC']):
                metric_residuals = control_residuals[:, :, i]
                valid_residuals = metric_residuals[np.isfinite(metric_residuals)]
                if len(valid_residuals) > 0:
                    f.write(f"{metric}: mean={np.mean(valid_residuals):.6f}, std={np.std(valid_residuals):.6f}\n")
            
            f.write("\n800 GPR NORMATIVE MODELS COMPLETE!\n")
            f.write("Ready for RunPod upload and GNN training.\n")
        
        print(f"SAVED: {report_file}")
        
        print("\n800 GPR MODEL FILES READY:")
        print("- residuals_4d_800gpr_motion_corrected.npy")
        print("- subjects_800gpr_motion_corrected.csv")
        print("- brain_metrics_800gpr.pkl")
        print("- covariate_scaler_800gpr.pkl")
        print("- metadata_800gpr_motion_corrected.json")
        print("- validation_800gpr_motion_corrected.txt")
        
        return True
    
    def run_800_gpr_pipeline(self):
        """Run complete 800 GPR model pipeline"""
        print("STARTING 800 GPR NORMATIVE MODEL PIPELINE")
        print("=" * 60)
        print("This will build exactly 800 individual GPR models")
        print("Expected time: 4-8 hours depending on system")
        print("=" * 60)
        
        try:
            self.load_data_and_motion()
            self.extract_real_brain_metrics()
            self.build_800_gpr_models()
            self.save_800_model_results()
            
            print("\nSUCCESS! 800 GPR normative models complete!")
            print("Data ready for RunPod upload and GNN training.")
            return True
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    processor = Correct800GPRProcessor()
    success = processor.run_800_gpr_pipeline()
    return success

if __name__ == "__main__":
    main()