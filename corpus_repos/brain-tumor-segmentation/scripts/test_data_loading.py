"""
Debug script to find problematic patient data
"""
from pathlib import Path
from create_dataset import BraTSSegmentationDataset
from tqdm import tqdm
import torch

def test_all_patients():
    """Test loading all patients to find corrupted data"""
    
    data_dir = Path.home() / "brain_tumor_thesis/segmentation_project/data/raw/brats_gli/training_data1_v2"
    patient_dirs = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("BraTS-GLI-")])
    
    print(f"Testing {len(patient_dirs)} patients...")
    
    failed_patients = []
    
    for patient_dir in tqdm(patient_dirs, desc="Testing patients"):
        try:
            dataset = BraTSSegmentationDataset([patient_dir])
            volume, seg = dataset[0]
            
            # Check for issues
            if torch.isnan(volume).any():
                failed_patients.append((patient_dir.name, "NaN in volume"))
            elif torch.isinf(volume).any():
                failed_patients.append((patient_dir.name, "Inf in volume"))
            elif seg.max() > 3:
                failed_patients.append((patient_dir.name, f"Invalid seg label: {seg.max()}"))
                
        except Exception as e:
            failed_patients.append((patient_dir.name, str(e)))
    
    if failed_patients:
        print(f"\n❌ Found {len(failed_patients)} problematic patients:")
        for patient, error in failed_patients[:20]:
            print(f"  - {patient}: {error}")
        
        # Save to file
        with open("failed_patients.txt", 'w') as f:
            for patient, error in failed_patients:
                f.write(f"{patient}\t{error}\n")
        print(f"\n💾 Full list saved to failed_patients.txt")
    else:
        print("\n✅ All patients loaded successfully!")
    
    return failed_patients

if __name__ == "__main__":
    failed = test_all_patients()
    print(f"\nTotal failed: {len(failed)}")
