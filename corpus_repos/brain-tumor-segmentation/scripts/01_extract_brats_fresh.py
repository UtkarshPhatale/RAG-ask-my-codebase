# save as: segmentation_project/scripts/01_extract_brats_fresh.py
import zipfile
import tarfile
import os
from pathlib import Path
from tqdm import tqdm
import shutil

class BraTSExtractor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path.home() / "brain_tumor_thesis"
        
        self.base_dir = Path(base_dir)
        self.brats_raw = self.base_dir / "data/real_datasets/brats_raw/BraTS-GLI"
        self.seg_project = self.base_dir / "segmentation_project"
        self.output_dir = self.seg_project / "data/raw/brats_gli"
        
    def extract_training_data(self):
        """Extract the main training dataset"""
        
        print("🔍 Looking for BraTS-GLI training data...")
        
        # Find the training data zip file
        training_zip = self.brats_raw / "BraTS2024-BraTS-GLI-TrainingData.zip"
        
        if not training_zip.exists():
            print(f"❌ Training data not found at: {training_zip}")
            print("Please ensure the file exists at this location")
            return False
        
        print(f"✅ Found training data: {training_zip}")
        print(f"📦 Extracting to: {self.output_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract with progress bar
        with zipfile.ZipFile(training_zip, 'r') as zip_ref:
            members = zip_ref.namelist()
            print(f"📊 Total files to extract: {len(members)}")
            
            for member in tqdm(members, desc="Extracting"):
                try:
                    zip_ref.extract(member, self.output_dir)
                except Exception as e:
                    print(f"⚠️ Error extracting {member}: {e}")
        
        print("✅ Extraction complete!")
        return True
    
    def analyze_extracted_data(self):
        """Analyze the extracted dataset structure"""
        
        print("\n📊 Analyzing extracted data structure...")
        
        # Find the actual data directory (might be nested)
        data_dirs = list(self.output_dir.rglob("BraTS-GLI-*"))
        
        if not data_dirs:
            print("❌ No patient directories found!")
            return
        
        # Get patient directories
        patient_dirs = [d for d in data_dirs if d.is_dir() and d.name.startswith("BraTS-GLI-")]
        
        print(f"✅ Found {len(patient_dirs)} patient cases")
        
        # Analyze first patient structure
        if patient_dirs:
            first_patient = patient_dirs[0]
            print(f"\n📁 Sample patient directory: {first_patient.name}")
            print("Files per patient:")
            
            for file in sorted(first_patient.iterdir()):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        # Save patient list
        patient_list_file = self.seg_project / "data/patient_list.txt"
        with open(patient_list_file, 'w') as f:
            for patient_dir in sorted(patient_dirs):
                f.write(f"{patient_dir.name}\n")
        
        print(f"\n✅ Patient list saved to: {patient_list_file}")
        
        return patient_dirs
    
    def verify_data_completeness(self, patient_dirs):
        """Verify each patient has all required sequences"""
        
        print("\n🔍 Verifying data completeness...")
        
        required_sequences = ['t1n', 't1c', 't2w', 't2f', 'seg']
        incomplete_patients = []
        
        for patient_dir in tqdm(patient_dirs, desc="Checking patients"):
            files = [f.name for f in patient_dir.iterdir()]
            
            # Check if all sequences present
            has_all = all(any(seq in f for f in files) for seq in required_sequences)
            
            if not has_all:
                incomplete_patients.append(patient_dir.name)
        
        if incomplete_patients:
            print(f"⚠️ Found {len(incomplete_patients)} incomplete patients:")
            for patient in incomplete_patients[:10]:  # Show first 10
                print(f"  - {patient}")
            if len(incomplete_patients) > 10:
                print(f"  ... and {len(incomplete_patients) - 10} more")
        else:
            print(f"✅ All {len(patient_dirs)} patients have complete data!")
        
        # Save verification report
        report_file = self.seg_project / "data/data_verification_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"BraTS-GLI Data Verification Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total patients: {len(patient_dirs)}\n")
            f.write(f"Complete patients: {len(patient_dirs) - len(incomplete_patients)}\n")
            f.write(f"Incomplete patients: {len(incomplete_patients)}\n\n")
            
            if incomplete_patients:
                f.write("Incomplete patient list:\n")
                for patient in incomplete_patients:
                    f.write(f"  - {patient}\n")
        
        print(f"📄 Verification report saved to: {report_file}")
        
        return len(patient_dirs) - len(incomplete_patients)

def main():
    extractor = BraTSExtractor()
    
    # Extract data
    if extractor.extract_training_data():
        # Analyze structure
        patient_dirs = extractor.analyze_extracted_data()
        
        if patient_dirs:
            # Verify completeness
            complete_count = extractor.verify_data_completeness(patient_dirs)
            
            print(f"\n" + "="*60)
            print(f"🎉 EXTRACTION COMPLETE")
            print(f"="*60)
            print(f"✅ {complete_count} patients ready for segmentation training")
            print(f"📁 Data location: {extractor.output_dir}")
            print(f"\n🚀 Next step: Run 02_create_dataset.py")

if __name__ == "__main__":
    main()
