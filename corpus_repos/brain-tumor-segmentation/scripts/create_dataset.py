import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

class BraTSSegmentationDataset(Dataset):
    """
    BraTS Dataset for 3D Segmentation
    Returns: 4-channel input (T1, T1CE, T2, FLAIR) + segmentation mask
    """
    
    def __init__(self, patient_dirs, transform=None, crop_size=(128, 128, 128)):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        data = self.load_patient_data(patient_dir)
        
        if data is None:
            return torch.zeros(4, *self.crop_size), torch.zeros(*self.crop_size)
        
        volume, segmentation = data
        
        if self.transform:
            volume, segmentation = self.transform(volume, segmentation)
        
        return volume, segmentation
    
    def load_patient_data(self, patient_dir):
        """Load all MRI sequences and segmentation for one patient"""
        
        try:
            files = {f.name: f for f in patient_dir.iterdir()}
            
            t1n = self.load_nifti(self.find_file(files, 't1n'))
            t1c = self.load_nifti(self.find_file(files, 't1c'))
            t2w = self.load_nifti(self.find_file(files, 't2w'))
            t2f = self.load_nifti(self.find_file(files, 't2f'))
            seg = self.load_nifti(self.find_file(files, 'seg'))
            
            volume = np.stack([t1n, t1c, t2w, t2f], axis=0)
            
            volume = self.preprocess_volume(volume)
            seg = self.preprocess_segmentation(seg)
            
            volume = self.crop_or_pad(volume)
            seg = self.crop_or_pad(seg, is_seg=True)
            
            volume = torch.FloatTensor(volume)
            seg = torch.LongTensor(seg)
            
            return volume, seg
            
        except Exception as e:
            print(f"Error loading {patient_dir.name}: {e}")
            return None
    
    def find_file(self, files, sequence):
        """Find file matching sequence type"""
        for fname, fpath in files.items():
            if sequence in fname:
                return fpath
        raise FileNotFoundError(f"Sequence {sequence} not found")
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return numpy array"""
        nifti = nib.load(str(filepath))
        return nifti.get_fdata()
    
    def preprocess_volume(self, volume):
        """Normalize each sequence independently"""
        preprocessed = np.zeros_like(volume)
        
        for i in range(volume.shape[0]):
            channel = volume[i]
            channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
            
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val > min_val:
                channel = (channel - min_val) / (max_val - min_val)
            
            preprocessed[i] = channel
        
        return preprocessed
    
    def preprocess_segmentation(self, seg):
        """Convert BraTS segmentation labels to sequential indices"""
        # BraTS labels: 0 (background), 1 (NCR), 2 (ED), 4 (ET)
        # Remap to: 0, 1, 2, 3 for proper one-hot encoding
        seg_remapped = np.zeros_like(seg, dtype=np.int64)
        seg_remapped[seg == 0] = 0  # Background
        seg_remapped[seg == 1] = 1  # Necrotic/Non-enhancing tumor
        seg_remapped[seg == 2] = 2  # Peritumoral edema
        seg_remapped[seg == 4] = 3  # GD-enhancing tumor
        return seg_remapped
    
    def crop_or_pad(self, volume, is_seg=False):
        """Crop or pad volume to target size"""
        target = self.crop_size
        
        if is_seg:
            current = volume.shape
            pad_value = 0
        else:
            current = volume.shape[1:]
            pad_value = 0
        
        result = []
        for curr, tgt in zip(current, target):
            if curr < tgt:
                pad_before = (tgt - curr) // 2
                pad_after = tgt - curr - pad_before
                result.append((pad_before, pad_after))
            else:
                crop_before = (curr - tgt) // 2
                result.append((crop_before, curr - tgt - crop_before))
        
        if is_seg:
            padded = np.pad(volume, result, mode='constant', constant_values=pad_value)
            h_start, w_start, d_start = [r[0] for r in result]
            h_end = h_start + target[0]
            w_end = w_start + target[1]
            d_end = d_start + target[2]
            return padded[h_start:h_end, w_start:w_end, d_start:d_end]
        else:
            result = [(0, 0)] + result
            padded = np.pad(volume, result, mode='constant', constant_values=pad_value)
            h_start, w_start, d_start = [r[0] for r in result[1:]]
            h_end = h_start + target[0]
            w_end = w_start + target[1]
            d_end = d_start + target[2]
            return padded[:, h_start:h_end, w_start:w_end, d_start:d_end]


def create_data_splits(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split patients into train/val/test sets"""
    
    print("📊 Creating data splits...")
    data_path = Path(data_dir)
    
    # Find all patient directories
    patient_dirs = sorted([d for d in data_path.iterdir() 
                          if d.is_dir() and d.name.startswith("BraTS-GLI-")])
    
    print(f"Found {len(patient_dirs)} patients")
    
    if len(patient_dirs) == 0:
        print(f"❌ No patient directories found in: {data_path}")
        print(f"Please check if path exists and contains BraTS-GLI-* folders")
        return None, None
    
    patient_ids = [d.name for d in patient_dirs]
    
    train_ids, temp_ids = train_test_split(
        patient_ids, train_size=train_ratio, random_state=42
    )
    
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_ratio_adjusted, random_state=42
    )
    
    print(f"✅ Train: {len(train_ids)} patients")
    print(f"✅ Val: {len(val_ids)} patients")
    print(f"✅ Test: {len(test_ids)} patients")
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'data_path': str(data_path)  # Save data path for reference
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits_file = output_path / 'data_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"💾 Splits saved to: {splits_file}")
    
    return splits, patient_dirs


def test_dataset_loading(data_dir):
    """Test if dataset loads correctly"""
    
    print("\n🧪 Testing dataset loading...")
    
    data_path = Path(data_dir)
    patient_dirs = sorted([d for d in data_path.iterdir() 
                          if d.is_dir() and d.name.startswith("BraTS-GLI-")])
    
    if not patient_dirs:
        print("❌ No patient directories found!")
        return
    
    print(f"Testing with patient: {patient_dirs[0].name}")
    
    dataset = BraTSSegmentationDataset([patient_dirs[0]])
    
    try:
        volume, seg = dataset[0]
        
        print(f"✅ Volume shape: {volume.shape}")
        print(f"✅ Segmentation shape: {seg.shape}")
        print(f"✅ Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"✅ Unique seg labels: {torch.unique(seg).tolist()}")
        
        print("\n🎉 Dataset loading successful!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def main():
    base_dir = Path.home() / "brain_tumor_thesis/segmentation_project"
    
    # UPDATED: Use your existing extracted data location
    data_dir = Path.home() / "brain_tumor_thesis/data/real_datasets/brats_raw/BraTS-GLI/extracted_data/training_data1_v2"
    
    splits_dir = base_dir / "data/splits"
    
    print(f"Looking for data in: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        print("\nPlease check your data location. Common paths:")
        print("  - ~/brain_tumor_thesis/data/real_datasets/brats_raw/BraTS-GLI/extracted_data/")
        print("  - ~/brain_tumor_thesis/data/real_datasets/brats_raw/BraTS-GLI/extracted_data/training_data1_v2/")
        return
    
    # Create splits
    result = create_data_splits(data_dir, splits_dir)
    
    if result[0] is None:
        return
    
    splits, patient_dirs = result
    
    # Test dataset
    test_dataset_loading(data_dir)
    
    print("\n" + "="*60)
    print("🎉 DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"✅ Data splits created and saved")
    print(f"✅ Dataset class tested and working")
    print(f"\n🚀 Next step: Run 03_train_unet.py to start training")


if __name__ == "__main__":
    main()
