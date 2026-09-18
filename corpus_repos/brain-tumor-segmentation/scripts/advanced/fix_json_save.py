"""Quick fix - save the intensity findings properly"""
import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/home/uphatal/brain_tumor_thesis/segmentation_project/results/v2/intensity_analysis")

# Re-run just the save part with proper serialization
findings_summary = {
    "hypothesis": "CONFIRMED",
    "key_findings": [
        {"modality": "T1C", "metric": "median", "difference_pct": 31.5, "p_value": 0.0143},
        {"modality": "T1C", "metric": "mean",   "difference_pct": 29.7, "p_value": 0.0212},
        {"modality": "T1N", "metric": "median", "difference_pct": 29.4, "p_value": 0.0266},
        {"modality": "T1N", "metric": "mean",   "difference_pct": 28.2, "p_value": 0.0293},
        {"modality": "T1N", "metric": "p95",    "difference_pct": 25.3, "p_value": 0.0372},
        {"modality": "T2W", "metric": "cv",     "difference_pct": 7.8,  "p_value": 0.0155},
    ],
    "recommendation": "Proceed with percentile normalization targeting T1C and T1N",
    "n_failures": 83,
    "n_successes": 121
}

with open(OUTPUT_DIR / "intensity_findings.json", 'w') as f:
    json.dump(findings_summary, f, indent=2)

print("✅ Saved intensity_findings.json")
print("\nKEY INSIGHT:")
print("Failure cases have 29-31% lower T1C intensity than success cases")
print("This means the model cannot distinguish NCR (dark) from background (dark)")
print("Percentile normalization will fix this by standardizing intensity ranges")
