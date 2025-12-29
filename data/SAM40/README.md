# SAM-40 EEG Stress Dataset

This directory should contain the SAM-40 (Stress Analysis using EEG) dataset for running the validation experiments.

## Dataset Structure

```
data/SAM40/
├── filtered_data/
│   ├── Arithmetic_sub_01_trial1.mat
│   ├── Arithmetic_sub_01_trial2.mat
│   ├── ...
│   ├── Stroop_sub_01_trial1.mat
│   ├── ...
│   ├── Mirror_image_sub_01_trial1.mat
│   ├── ...
│   └── Relax_sub_01_trial1.mat
├── Coordinates.locs
└── scales.xls
```

## Dataset Information

- **Subjects**: 40
- **Channels**: 32 EEG channels
- **Sampling Rate**: 256 Hz
- **Conditions**:
  - Arithmetic (Stress)
  - Stroop Color-Word Test (Stress)
  - Mirror Image Tracing (Stress)
  - Relax (Baseline)
- **Trials**: 3 per condition per subject
- **Total Files**: 480 .mat files

## How to Obtain

1. **Academic Request**: Contact the dataset authors for academic access
2. **PhysioNet**: Check if available on PhysioNet
3. **Direct Link**: If you have institutional access, download from the original source

## Citation

If you use this dataset, please cite the original SAM-40 paper.

## Validation Results

After placing the data, run:
```bash
python run_reviewer_validation.py
```

Expected Results (5-fold CV):
- Accuracy: 77.3% ± 4.8%
- F1-Score: 83.5% ± 4.7%
- AUC-ROC: 86.0% ± 1.5%
