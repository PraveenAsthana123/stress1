# Sample Data for GenAI-RAG-EEG

100-row sample datasets for testing and demonstration.

## Datasets

| Dataset | Samples | Channels | Rate | Paradigm |
|---------|---------|----------|------|----------|
| SAM-40 | 100 | 32 | 128 Hz | Cognitive (Stroop) |
| WESAD | 100 | 14 | 700 Hz | TSST Protocol |
| EEGMAT | 100 | 21 | 500 Hz | Mental Arithmetic |

## Loading Data

```python
import numpy as np
data = np.load('data/SAM40/sample_100/sam40_sample_100.npz')
X, y = data['X'], data['y']
```

Generated: 2025-12-29 19:50:43
