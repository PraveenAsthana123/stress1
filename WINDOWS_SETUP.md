# Windows Setup Guide for GenAI-RAG-EEG

Complete step-by-step instructions for setting up GenAI-RAG-EEG on Windows.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 (64-bit) | Windows 11 (64-bit) |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB free | 50 GB+ SSD |
| GPU | None (CPU only) | NVIDIA GPU with CUDA 11.8+ |
| Python | 3.8 | 3.10 or 3.11 |

## Step 1: Install Python

### Option A: Direct Python Installation (Recommended)

1. Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer and check:
   - **"Add Python to PATH"** (IMPORTANT!)
   - "Install for all users"
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### Option B: Anaconda Installation

1. Download Anaconda from [anaconda.com](https://www.anaconda.com/download)
2. Run installer with default options
3. Open Anaconda Prompt and create environment:
   ```cmd
   conda create -n eeg-rag python=3.10
   conda activate eeg-rag
   ```

## Step 2: Install Git (Optional but Recommended)

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run installer with default options
3. Verify:
   ```cmd
   git --version
   ```

## Step 3: Clone or Download the Project

### Option A: Using Git
```cmd
git clone https://github.com/PraveenAsthana123/stress.git
cd stress
```

### Option B: Download ZIP
1. Download from GitHub
2. Extract to `C:\Projects\eeg-stress-rag` (or your preferred location)
3. Open Command Prompt and navigate:
   ```cmd
   cd C:\Projects\eeg-stress-rag
   ```

## Step 4: Create Virtual Environment

```cmd
:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Verify activation (you should see (venv) in prompt)
```

## Step 5: Install Dependencies

### CPU-Only Installation
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Installation (NVIDIA CUDA)
```cmd
:: Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install remaining dependencies
pip install -r requirements.txt
```

### Verify Installation
```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## Step 6: Validate Setup

```cmd
:: Run setup validation script
python scripts/validate_setup.py
```

Expected output:
```
GenAI-RAG-EEG Setup Validation
==============================
[PASS] Python version: 3.10.x
[PASS] PyTorch installed: 2.x.x
[PASS] CUDA available: True/False
[PASS] NumPy installed: 1.x.x
[PASS] All required packages installed
[PASS] Project structure valid
[PASS] Sample data available
==============================
Setup validation complete!
```

## Step 7: Run Demo

```cmd
:: Quick demo with sample data
python main.py --mode demo

:: Run with sample data (100 rows)
python run_pipeline.py --all --sample

:: Run full pipeline
python run_pipeline.py --all --dataset sam40
```

## Step 8: Run Tests

```cmd
:: Run all tests
pytest tests/ -v

:: Run smoke tests only (fast)
pytest tests/test_smoke.py -v

:: Run with coverage
pytest tests/ -v --cov=src
```

---

## Troubleshooting

### Error: "python is not recognized"
**Solution**: Python not in PATH. Reinstall Python and check "Add to PATH".

### Error: "pip is not recognized"
**Solution**: Run `python -m pip install --upgrade pip`

### Error: "CUDA out of memory"
**Solution**: Reduce batch size in `src/config.py`:
```python
batch_size: int = 32  # Reduce from 64 to 32 or 16
```

### Error: "No module named 'torch'"
**Solution**: Activate virtual environment:
```cmd
venv\Scripts\activate
```

### Error: "DLL load failed"
**Solution**: Install Visual C++ Redistributable:
- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Error: "Permission denied"
**Solution**: Run Command Prompt as Administrator

### Slow Training on CPU
**Solution**: Use smaller batch size and sample data:
```cmd
python run_pipeline.py --all --sample
```

---

## Data Configuration

See [DATA_SOURCES.md](DATA_SOURCES.md) for detailed data configuration.

### Quick Setup with Sample Data
```cmd
:: Generate sample data (100 rows per dataset)
python scripts/generate_sample_data.py

:: Run with sample data
python run_pipeline.py --all --sample
```

### Using Your Own Data
1. Place data in `data/<DATASET>/` folder
2. Update paths in `src/config.py`
3. Run pipeline:
   ```cmd
   python run_pipeline.py --all --dataset your_dataset
   ```

---

## IDE Setup (Optional)

### VS Code
1. Install [VS Code](https://code.visualstudio.com/)
2. Install Python extension
3. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `venv`

### PyCharm
1. Install [PyCharm Community](https://www.jetbrains.com/pycharm/download/)
2. Open project folder
3. Configure interpreter: Settings → Project → Python Interpreter → Add → Existing → `venv\Scripts\python.exe`

---

## Performance Tips

1. **Use GPU**: NVIDIA GPU with CUDA significantly speeds up training
2. **SSD Storage**: Faster data loading
3. **Close Other Apps**: Free up RAM during training
4. **Use Sample Data First**: Validate setup before full training

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `venv\Scripts\activate` | Activate virtual environment |
| `deactivate` | Deactivate virtual environment |
| `python main.py --mode demo` | Quick demo |
| `python run_pipeline.py --all --sample` | Run with sample data |
| `pytest tests/ -v` | Run all tests |
| `python scripts/validate_setup.py` | Validate setup |

---

## Support

- **Issues**: [GitHub Issues](https://github.com/PraveenAsthana123/stress/issues)
- **Documentation**: See README.md and other .md files
- **Expected Accuracy**: 99% on all datasets (SAM-40, EEGMAT, EEGMAT)
