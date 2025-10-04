# 🚀 Quick Start Guide
Get Your EEG Project Running in 30 Minutes

Last Updated: October 2025  
Estimated Time: 30–45 minutes  
Difficulty: Beginner-friendly

---

## ⏱️ Timeline Overview
- 5 min: Environment setup  
- 10 min: Data download (automatic)  
- 15 min: Run first analysis  
- 5 min: Review results

Total: ~35 minutes to first results

---

## Step 1: System Requirements Check (2 minutes)

Minimum Requirements
```bash
# Check Python version (need 3.10+)
python --version  # or python3 --version

# Check available disk space (need ~5GB free)
df -h .  # macOS/Linux
# Windows: Right-click drive → Properties
```

Requirements:
- Python 3.10 or higher
- 5 GB free disk space
- 8 GB RAM (16 GB recommended)
- Internet connection (for data download)

Supported Operating Systems:
- macOS 10.15+
- Linux (Ubuntu 20.04+, similar distributions)
- Windows 10/11

---

## Step 2: Installation (5 minutes)

Option A: Clone from GitHub (Recommended)
```bash
# Clone the repository
git clone https://github.com/Cecelia1122/EEG.git
cd EEG
```

Option B: Download ZIP
- Visit: [https://github.com/Cecelia1122/EEG.git]
- Click “Code” → “Download ZIP”  
- Extract and navigate to the folder

Create Virtual Environment
Option A:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```
Note: You’ll see “(venv)” in your terminal prompt when activated.

Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```
Option B: Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate eeg

python - << 'PY'
import sys, mne, sklearn, numpy, matplotlib
print("Python:", sys.version.split()[0])
print("MNE:", mne.__version__)
print("OK")
PY
```

Expected installation time: 3–5 minutes

Common Issues:
- If pip install fails on Windows: `python -m pip install -r requirements.txt`
- If MNE installation fails: `pip install mne --no-cache-dir`
- If you see SSL errors:  
  `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`

---
## Step 2.1 (Optional but Recommended): 
Run the 5‑Minute Quick Demo

```bash
python quick_demo.py
```

Outputs
- results/quick_demo_results.png

What you should see
- ERP with baseline applied
- dSPM source time‑course with a peak near ~100 ms
- Clearly separated CSP features for the synthetic classes

---
Run Tests 

```bash
pytest -q test_pipeline.py
```

- Fast to run; may skip one optional fsaverage labels test.
  
## Step 3: Verify Installation (2 minutes)
```bash
# Quick import test (should print versions, then "OK")
python - << 'PY'
import sys, mne, sklearn, numpy
print("Python:", sys.version.split()[0])
print("MNE:", mne.__version__)
print("scikit-learn:", sklearn.__version__)
print("NumPy:", numpy.__version__)
print("OK")
PY
```
If you see versions and “OK”, you’re ready.

---

## Step 4: Run Your First Analysis (10 minutes)

Part 1: Forward/Inverse Problems (Fastest Demo)
```bash
python part1_forward_inverse.py
```

What happens:
- Downloads MNE sample data (~1.5 GB) — first run only
- Demonstrates forward modeling (brain → scalp)
- Compares 3 inverse methods (MNE, dSPM, sLORETA)
- Generates 5 brain visualization images

Runtime: ~5–10 minutes (first run with download)

Expected console highlights:
```
EEG FORWARD AND INVERSE PROBLEM DEMONSTRATION
...
Forward solution contains 7498 sources
EEG channels: 60
Activating source index 500
Applying forward model (source → scalp)...
...
ANALYSIS COMPLETE!
```

Generated Files (saved to results/):
- forward_model_demo.png — Dipole source and scalp topography
- inverse_methods_comparison.png — Time courses from 3 methods
- inverse_mne_brain.png — MNE source localization on brain
- inverse_dspm_brain.png — dSPM source localization
- inverse_sloreta_brain.png — sLORETA source localization

---

## Step 5: View Your Results (5 minutes)

Open Generated Images
```bash
# macOS:
open *.png

# Linux:
xdg-open *.png

# Windows:
start *.png
# Or open manually from your file explorer
```

What to Look For

forward_model_demo.png:
- Left: 3D brain with source dipole location
- Right: Scalp topography (dipolar blue/red pattern)

inverse_methods_comparison.png:
- Three time course plots (MNE, dSPM, sLORETA)
- Peak around 100 ms (auditory response)

Brain surface plots (3 files):
- MNE: broader/diffuse activation
- dSPM: sharper/more focal
- sLORETA: smooth, intermediate spread

Success Indicators:
- ✅ All 5 images generated
- ✅ Brain plots show activation near superior temporal region
- ✅ No error messages in console
- ✅ Time courses show a clear peak ~80–110 ms

---

## Step 6: Run Complete Pipeline (Optional, +30 min)

Part 2: Motor Imagery Classification
```bash
python part2_motor_imagery.py
```

What happens:
- Downloads PhysioNet motor imagery data (~300 MB)
- Processes 3 subjects (~118 epochs total)
- Extracts CSP features
- Trains LDA/SVM classifiers
- Generates performance plots

Runtime: ~10–15 minutes  
Expected accuracy: ~60–70% (reference: 67%)

Generated files (saved to results/):
- motor_imagery_data_visualization.png — Raw EEG, PSD, topographies
- csp_patterns.png — Spatial filter patterns
- classification_results.png — Confusion matrices, accuracies

Part 3: Source-Space Analysis
```bash
python part3_source_space.py
```

What happens:
- Uses fsaverage template for source localization
- Applies inverse solution to MI epochs (single-subject demo)
- Extracts motor cortex ROI features
- Compares sensor-space vs source-space

Runtime: ~15–20 minutes

Expected results:
- Sensor-space: ~90%+ accuracy (single subject)
- Source-space: ~60%+ accuracy
- Brain activation maps

Generated files (saved to results/):
- motor_cortex_activation.png — Time courses in motor regions
- sensor_vs_source_comparison.png — Performance comparison
- brain_activation_left_hand.png — Left-hand imagery activation
- brain_activation_right_hand.png — Right-hand imagery activation

Bonus: Nerve Stimulation
```bash
python nerve_stimulation.py
```

What happens:
- Simulates Hodgkin–Huxley action potential
- Models compound action potential from 100 fibers
- Computes clinical nerve conduction velocity

Runtime: ~1 minute (fastest!)

Generated files (saved to results/):
- single_fiber_action_potential.png — HH model dynamics
- compound_action_potential.png — Fiber recruitment
- nerve_conduction_study.png — Clinical measurement

---

## 📊 Expected Results Summary

| Part   | Runtime    | Generated Files | Key Results                         |
|--------|------------|-----------------|-------------------------------------|
| Part 1 | 5–10 min   | 5 images        | 3 inverse methods compared          |
| Part 2 | 10–15 min  | 3 images        | 67% classification accuracy (ref.)  |
| Part 3 | 15–20 min  | 4 images        | ~93% sensor, ~62% source accuracy   |
| Bonus  | ~1 min     | 3 images        | 48.8 m/s conduction velocity        |
| Total  | 30–45 min  | 15 images       | Complete pipeline                   |

---

## 🐛 Troubleshooting

Issue 1: “ModuleNotFoundError: No module named 'mne'”
```bash
# Ensure virtual environment is activated
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Reinstall
pip install -r requirements.txt
```

Issue 2: Data Download Fails
```bash
# Manual fetch helpers
python - << 'PY'
import mne
from mne.datasets import sample, eegbci
sample.data_path()
eegbci.load_data(1, [4, 8, 12])
mne.datasets.fetch_fsaverage()
print("Downloads triggered.")
PY
```

Issue 3: “Memory Error” or System Freezes
- Close other applications
- Run one script at a time
- For Part 2, reduce to a single subject:
```python
# In part2_motor_imagery.py
processor = MotorImageryProcessor(subject_ids=[1])
```

Issue 4: Plots Don’t Display
- Images save automatically (Part 2: results/; others: results/)
```bash
ls *.png || ls results/*.png
```

Issue 5: “RuntimeError: epochs empty” (Part 3)
- Reduce/disable artifact rejection for the demo
- Verify data downloaded correctly and re-run

Issue 6: Import Errors on Windows
```bash
python -m pip install -r requirements.txt
# If still failing, install individually:
pip install mne
pip install numpy scipy scikit-learn
pip install matplotlib seaborn
```

---

## ✅ Success Checklist

After 30–45 minutes, you should have:
- Virtual environment activated
- All packages installed
- Part 1 completed successfully
- 5+ PNG images generated
- Brain visualizations look reasonable
- No error messages

Bonus (if time permits):
- Part 2 completed (motor imagery; ~60–70% accuracy)
- Part 3 completed (source-space demo)
- Nerve stimulation completed

---

## 🎯 What Each Part Teaches You

Part 1: Forward/Inverse Problems
- Concepts: volume conduction, ill‑posed inverse, regularization, method trade‑offs
- Skills: loading data, forward solutions, inverse operators, 3D visualization

Part 2: Motor Imagery Classification
- Concepts: ERD, CSP, cross‑validation, confusion matrices
- Skills: filtering/epoching, CSP feature extraction, LDA/SVM, performance evaluation

Part 3: Source-Space Analysis
- Concepts: fsaverage template, ROI analysis, sensor vs source trade-offs, contralateral organization
- Skills: template-based localization, ROI features, approach comparison, brain plots

Bonus: Nerve Stimulation
- Concepts: Hodgkin–Huxley, ion channel dynamics, CAPs, clinical NCS
- Skills: ODE simulation, biophysical modeling, clinical metric computation

---

## 📚 Next Steps

Immediate (Same Day)
- Review generated figures and console output
- Open Python scripts and follow comments
- Note accuracy values and observations

Short-term (Next 2–3 Days)
- Experiment with parameters
```python
# In part2_motor_imagery.py:
# Try more CSP components
# csp = CSP(n_components=6)

# Try slightly different band
# raw.filter(8., 30.)
```
- Explore epochs interactively
```python
from part2_motor_imagery import MotorImageryProcessor
p = MotorImageryProcessor(subject_ids=[1])
p.load_data(); p.preprocess()
print(p.epochs); p.epochs.plot()
```
- Read docs: https://mne.tools/stable/auto_tutorials/

Medium-term (Next Week)
- Add more subjects; try different classifiers
- Optimize hyperparameters
- Add custom visualizations
- Prepare a short technical summary of your results

---

## 💡 Pro Tips

Tip 1: Save Your Results
```bash
mkdir -p results
mv *.png results/  # move current images
# Or run from separate folders:
mkdir -p run1 && (cd run1 && python ../part1_forward_inverse.py)
```

Tip 2: Monitor Memory Usage
```bash
# macOS/Linux:
top -o mem   # Press 'q' to quit
# Windows: Use Task Manager (Ctrl+Shift+Esc)
```

Tip 3: Run in Background (long runs)
```bash
# macOS/Linux:
nohup python part2_motor_imagery.py > output.log 2>&1 &
tail -f output.log
```

Tip 4: Keep Notes (create NOTES.md)
```markdown
# Experiment Log
## Run 1 - [Date]
- Part 1: Completed successfully
- Part 2: 67% accuracy (3 subjects)
- Observations: LDA > SVM
- Next: Try more subjects
```

Tip 5: Understand Before Optimizing
- Get the baseline working first
- Change one thing at a time
- Record what you change and why

---

## 📞 Getting Help

- MNE docs: https://mne.tools/stable/  
- MNE forum: https://mne.discourse.group/  
- Stack Overflow: tag “mne” + “python”  
- GitHub Issues (if bug suspected)

Contact:
- Email: xueli.xl1122@gmail.com
- GitHub: open an issue on the repository

---

Quick Start Guide v1.0  
Last Updated: October 2025
