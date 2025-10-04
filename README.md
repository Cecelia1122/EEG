# EEG Motor Imagery Analysis: Forward/Inverse Problems & Classification

- Author: Xue Li  
- Contact: xueli.xl1122@gmail.com  
- Date: October 2025  

A comprehensive neural engineering portfolio project demonstrating expertise in EEG signal processing, source localization, brain-computer interfaces, and computational neuroscience.

---

## 🎯 Project Overview

This project implements a complete pipeline for neural signal analysis, covering:

- Forward and Inverse Problems — EEG source localization using multiple methods  
- Motor Imagery Classification — Machine learning-based BCI with 67% accuracy  
- Source-Space Analysis — Integration of source localization with motor imagery (93% sensor-space accuracy)  
- Nerve Stimulation Simulation — Hodgkin–Huxley model and compound action potentials

Key Achievement: Demonstrates proficiency in biosignal processing, computational modeling, and neural engineering principles relevant to neuroprosthetics, BCIs, and clinical neurophysiology.

---

## 📊 Results Summary

### Part 1: Forward/Inverse Problems

- ✅ Implemented 3 inverse methods (MNE, dSPM, sLORETA)  
- ✅ Source localization to auditory cortex with realistic activation patterns  
- ✅ dSPM peak: 7.39×10⁴, 88.2 ms latency, 1057 active sources  
- ✅ Demonstrates understanding of ill-posed inverse problems and regularization

### Part 2: Motor Imagery Classification

- ✅ 67% test accuracy with LDA (34% improvement over chance)  
- ✅ 66% CV accuracy with CSP feature extraction  
- ✅ Processed 118 epochs from 3 subjects (60 left hand, 58 right hand)  
- ✅ Within published literature range (60–85%) for 2-class motor imagery BCIs

### Part 3: Source-Space Analysis (fsaverage)

- ✅ 93.3% sensor-space accuracy using CSP (±8.9%)  
- ✅ 62.2% source-space accuracy using motor ROI features (±11.3%)  
- ✅ Used fsaverage template for group-level analysis  
- ✅ Bilateral motor cortex activation patterns observed  
- ✅ Validates both sensor and source-space approaches

### Bonus: Nerve Stimulation Simulation

- ✅ Hodgkin–Huxley model: Threshold −47.9 mV, Peak 40.9 mV, Duration 1.48 ms  
- ✅ Fiber bundle: 100 fibers, 5–20 μm diameter, 17.5–70 m/s conduction velocity  
- ✅ Conduction velocity: 48.8 m/s (onset), 22.9 m/s (peak) — physiologically realistic  
- ✅ Demonstrates understanding of nerve excitability and clinical neurophysiology

---

## 📁 Project Structure

```
eeg-neural-engineering-portfolio/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Python dependencies via conda
│
├── part1_forward_inverse.py           # Forward/inverse EEG source localization
├── part2_motor_imagery.py             # Motor imagery BCI classification
├── part3_source_space.py              # Source-space motor imagery analysis
├── nerve_stimulation.py               # Hodgkin–Huxley & CAP simulation
│
├── quick_demo.py                       # A simplified version showing core concepts without full complexity
├── test_pipeline.py                    # Tests core functionality to ensure reproducibility
│
├── setup.py                           # Automated setup script
├── QUICK_START.md                     # 30-minute quick start guide
│
├── data/                              # Auto-downloaded datasets
│   ├── MNE-sample-data/               # ~1.5GB (Part 1)
│   ├── MNE-fsaverage-data/            # ~100MB (Part 3)
│   └── physionet-motor-imagery/       # ~300MB (Parts 2–3)
│
└── results/                           # Generated visualizations
    ├── forward_model_demo.png
    ├── inverse_methods_comparison.png
    ├── inverse_mne_brain.png
    ├── inverse_dspm_brain.png
    ├── inverse_sloreta_brain.png
    ├── motor_imagery_data_visualization.png
    ├── csp_patterns.png
    ├── classification_results.png
    ├── motor_cortex_activation.png
    ├── sensor_vs_source_comparison.png
    ├── brain_activation_left_hand.png
    ├── brain_activation_right_hand.png
    ├── single_fiber_action_potential.png
    ├── compound_action_potential.png
    └── nerve_conduction_study.png
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Cecelia1122/eeg-neural-engineering-portfolio.git
cd eeg-neural-engineering-portfolio

Option A:
# Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
Option B:
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate eeg
```
### Running the Analysis

```bash
# Part 1: Forward/Inverse Problems (~5–10 min)
python part1_forward_inverse.py

# Part 2: Motor Imagery Classification (~10–15 min)
python part2_motor_imagery.py

# Part 3: Source-Space Analysis (~15–20 min)
python part3_source_space.py

# Bonus: Nerve Stimulation (~1 min)
python nerve_stimulation.py
```

Note: First run downloads required datasets automatically.

---

## 🔬 Methods & Technical Details

### Signal Processing Pipeline

- Filtering: 7–30 Hz bandpass (mu and beta bands for motor imagery)  
- Epoching: 1–3 seconds post-cue for motor imagery tasks  
- Artifact Rejection: Amplitude-based (≈300 μV threshold)  
- Preprocessing: Montage application; ICA optional

### Feature Extraction

- Common Spatial Patterns (CSP): 4 components, log-variance features  
- Band Power: Mu (8–12 Hz) and Beta (13–30 Hz) spectral power  
- Source ROI: Mean power in bilateral precentral gyrus (motor cortex)

### Source Localization

- Forward Model: fsaverage template with 3-layer BEM for EEG  
- Inverse Methods:  
  - MNE (Minimum Norm Estimate): L2 regularization, λ² = 1/9  
  - dSPM (dynamic Statistical Parametric Mapping): Noise-normalized MNE  
  - sLORETA (standardized LORETA): Zero-error localization in noise-free case  
- Regularization: SNR = 3, loose orientation constraint = 0.2, depth weighting = 0.8

### Classification

- Algorithms: LDA, SVM (linear), SVM (RBF)  
- Validation: 5-fold stratified cross-validation  
- Train/Test Split: 80/20  
- Metrics: Accuracy, precision, recall, F1-score, confusion matrices

### Biophysical Modeling

- Hodgkin–Huxley Equations: Ion channel dynamics (Na⁺, K⁺, leak)  
- Compound Action Potentials: 100 fibers with clipped diameters (5–20 μm)  
- Conduction Velocity: ≈ 3.5 × diameter (μm) for myelinated fibers (17.5–70 m/s)  
- Clinical Simulation: Two-point stimulation for CV measurement

---

## 📈 Detailed Results & Discussion

### Part 1: Forward/Inverse — Summary Table

| Method   | Peak Amplitude | Peak Latency | Active Sources | Spatial Characteristics          |
|----------|----------------|--------------|----------------|----------------------------------|
| MNE      | 7.47×10⁴       | 103.2 ms     | 94             | Broad, diffuse activation        |
| dSPM     | 7.39×10⁴       | 88.2 ms      | 1057           | Sharp, focal localization        |
| sLORETA  | 5.62×10⁴       | 101.6 ms     | 1004           | Smooth, intermediate spread      |

Interpretation: dSPM provides an effective balance of spatial precision and noise sensitivity for many applications.

### Part 2: Motor Imagery — Performance Context

67% Test Accuracy Analysis:
- Literature: Published 2-class motor imagery BCIs report 60–85%  
- Relative improvement over chance: (0.67 − 0.50) / 0.50 = 34%  
- Cross-validation: 66% mean CV accuracy (±17.1% std)

Performance Factors:
- Limited training data: 118 epochs from 3 subjects  
- Subject variability: Motor imagery varies across individuals  
- Generic parameters: No subject-specific band/CSP tuning  
- Classifier comparison: LDA (67%), SVM-linear (62%), SVM-RBF (54%)

Potential Improvements:
- Add more subjects  
- Optimize CSP components (6–8)  
- Subject-specific band selection  
- Shrinkage LDA

### Part 3: Sensor vs Source-Space — Insights

- Sensor-space performance: 93.3% accuracy with 1 subject shows strong discriminative patterns (variance ±8.9%)  
- Source-space results: 62.2% accuracy using two ROI features (bilateral motor cortex power)  
- Key finding: Motor cortex activation patterns validate contralateral organization

### Bonus: Nerve Stimulation — Validation

- Hodgkin–Huxley: Resting −65 mV, threshold −47.9 mV, peak +40.9 mV, duration 1.48 ms  
- Conduction velocity: Onset-based 48.8 m/s in normal motor nerve range (40–60 m/s)

---

## 💡 Key Contributions & Learning Outcomes

- EEG Signal Processing; Source Localization (MNE, dSPM, sLORETA); Machine Learning (CSP, LDA, SVM); Computational Modeling (Hodgkin–Huxley)  
- Clean, reproducible code and publication-quality figures

---

## 📚 Dataset Information

- PhysioNet Motor Movement/Imagery Database (runs 4, 8, 12; subjects 1–3)  
- MNE Sample Dataset for forward/inverse demos  
- MNE fsaverage template for group-level source localization

---

## 🎓 Application to Neural Engineering Research

Readiness for PhD research demonstrated across signal processing, inverse problems, BCIs, and computational modeling relevant to neuroprosthetics and clinical neurophysiology.

---

## 🔮 Future Directions

- More subjects; EEGNet baseline; online BCI; subject MRIs; connectivity; multimodal; clinical validation; closed-loop neurofeedback

---

## 📖 References

- Hämäläinen & Ilmoniemi (1994); Pascual-Marqui (2002); Pfurtscheller & Neuper (2001); Blankertz et al. (2008); Ramoser et al. (2000); Hodgkin & Huxley (1952); Gramfort et al. (2013)

---

## 💻 Technical Requirements

- Python 3.8+; MNE 1.5+; NumPy/SciPy/scikit-learn; matplotlib/seaborn; optional pyvista for 3D

- macOS, Linux, Windows supported

---

## 📧 Contact

- Xue Li — xueli.xl1122@gmail.com  
- GitHub: [@Cecelia1122](https://github.com/Cecelia1122)  
- Repository: https://github.com/Cecelia1122/eeg-neural-engineering-portfolio

---

## 📄 License

MIT License.  
Data: PhysioNet ODbL; MNE sample/fsaverage BSD-3-Clause.

---

## 🙏 Acknowledgments

- MNE-Python team; PhysioNet; mentors and collaborators; AI tooling support# EEG
