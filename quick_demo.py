"""
Quick Demo: 5-Minute EEG Analysis
A simplified version showing core concepts without full complexity

Author: Xue Li
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("QUICK DEMO: 5-Minute EEG Analysis")
print("=" * 60)
print("\nThis demonstrates:")
print("1. Loading EEG data")
print("2. Source localization (inverse problem)")
print("3. Simple classification example")
print("=" * 60)


# Auto-create a results directory next to this script
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def out(name: str) -> str:
    return str(RESULTS_DIR / name)

# ============================================================================
# Part 1: Load Sample Data
# ============================================================================

print("\n[1/3] Loading sample data...")
data_path = Path(sample.data_path())
subjects_dir = data_path / 'subjects'  # not used here but kept for completeness

# Load evoked data
fname_evoked = data_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'
ev = mne.read_evokeds(str(fname_evoked), condition='Left Auditory', verbose=False)
# If a specific condition is given, MNE returns an Evoked (not a list)
evoked = ev if isinstance(ev, mne.Evoked) else ev[0]

# Modern API: pick by type name (no keyword args)
try:
    evoked.pick('eeg')  # removes everything except EEG
except TypeError:
    # Fallback for older versions
    picks = mne.pick_types(evoked.info, eeg=True, meg=False)
    evoked.pick(picks)

# Baseline-correct to stabilize pre-stimulus
evoked.apply_baseline((None, 0.0))

print(f"   Loaded: {len(evoked.ch_names)} EEG channels")
print(f"   Duration: {len(evoked.times)} time points")
print(f"   Sampling rate: {evoked.info['sfreq']} Hz")

# ============================================================================
# Part 2: Source Localization
# ============================================================================

print("\n[2/3] Computing source localization...")

# Load forward solution
fname_fwd = data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(str(fname_fwd))
fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)

# Load noise covariance
fname_cov = data_path / 'MEG' / 'sample' / 'sample_audvis-cov.fif'
noise_cov = mne.read_cov(str(fname_cov))

# Create inverse operator
inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)

# Apply inverse solution (dSPM)
stc = apply_inverse(evoked, inv, lambda2=1.0 / 9.0, method='dSPM', verbose=False)

# Get peak location restricted to post-stim window (50–150 ms) and as index
peak_idx, peak_time = stc.get_peak(hemi='lh', vert_as_index=True, tmin=0.05, tmax=0.15)

print(f"   Source localization complete!")
print(f"   Peak index (lh): {peak_idx}")
print(f"   Peak time: {peak_time * 1000:.1f} ms")
print(f"   Peak amplitude: {float(stc.data.max()):.2f}")

# ============================================================================
# Part 3: Simple Classification Demo
# ============================================================================

print("\n[3/3] Classification demo (synthetic data)...")

# Create synthetic motor imagery-like data
np.random.seed(42)
n_epochs = 60
n_channels = 10
n_times = 100

# Class 1: Higher activity in left channels
X1 = np.random.randn(n_epochs // 2, n_channels, n_times)
X1[:, :5, :] += 0.8  # Boost left channels

# Class 2: Higher activity in right channels
X2 = np.random.randn(n_epochs // 2, n_channels, n_times)
X2[:, 5:, :] += 0.8  # Boost right channels

# Combine
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(n_epochs // 2), np.ones(n_epochs // 2)])

# Apply CSP
csp = CSP(n_components=2, reg=None, log=True)
X_csp = csp.fit_transform(X, y)

# Train classifier
clf = LinearDiscriminantAnalysis()
clf.fit(X_csp, y)

# Test accuracy (on training set for demo simplicity)
accuracy = clf.score(X_csp, y)

print(f"   Created synthetic data: {n_epochs} epochs, {n_channels} channels")
print(f"   Applied CSP feature extraction")
print(f"   Classification accuracy: {accuracy:.1%}")
print(f"   (High accuracy expected - this is training data)")

# ============================================================================
# Visualization
# ============================================================================

print("\n[4/3] Creating visualizations...")

fig = plt.figure(figsize=(15, 5))

# Plot 1: Evoked response
ax1 = plt.subplot(131)
evoked.plot(axes=ax1, show=False, spatial_colors=True, gfp=True)
ax1.set_title('EEG Evoked Response', fontweight='bold')

# Plot 2: Source time course at peak index
ax2 = plt.subplot(132)
ax2.plot(stc.times * 1000, stc.data[peak_idx, :])
ax2.axvline(peak_time * 1000, color='r', linestyle='--', label='Peak')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Source Amplitude')
ax2.set_title('Source Time Course at Peak', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: CSP features
ax3 = plt.subplot(133)
colors = ['blue' if yi == 0 else 'red' for yi in y]
ax3.scatter(X_csp[:, 0], X_csp[:, 1], c=colors, alpha=0.6, s=50)
ax3.set_xlabel('CSP Component 1')
ax3.set_ylabel('CSP Component 2')
ax3.set_title('CSP Feature Space', fontweight='bold')
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Class 2', markerfacecolor='red', markersize=8),
]
ax3.legend(handles=legend_elems)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out('quick_demo_results.png'), dpi=150, bbox_inches='tight')
print("   Saved: quick_demo_results.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)
print("\nWhat you just did:")
print("  1. Loaded real EEG data (sample dataset)")
print("  2. Computed source localization using dSPM inverse method")
print("  3. Demonstrated CSP feature extraction and classification")
print("\nGenerated files:")
print("  - quick_demo_results.png (3-panel visualization)")
print("\nNext steps:")
print("  - Run full Part 1: python part1_forward_inverse.py")
print("  - Run full Part 2: python part2_motor_imagery.py")
print("  - Explore the code in detail")
print("=" * 60)

print("\nKey concepts demonstrated:")
print("  ✓ EEG data structure and visualization")
print("  ✓ Inverse problem solving (source localization)")
print("  ✓ Feature extraction for classification")
print("  ✓ Spatial patterns in brain signals")

print(f"\nTotal time: <5 minutes")
print(f"Complexity: Beginner-friendly")
print(f"Purpose: Quick overview of pipeline")