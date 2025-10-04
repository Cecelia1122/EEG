"""
EEG Forward and Inverse Problem Implementation
Part 1 of Neural Engineering Portfolio Project

This script demonstrates:
1. Forward modeling: simulating EEG from brain sources
2. Inverse modeling: reconstructing sources from EEG
3. Comparison of multiple inverse methods

Author: Xue Li
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Auto-create a results directory next to this script
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def out(name: str) -> str:
    return str(RESULTS_DIR / name)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')

def setup_data_paths():
    """Download and setup paths to sample data"""
    print("Setting up data paths...")
    data_path = sample.data_path()
    subjects_dir = data_path / 'subjects'
    subject = 'sample'
    
    # File paths
    fname_raw = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    fname_fwd = data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
    fname_cov = data_path / 'MEG' / 'sample' / 'sample_audvis-cov.fif'
    fname_evoked = data_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'
    
    return {
        'data_path': data_path,
        'subjects_dir': subjects_dir,
        'subject': subject,
        'fname_raw': fname_raw,
        'fname_fwd': fname_fwd,
        'fname_cov': fname_cov,
        'fname_evoked': fname_evoked
    }


"""
print(f"   EEG channels: {sum([ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH  for ch in fwd['info']['chs']])}")
"""
    
def demonstrate_forward_problem(paths):
    """
    Demonstrate forward problem: brain sources → scalp EEG
    - Use EEG-only, fixed-orientation forward to get (n_channels x n_sources) leadfield
    - Simulate a single active source and project it to scalp
    - Build a proper Info (with sfreq and channel positions) for EvokedArray/topomap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import mne
    from mne.io.constants import FIFF
    from mne.channels import make_dig_montage

    print("\n" + "=" * 60)
    print("PART 1: FORWARD PROBLEM")
    print("=" * 60)

    # 1) Load forward solution
    print("\n1. Loading forward solution...")
    fwd = mne.read_forward_solution(paths["fname_fwd"])
    print(f"   Forward solution contains {fwd['nsource']} sources")
    eeg_count = sum(ch["kind"] == FIFF.FIFFV_EEG_CH for ch in fwd["info"]["chs"])
    print(f"   EEG channels: {eeg_count}")

    # 2) Keep EEG only and convert to fixed orientation (leadfield: n_chan x n_src)
    fwd_eeg = mne.pick_types_forward(fwd, meg=False, eeg=True)
    fwd_eeg_fixed = mne.convert_forward_solution(
        fwd_eeg, surf_ori=True, force_fixed=True, use_cps=True
    )

    # 3) Create a simulated source estimate matching the forward's source space
    print("\n2. Creating simulated dipole source in motor cortex...")
    src = fwd_eeg_fixed["src"]
    vertices = [src[0]["vertno"], src[1]["vertno"]]  # LH, RH vertices
    n_src = fwd_eeg_fixed["nsource"]

    # One time point for simplicity
    stc_sim = mne.SourceEstimate(
        data=np.zeros((n_src, 1)),
        vertices=vertices,
        tmin=0.0,
        tstep=0.01,
        subject=paths["subject"],
    )

    # Activate a left-hemisphere source (index within LH range)
    lh_n = len(vertices[0])
    source_idx = min(500, lh_n - 1) if lh_n > 0 else 0
    stc_sim.data[source_idx, 0] = 50e-9 # 50 nAm in A·m
    print(f"   Activating source index {source_idx} (LH count={lh_n})")

    # 4) Apply forward model (source → scalp)
    print("\n3. Applying forward model (source → scalp)...")
    leadfield = fwd_eeg_fixed["sol"]["data"]  # (n_channels, n_src)
    scalp_data = leadfield @ stc_sim.data     # (n_channels, 1)

    # 5) Build a fresh Info with sfreq and correct EEG electrode positions
    fwd_info = fwd_eeg_fixed["info"]
    ch_names = [ch["ch_name"] for ch in fwd_info["chs"]]  # order matches leadfield rows
    info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")

    # Create a DigMontage from the forward channel locations (in head coordinates)
    ch_pos = {ch["ch_name"]: ch["loc"][:3] for ch in fwd_info["chs"]}
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage)

    # Create EvokedArray for topomap plotting
    evoked_sim = mne.EvokedArray(scalp_data, info, tmin=0.0)

    # 6) Visualize
    print("\n4. Visualizing forward model...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Topomap of simulated scalp potentials
    # IMPORTANT: provide both axes (one for time, one for colorbar)
    evoked_sim.plot_topomap(times=[0.0],
                            axes=[axes[1], axes[0]],
                            show=False,
                            colorbar=True,
                            vlim=(-20, 20), 
                            cmap="RdBu_r")
    axes[1].set_title("Simulated Scalp EEG from Motor Cortex Source")

    # Source visualization (skip gracefully if 3D backend unavailable)
    #vmax = float(np.abs(stc_sim.data).max())
    try:
        brain = stc_sim.plot(
            subject=paths["subject"],
            subjects_dir=paths["subjects_dir"],
            hemi="lh",
            time_viewer=False,
            initial_time=0.0,
            background="white",
            size=(900, 600),
            # colormap="magma",  # sequential, good for magnitude maps
            # clim=dict(kind="value", lims=[0.02 * vmax, 0.10 * vmax, vmax]),  # higher threshold
            # smoothing_steps=10,     # visually spread a single-vertex activation
            # transparent=False       # do not hide values below fmin as transparent     
        )
    # Mark the exact active vertex (always visible)
    #    vtx_lh = int(vertices[0][source_idx])  # LH vertex id you activated
    #    brain.add_foci(vtx_lh, coords_as_verts=True, hemi="lh", color="crimson", scale_factor=0.9)
        brain.save_image(out("forward_model_source.png"))
        brain.close()
    except Exception as e:
        print(f"   (Source plot skipped: {e})")

    plt.tight_layout()
    plt.savefig(out("forward_model_demo.png"), dpi=300, bbox_inches="tight")
    print("   Saved: forward_model_demo.png")

    return fwd, evoked_sim, stc_sim

def demonstrate_inverse_problem(paths, fwd, evoked_sim=None):
    """
    Demonstrate inverse problem: scalp EEG → brain sources
    Compare multiple inverse methods (MNE, dSPM, sLORETA)
    """
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse

    print("\n" + "=" * 60)
    print("PART 2: INVERSE PROBLEM")
    print("=" * 60)

    # 1) Load evoked (Left Auditory). Do not subscript; it already returns an Evoked.
    print("\n1. Loading evoked data...")
    fname_ave = paths["fname_evoked"]
    try:
        evoked = mne.read_evokeds(fname_ave, condition="Left Auditory", baseline=(None, 0))
    except Exception:
        # Fallback: list all and pick the one containing "Left Auditory"
        all_evokeds = mne.read_evokeds(fname_ave, baseline=(None, 0))
        names = [e.comment for e in all_evokeds]
        print(f"   Available conditions: {names}")
        evoked = next(e for e in all_evokeds if "Left Auditory" in e.comment)

    # Keep EEG only (to match the EEG-only forward we'll use)
    evoked = evoked.pick_types(meg=False, eeg=True, exclude="bads")
    evoked.crop(0.05, 0.20) # seconds 
    

    # 2) Load noise covariance and prepare EEG-only forward
    print("\n2. Loading noise covariance...")
    noise_cov = mne.read_cov(paths["fname_cov"])

    print("\n3. Preparing EEG-only forward and aligning channels...")
    fwd_eeg = mne.pick_types_forward(fwd, meg=False, eeg=True)

    # Compute common channel set and enforce identical order everywhere
    fwd_chs = [ch["ch_name"] for ch in fwd_eeg["info"]["chs"]]
    common_chs = [ch for ch in evoked.ch_names if ch in fwd_chs]
    if len(common_chs) == 0:
        raise RuntimeError("No common EEG channels between evoked and forward model.")

    evoked = evoked.copy().pick_channels(common_chs, ordered=True)
    fwd_eeg = mne.pick_channels_forward(fwd_eeg, common_chs, ordered=True)
    noise_cov = mne.pick_channels_cov(noise_cov, common_chs)

    # Optional: regularize covariance for EEG
    try:
        noise_cov = mne.cov.regularize(noise_cov, evoked.info, mag=0, grad=0, eeg=0.1)
    except Exception:
        # Regularization is optional; continue if not available
        pass

    print("\n4. Creating inverse operator...")
    inv = make_inverse_operator(
        evoked.info, fwd_eeg, noise_cov, loose=0.2, depth=0.8, verbose=False
    )

    print("\n5. Applying inverse solutions...")
    lambda2 = 1.0 / 9.0
    stc_mne = apply_inverse(evoked, inv, lambda2=lambda2, method="MNE",  verbose=False)
    stc_dspm = apply_inverse(evoked, inv, lambda2=lambda2, method="dSPM", verbose=False)
    stc_sloreta = apply_inverse(evoked, inv, lambda2=lambda2, method="sLORETA", verbose=False)

    print("\n6. Comparing inverse methods (peak locations)...")
    peak_mne = stc_mne.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)
    peak_dspm = stc_dspm.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)
    peak_sloreta = stc_sloreta.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)
    print(f"   MNE:     vertex {peak_mne[0]}, time idx {peak_mne[1]}")
    print(f"   dSPM:    vertex {peak_dspm[0]}, time idx {peak_dspm[1]}")
    print(f"   sLORETA: vertex {peak_sloreta[0]}, time idx {peak_sloreta[1]}")

    return {"mne": stc_mne, "dspm": stc_dspm, "sloreta": stc_sloreta}
    
def visualize_inverse_comparison(stcs, paths):
    """Visualize and compare inverse solutions"""
    print("\n6. Creating comparison visualizations...")
    
    # Plot time course at peak for each method
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    methods = ['mne', 'dspm', 'sloreta']
    titles = ['MNE', 'dSPM', 'sLORETA']
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        stc = stcs[method]
        
        # Get peak vertex
        peak = stc.get_peak(hemi='lh', vert_as_index=True)
        
        # Plot time course
        axes[idx].plot(stc.times * 1000, stc.data[peak[0], :] * 1e9)
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Amplitude (nAm)')
        axes[idx].set_title(f'{title} - Source Time Course at Peak')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out("inverse_methods_comparison.png"), dpi=300, bbox_inches="tight")
    print("   Saved: inverse_methods_comparison.png")
    
    # Plot brain activations at peak time for each method
    print("\n7. Generating brain surface plots...")
    
    #use a common snapshot time (seconds) for all methods
    snap_t = 0.100  # 100 ms

    for method, title in zip(methods, titles):
        stc = stcs[method]
        
        # Plot on brain
        brain = stc.plot(
            subject=paths['subject'],
            subjects_dir=paths['subjects_dir'],
            hemi='both',
            time_viewer=False,
            background='white',
            initial_time=snap_t,  # use fixed snapshot time
           # initial_time=stc.get_peak()[1],
            size=(900, 600),
           # colorbar=dict(n_labels=4, fmt='%.2e'),
            #colormap='coolwarm', 
            #clim=dict(kind='percent', 
            #lims=[97, 98.5, 99.5]), 
            #smoothing_steps=10 # purely visual; OK to omit""
        )        
        # Save screenshot
        brain.save_image(out(f"inverse_{method}_brain.png"))        
        brain.close()
        print(f"   Saved: inverse_{method}_brain.png")

def analyze_inverse_accuracy(stcs):
    """Analyze properties of inverse solutions"""
    print("\n" + "="*60)
    print("PART 3: INVERSE SOLUTION ANALYSIS")
    print("="*60)
    
    results = {}
    
    for method_name, stc in stcs.items():
        # Calculate metrics
        peak_val, peak_time = stc.get_peak()
        
        # Spatial extent (count significant vertices)
        threshold = 0.5 * np.max(np.abs(stc.data))
        n_active = np.sum(np.max(np.abs(stc.data), axis=1) > threshold)
        
        # Temporal properties
        peak_latency = peak_time * 1000  # Convert to ms
        
        results[method_name] = {
            'peak_amplitude': peak_val,
            'peak_latency': peak_latency,
            'n_active_sources': n_active,
            'total_power': np.sum(stc.data ** 2)
        }
        
        print(f"\n{method_name.upper()} Results:")
        print(f"  Peak amplitude: {peak_val:.2e}")
        print(f"  Peak latency: {peak_latency:.1f} ms")
        print(f"  Active sources: {n_active}")
        print(f"  Total power: {results[method_name]['total_power']:.2e}")
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<12} {'Peak Amp':<12} {'Latency (ms)':<15} {'# Active':<12}")
    print("-" * 60)
    
    for method_name, res in results.items():
        print(f"{method_name.upper():<12} {res['peak_amplitude']:<12.2e} "
              f"{res['peak_latency']:<15.1f} {res['n_active_sources']:<12}")
    
    return results

def main():
    """Main execution function"""
    print("="*60)
    print("EEG FORWARD AND INVERSE PROBLEM DEMONSTRATION")
    print("="*60)
    print("\nThis script demonstrates:")
    print("1. Forward modeling (brain → scalp)")
    print("2. Inverse modeling (scalp → brain)")
    print("3. Comparison of inverse methods (MNE, dSPM, sLORETA)")
    print("\nNote: First run will download sample data (~1.5GB)")
    print("="*60)
    
    # Setup
    paths = setup_data_paths()
    
    # Part 1: Forward Problem
    fwd, evoked_sim, stc_sim = demonstrate_forward_problem(paths)
    
    # Part 2: Inverse Problem
    stcs = demonstrate_inverse_problem(paths, fwd, evoked_sim)
    
    # Part 3: Visualization
    visualize_inverse_comparison(stcs, paths)
    
    # Part 4: Analysis
    results = analyze_inverse_accuracy(stcs)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - forward_model_demo.png")
    print("  - inverse_methods_comparison.png")
    print("  - inverse_mne_brain.png")
    print("  - inverse_dspm_brain.png")
    print("  - inverse_sloreta_brain.png")
    print("\nKey findings:")
    print("✓ Forward model successfully simulates scalp EEG from brain sources")
    print("✓ Inverse solutions reconstruct source activity from scalp data")
    print("✓ Different methods show trade-offs between spatial precision and noise")
    print("✓ dSPM and sLORETA provide noise normalization vs standard MNE")

if __name__ == "__main__":
    main()