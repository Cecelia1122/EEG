"""
Peripheral Nerve Stimulation Simulation
Bonus component for Neural Engineering Portfolio

This script simulates:
1. Single nerve fiber action potentials (Hodgkin-Huxley model)
2. Compound action potentials (multiple fiber recruitment)
3. Stimulus-response relationships
4. Nerve conduction velocity measurements

Author: Xue Li
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# Auto-create a results directory next to this script
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
def out(name: str) -> str:
    return str(RESULTS_DIR / name)

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# PART 1: Single Fiber - Hodgkin-Huxley Model
# ============================================================================

class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley model of nerve action potential
    
    The classic model of nerve excitability, showing how voltage-gated
    sodium and potassium channels generate action potentials.
    """
    
    def __init__(self):
        # Membrane capacitance (μF/cm²)
        self.C_m = 1.0
        
        # Maximum conductances (mS/cm²)
        self.g_Na = 120.0  # Sodium
        self.g_K = 36.0    # Potassium
        self.g_L = 0.3     # Leak
        
        # Reversal potentials (mV)
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.4
        
        # Temperature factor
        self.T = 6.3  # 6.3°C
    
    def alpha_m(self, V):
        """Sodium activation rate"""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Sodium deactivation rate"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Sodium inactivation rate"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Sodium de-inactivation rate"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """Potassium activation rate"""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """Potassium deactivation rate"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def I_Na(self, V, m, h):
        """Sodium current"""
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def I_K(self, V, n):
        """Potassium current"""
        return self.g_K * n**4 * (V - self.E_K)
    
    def I_L(self, V):
        """Leak current"""
        return self.g_L * (V - self.E_L)
    
    def I_inj(self, t, stimulus_start, stimulus_duration, stimulus_amplitude):
        """Injected current (stimulus)"""
        if stimulus_start <= t <= stimulus_start + stimulus_duration:
            return stimulus_amplitude
        return 0.0
    
    def derivatives(self, y, t, stimulus_start, stimulus_duration, stimulus_amplitude):
        """
        Calculate derivatives for ODE solver
        y = [V, m, h, n]
        """
        V, m, h, n = y

        # Get scalar external current at this time t
        I_ext = self.I_inj(t, stimulus_start, stimulus_duration, stimulus_amplitude)
        
        # Gating variable derivatives
        dm_dt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dh_dt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dn_dt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        # Membrane potential derivative
        I_total = I_ext - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)
        dV_dt = I_total / self.C_m
        
        return [dV_dt, dm_dt, dh_dt, dn_dt]
    
    def simulate(self, t_max=50, dt=0.01, stimulus_start=10, 
                 stimulus_duration=0.5, stimulus_amplitude=10):
        """
        Simulate action potential
        
        Parameters:
        -----------
        t_max : float
            Simulation duration (ms)
        dt : float
            Time step (ms)
        stimulus_start : float
            When to start stimulus (ms)
        stimulus_duration : float
            Stimulus duration (ms)
        stimulus_amplitude : float
            Stimulus current (μA/cm²)
        """
        # Time array
        t = np.arange(0, t_max, dt)
        
        # Initial conditions: resting state
        V_rest = -65.0  # mV
        m_rest = self.alpha_m(V_rest) / (self.alpha_m(V_rest) + self.beta_m(V_rest))
        h_rest = self.alpha_h(V_rest) / (self.alpha_h(V_rest) + self.beta_h(V_rest))
        n_rest = self.alpha_n(V_rest) / (self.alpha_n(V_rest) + self.beta_n(V_rest))
        y0 = [V_rest, m_rest, h_rest, n_rest]
        
        # Solve ODE (pass stimulus parameters, not a whole array)
        solution = odeint(
            self.derivatives, y0, t,
            args=(stimulus_start, stimulus_duration, stimulus_amplitude)
        )
        
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        
        # Recompute I_ext over time for plotting/return
        I_ext = np.array([
            self.I_inj(ti, stimulus_start, stimulus_duration, stimulus_amplitude) for ti in t
        ])
        
        # Calculate ionic currents
        I_Na_arr = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K_arr = self.g_K * n**4 * (V - self.E_K)
        I_L_arr = self.g_L * (V - self.E_L)
        
        return {
            't': t,
            'V': V,
            'm': m,
            'h': h,
            'n': n,
            'I_Na': I_Na_arr,
            'I_K': I_K_arr,
            'I_L': I_L_arr,
            'I_ext': I_ext
        }


# ============================================================================
# PART 2: Compound Action Potential (Multiple Fibers)
# ============================================================================

class CompoundActionPotential:
    """
    Simulate compound action potential from multiple nerve fibers
    Models fiber recruitment and conduction velocity
    """
    """
    Parameters
    ----------
    n_fibers : int
        Number of fibers in the bundle.
    velocity_scale : float
        Linear scale for CV vs diameter (m/s per μm). Motor nerves typically ~3.0–4.5.
    seed : int | None
        Random seed for reproducibility.
    diameter_min : float
        Lower clip (μm) for fiber diameters to keep within a motor-like range.
    diameter_max : float
        Upper clip (μm) for fiber diameters to keep within a motor-like range.
    """
    def __init__(self, n_fibers=100, velocity_scale=3.5, seed=None,
                 diameter_min=5.0, diameter_max=20.0):
        self.n_fibers = n_fibers
        rng = np.random.default_rng(seed)

        # Fiber diameter distribution (μm), clipped to motor-like range
        diam = rng.gamma(shape=3, scale=3, size=n_fibers) + 2
        self.diameters = np.clip(diam, diameter_min, diameter_max)

        # Conduction velocity proportional to diameter (m/s)
        self.velocities = velocity_scale * self.diameters

        # Activation thresholds (lower for larger fibers)
        self.thresholds = 10.0 / self.diameters  # Inversely proportional
        
    def single_fiber_ap(self, t, velocity, distance=50):
        """
        Simulate single fiber action potential at recording site
        
        Parameters:
        -----------
        t : array
            Time array (ms)
        velocity : float
            Conduction velocity (m/s)
        distance : float
            Distance from stimulus to recording (mm)
        """
        # Convert units: distance in mm, velocity in m/s, time in ms
        delay = distance / (velocity * 1000 / 1000)  # ms
        
        # Action potential shape (simplified)
        # Use double exponential for realistic AP shape
        t_shifted = t - delay
        
        # Only show AP after it arrives
        ap = np.zeros_like(t)
        mask = t_shifted > 0
        
        # Depolarization phase (fast)
        rise = 100 * t_shifted[mask] * np.exp(-t_shifted[mask] / 0.5)
        
        # Repolarization phase (slower)
        fall = -30 * t_shifted[mask] * np.exp(-t_shifted[mask] / 2.0)
        
        ap[mask] = rise + fall
        
        return ap
    
    def simulate_recruitment(self, stimulus_amplitudes, distance=50):
        """
        Simulate recruitment curve: how many fibers activate at each stimulus level
        
        Parameters:
        -----------
        stimulus_amplitudes : array
            Range of stimulus intensities
        distance : float
            Distance from stimulus to recording (mm)
        """
        t = np.arange(0, 20, 0.01)  # Time array (ms)
        
        caps = []
        n_recruited = []
        
        for stim_amp in stimulus_amplitudes:
            # Determine which fibers are recruited
            recruited = stim_amp > self.thresholds
            n_recruited.append(np.sum(recruited))
            
            # Sum contributions from all recruited fibers
            cap = np.zeros_like(t)
            for i, is_recruited in enumerate(recruited):
                if is_recruited:
                    cap += self.single_fiber_ap(t, self.velocities[i], distance)
            
            caps.append(cap)
        
        return {
            't': t,
            'caps': np.array(caps),
            'stimulus_amplitudes': stimulus_amplitudes,
            'n_recruited': np.array(n_recruited)
        }


# ============================================================================
# PART 3: Nerve Conduction Study Simulation
# ============================================================================

# --- Use onset or negative-peak latency in simulate_nerve_conduction_study() ---

def simulate_nerve_conduction_study():
    """
    Simulate a clinical nerve conduction study
    Measure conduction velocity by stimulating at two points
    """
    print("="*60)
    print("NERVE CONDUCTION STUDY SIMULATION")
    print("="*60)

    # Use motor-like velocity scale by default
    cap_sim = CompoundActionPotential(n_fibers=50, velocity_scale=3.5, seed=42)

    # Stimulation at two points
    distances = [40, 80]  # mm from recording electrode
    stimulus_amp = 5.0    # Supramaximal stimulus

    t = np.arange(0, 20, 0.01)

    results = {}

    for distance in distances:
        recruited = stimulus_amp > cap_sim.thresholds
        cap = np.zeros_like(t)
        for i, is_recruited in enumerate(recruited):
            if is_recruited:
                cap += cap_sim.single_fiber_ap(t, cap_sim.velocities[i], distance)
        results[distance] = cap

    # Helper: onset latency at 10% of negative peak magnitude
    def onset_latency(cap, t, frac=0.1):
        idx_min = np.argmin(cap)
        neg_peak = -cap[idx_min]
        if neg_peak <= 0:
            return t[idx_min]
        thresh = -frac * neg_peak
        # find first crossing toward negativity before the negative peak
        for i in range(idx_min):
            if cap[i] <= thresh:
                return t[i]
        return t[idx_min]

    # Compute both peak-negative and onset latencies
    latency_40_peak = t[np.argmin(results[40])]
    latency_80_peak = t[np.argmin(results[80])]
    latency_40_onset = onset_latency(results[40], t, frac=0.1)
    latency_80_onset = onset_latency(results[80], t, frac=0.1)

    # Conduction velocities
    distance_diff = (80 - 40) / 1000  # m
    time_diff_peak = (latency_80_peak - latency_40_peak) / 1000  # s
    time_diff_onset = (latency_80_onset - latency_40_onset) / 1000  # s
    cv_peak = distance_diff / time_diff_peak if time_diff_peak > 0 else np.nan
    cv_onset = distance_diff / time_diff_onset if time_diff_onset > 0 else np.nan

    print(f"\nNerve Conduction Study Results:")
    print(f"  Proximal stimulation (40mm):")
    print(f"    Peak-negative latency: {latency_40_peak:.2f} ms, Onset latency: {latency_40_onset:.2f} ms")
    print(f"  Distal stimulation (80mm):")
    print(f"    Peak-negative latency: {latency_80_peak:.2f} ms, Onset latency: {latency_80_onset:.2f} ms")
    print(f"  Latency difference (peak): {latency_80_peak - latency_40_peak:.2f} ms")
    print(f"  Latency difference (onset): {latency_80_onset - latency_40_onset:.2f} ms")
    print(f"  Conduction velocity (peak): {cv_peak:.1f} m/s")
    print(f"  Conduction velocity (onset): {cv_onset:.1f} m/s")
    print(f"  (Motor nerves often ~40–60 m/s; fastest myelinated can exceed ~80–100 m/s)")

    return {
        't': t,
        'cap_40mm': results[40],
        'cap_80mm': results[80],
        'conduction_velocity_peak': cv_peak,
        'conduction_velocity_onset': cv_onset,
        'latency_40_peak': latency_40_peak,
        'latency_80_peak': latency_80_peak,
        'latency_40_onset': latency_40_onset,
        'latency_80_onset': latency_80_onset
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_single_fiber():
    """Visualize single fiber action potential"""
    print("\n" + "="*60)
    print("PART 1: SINGLE FIBER ACTION POTENTIAL")
    print("="*60)
    
    print("\nSimulating Hodgkin-Huxley model...")
    neuron = HodgkinHuxleyNeuron()
    result = neuron.simulate(stimulus_amplitude=15, stimulus_duration=3.0)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Membrane potential
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(result['t'], result['V'], 'b-', linewidth=2)
    ax1.axhline(y=-65, color='gray', linestyle='--', alpha=0.5, label='Resting potential')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero mV')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Action Potential', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gating variables
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(result['t'], result['m'], label='m (Na⁺ activation)', linewidth=2)
    ax2.plot(result['t'], result['h'], label='h (Na⁺ inactivation)', linewidth=2)
    ax2.plot(result['t'], result['n'], label='n (K⁺ activation)', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Gating Variable')
    ax2.set_title('Ion Channel Gating', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ionic currents
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(result['t'], result['I_Na'], label='I_Na', linewidth=2)
    ax3.plot(result['t'], result['I_K'], label='I_K', linewidth=2)
    ax3.plot(result['t'], result['I_L'], label='I_Leak', linewidth=2)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Current (μA/cm²)')
    ax3.set_title('Ionic Currents', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 4: Phase plane (V vs m)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(result['V'], result['m'], linewidth=2)
    ax4.set_xlabel('Membrane Potential (mV)')
    ax4.set_ylabel('Na⁺ Activation (m)')
    ax4.set_title('Phase Plane: V-m', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Stimulus
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(result['t'], result['I_ext'], 'r-', linewidth=2)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Stimulus Current (μA/cm²)')
    ax5.set_title('External Stimulus', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-1, 15])
    
    plt.tight_layout()
    plt.savefig(out('single_fiber_action_potential.png'), dpi=300, bbox_inches='tight')
    print("  Saved: single_fiber_action_potential.png")
    
    # Print key measurements
    v_peak = np.max(result['V'])
    v_rest = result['V'][0]
    ap_amplitude = v_peak - v_rest
    
    # Compute dV/dt
    dV_dt = np.diff(result['V']) / np.diff(result['t'])

    # Find stimulus onset index for a robust search window
    stim_on_idx = np.where(result['I_ext'] > 0)[0][0]
    search_start = max(stim_on_idx - 1, 0)
    # Threshold = V when dV/dt crosses a small fraction of its peak during the upstroke
    dV_dt_seg = dV_dt[search_start:]
    peak_dv = np.max(dV_dt_seg)
    frac = 0.1  # 10% of peak slope as threshold criterion
    cross_idx_rel = np.argmax(dV_dt_seg >= frac * peak_dv)
    threshold_idx = search_start + cross_idx_rel
    v_threshold = result['V'][threshold_idx]
    
    # AP duration at half-maximum
    half_max = v_rest + ap_amplitude / 2
    above_half = result['V'] > half_max
    ap_duration = np.sum(above_half) * (result['t'][1] - result['t'][0])

    print(f"\n  Action Potential Properties:")
    print(f"    Resting potential: {v_rest:.1f} mV")
    print(f"    Threshold: {v_threshold:.1f} mV")
    print(f"    Peak amplitude: {v_peak:.1f} mV")
    print(f"    AP amplitude: {ap_amplitude:.1f} mV")
    print(f"    Duration (at half-max): {ap_duration:.2f} ms")


def visualize_compound_ap():
    """Visualize compound action potential and recruitment"""
    print("\n" + "="*60)
    print("PART 2: COMPOUND ACTION POTENTIAL")
    print("="*60)
    
    print("\nSimulating fiber recruitment...")
    cap_sim = CompoundActionPotential(n_fibers=100, velocity_scale=3.5, seed=42)
    
    # Test different stimulus intensities
    stimulus_range = np.linspace(0.1, 10, 20)
    result = cap_sim.simulate_recruitment(stimulus_range, distance=50)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Fiber diameter distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(cap_sim.diameters, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Fiber Diameter (μm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Nerve Fiber Size Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Conduction velocity vs diameter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(cap_sim.diameters, cap_sim.velocities, alpha=0.6, s=50)
    ax2.set_xlabel('Fiber Diameter (μm)')
    ax2.set_ylabel('Conduction Velocity (m/s)')
    ax2.set_title('Diameter-Velocity Relationship', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CAPs at different stimulus intensities
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show CAPs for select stimulus levels
    indices = [5, 10, 15, 19]
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for idx, color in zip(indices, colors):
        ax3.plot(result['t'], result['caps'][idx], 
                label=f'{stimulus_range[idx]:.1f} mA', 
                linewidth=2, color=color)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude (arbitrary units)')
    ax3.set_title('Compound Action Potentials', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recruitment curve
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot number of recruited fibers
    ax4_twin = ax4.twinx()
    
    # Peak amplitude of CAP
    peak_amplitudes = np.array([np.max(np.abs(cap)) for cap in result['caps']])
    
    line1 = ax4.plot(stimulus_range, result['n_recruited'], 'b-o', 
                     linewidth=2, markersize=6, label='Fibers recruited')
    ax4.set_xlabel('Stimulus Intensity (mA)')
    ax4.set_ylabel('Number of Fibers', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    line2 = ax4_twin.plot(stimulus_range, peak_amplitudes, 'r-s', 
                         linewidth=2, markersize=6, label='CAP amplitude')
    ax4_twin.set_ylabel('CAP Amplitude (a.u.)', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title('Recruitment Curve', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out('compound_action_potential.png'), dpi=300, bbox_inches='tight')
    print("  Saved: compound_action_potential.png")
    
    print(f"\n  Nerve Bundle Properties:")
    print(f"    Total fibers: {cap_sim.n_fibers}")
    print(f"    Diameter range: {cap_sim.diameters.min():.1f} - {cap_sim.diameters.max():.1f} μm")
    print(f"    Velocity range: {cap_sim.velocities.min():.1f} - {cap_sim.velocities.max():.1f} m/s")
    print(f"    Threshold range: {cap_sim.thresholds.min():.2f} - {cap_sim.thresholds.max():.2f} mA")


def visualize_conduction_study(method: str = "onset"):
    """Visualize nerve conduction study.

    Parameters
    ----------
    method : {"onset", "peak"}, optional
        Which latency to use for plotting and velocity in the title.
        - "onset": uses 10% negative-peak onset latency
        - "peak": uses peak-negative latency
        Default is "onset" (closer to clinical practice).
    """
    print("\n" + "="*60)
    print("PART 3: NERVE CONDUCTION STUDY")
    print("="*60)

    result = simulate_nerve_conduction_study()

    # Extract data
    t = result["t"]
    cap_40 = result["cap_40mm"]
    cap_80 = result["cap_80mm"]

    # Choose latency type with graceful fallback to legacy keys
    method = method.lower()
    if method not in ("onset", "peak"):
        method = "onset"

    if method == "onset":
        lat40 = result.get("latency_40_onset", result.get("latency_40"))
        lat80 = result.get("latency_80_onset", result.get("latency_80"))
        cv = result.get("conduction_velocity_onset", result.get("conduction_velocity"))
        method_label = "Onset"
    else:  # "peak"
        lat40 = result.get("latency_40_peak", result.get("latency_40"))
        lat80 = result.get("latency_80_peak", result.get("latency_80"))
        cv = result.get("conduction_velocity_peak", result.get("conduction_velocity"))
        method_label = "Peak-negative"

    if lat40 is None or lat80 is None or cv is None:
        raise KeyError(
            "Latency or conduction velocity keys not found in result. "
            "Check simulate_nerve_conduction_study() return dict."
        )

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Overlaid CAPs
    ax1 = axes[0]
    ax1.plot(t, cap_40, 'b-', linewidth=2, label='Proximal (40mm)')
    ax1.plot(t, cap_80, 'r-', linewidth=2, label='Distal (80mm)')
    ax1.axvline(x=lat40, color='b', linestyle='--', alpha=0.5)
    ax1.axvline(x=lat80, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_title('Nerve Conduction Study: CAPs at Two Stimulation Sites',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency-distance relation
    ax2 = axes[1]
    distances = [40, 80]
    latencies = [lat40, lat80]
    ax2.plot(distances, latencies, 'ko-', linewidth=2, markersize=10)

    # Regression line (through the two points)
    slope = (latencies[1] - latencies[0]) / (distances[1] - distances[0])  # ms/mm
    intercept = latencies[0] - slope * distances[0]
    x_fit = np.array([30, 90])
    y_fit = slope * x_fit + intercept
    ax2.plot(x_fit, y_fit, 'k--', alpha=0.5, label=f'Slope = {slope:.3f} ms/mm')

    ax2.set_xlabel('Distance from Recording Site (mm)')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title(f'Conduction Velocity ({method_label}) = {cv:.1f} m/s',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([30, 90])

    plt.tight_layout()
    plt.savefig(out('nerve_conduction_study.png'), dpi=300, bbox_inches='tight')
    print("  Saved: nerve_conduction_study.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PERIPHERAL NERVE STIMULATION SIMULATION".center(58) + "║")
    print("║" + "  Bonus Component - Neural Engineering Portfolio".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")
    
    print("This simulation demonstrates:")
    print("1. Single fiber action potentials (Hodgkin-Huxley model)")
    print("2. Compound action potentials from fiber recruitment")
    print("3. Clinical nerve conduction velocity measurement")
    print("="*60)
    
    # Part 1: Single fiber
    visualize_single_fiber()
    
    # Part 2: Compound AP
    visualize_compound_ap()
    
    # Part 3: Conduction study
    visualize_conduction_study()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - single_fiber_action_potential.png")
    print("  - compound_action_potential.png")
    print("  - nerve_conduction_study.png")
    
    print("\nKey Concepts Demonstrated:")
    print("  ✓ Hodgkin-Huxley equations for nerve excitability")
    print("  ✓ Ion channel dynamics (Na⁺, K⁺ gating)")
    print("  ✓ Compound action potential formation")
    print("  ✓ Fiber recruitment and size principle")
    print("  ✓ Conduction velocity measurement")
    print("  ✓ Clinical neurophysiology applications")
    
    print("\nRelevance to Neural Engineering:")
    print("  • Understanding nerve stimulation for neuroprosthetics")
    print("  • Functional electrical stimulation (FES) design")
    print("  • Diagnostic neurophysiology")
    print("  • Nerve interface design for BCIs")
    print("="*60)


if __name__ == "__main__":
    main()