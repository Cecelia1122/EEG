"""
Quick Setup Script for EEG Motor Imagery Pipeline
This script helps you get started by:
1. Checking Python version
2. Creating necessary directories
3. (Optionally) downloading MNE sample data (~1.5 GB)
4. Verifying installations

Author: Xue Li
Date: October 2025
"""

import sys
import os
import argparse
from pathlib import Path

BANNER = """
╔==========================================================╗
║                                                          ║
║         EEG Motor Imagery Pipeline - Setup Script        ║
║                       Author: Xue Li                     ║
║                                                          ║
╚==========================================================╝
"""

def check_python_version() -> bool:
    print("="*60)
    print("CHECKING PYTHON VERSION")
    print("="*60)

    v = sys.version_info
    print(f"\nPython version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required!")
        return False
    print("✅ Python version is adequate")
    return True

def create_directories() -> bool:
    print("\n" + "="*60)
    print("CREATING DIRECTORIES")
    print("="*60)

    for d in ["data", "results", "docs", "notebooks"]:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {d}/")
        else:
            print(f"✓  Exists: {d}/")
    return True

def check_dependencies() -> bool:
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)

    required = {
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "mne": "mne",
        "seaborn": "seaborn",
    }
    missing = []
    for import_name, pkg in required.items():
        try:
            __import__(import_name)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - NOT FOUND")
            missing.append(pkg)

    if missing:
        print("\n⚠️  Missing packages detected!")
        print("Install with conda (recommended):")
        print("  conda install -c conda-forge " + " ".join(missing))
        print("\nOr pip:")
        print("  pip install " + " ".join(missing))
        return False

    print("\n✅ All required packages are installed!")
    return True

def _sample_present_at(path: Path) -> bool:
    """
    Return True if the extracted MNE sample dataset appears to exist under path.
    Expected subfolder: MNE-sample-data/MEG/sample
    """
    expected = path / "MNE-sample-data" / "MEG" / "sample"
    return expected.exists()

def download_sample_data(sample_mode: str = "auto", mne_data_dir: str | None = None) -> bool:
    """
    sample_mode: 'auto' (default), 'yes', or 'no'
      auto: download only if missing
      yes: force download
      no: skip download
    mne_data_dir: target directory for dataset (default: ~/mne_data or MNE_DATA env)
    """
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE DATA")
    print("="*60)

    try:
        import mne
        from mne.datasets import sample
    except Exception as e:
        print(f"\n❌ MNE not available, cannot manage sample data: {e}")
        return False

    # Resolve target directory
    target_dir = Path(
        mne_data_dir or os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    ).expanduser()
    os.environ["MNE_DATA"] = str(target_dir)  # Let MNE use this location
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset directory: {target_dir}")
    present = _sample_present_at(target_dir)

    if sample_mode == "no":
        print("⏭️  Skipping sample data download (--sample no)")
        if not present:
            print("ℹ️  Sample dataset not found yet. Parts 1 and 3 will attempt to download when run.")
        return True

    if present and sample_mode in ("auto",):
        print("✅ Sample dataset already present. Reusing existing files.")
        return True

    if present and sample_mode == "yes":
        print("↻ Forcing re-download (--sample yes).")

    print("\nThis download is ~1.5 GB and may take a while.")
    print("If your connection is unstable, consider manual/resumable download (instructions will follow on error).")

    try:
        # Trigger MNE download (uses MNE_DATA path)
        # 'path' sets the base directory; download=True forces checks/download
        data_path = sample.data_path(path=str(target_dir), download=True, verbose=True)
        print(f"\n✅ Sample data ready at: {data_path}")
        return True
    except KeyboardInterrupt:
        print("\n⛔ Download interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading data: {e}")
        # Provide resumable manual download instructions
        tar_name = "MNE-sample-data-processed.tar.gz"
        print("\nYou can download manually with resume support:")
        print("1) Using curl (resumable):")
        print(f'   curl -L -C - -o "{target_dir}/{tar_name}" "https://osf.io/86qa2/download?version=6"')
        print("2) Then extract:")
        print(f'   tar -xzf "{target_dir}/{tar_name}" -C "{target_dir}"')
        print("3) Re-run setup or scripts. Ensure the folder:")
        print(f"   {target_dir}/MNE-sample-data/MEG/sample  exists.")
        return False

def create_test_script() -> bool:
    print("\n" + "="*60)
    print("CREATING TEST SCRIPT")
    print("="*60)

    test_code = '''"""
Quick test to verify installation
"""

import numpy as np
import matplotlib.pyplot as plt
import mne

print("Testing basic functionality...")

# Test 1: NumPy
print("\\n1. NumPy:")
x = np.array([1, 2, 3, 4, 5])
print(f"   Mean: {np.mean(x)}")

# Test 2: Matplotlib
print("\\n2. Matplotlib:")
plt.figure(figsize=(6, 4))
plt.plot(x, x**2)
plt.title("Test Plot")
plt.savefig("results/test_plot.png")
print("   Saved: results/test_plot.png")
plt.close()

# Test 3: MNE
print("\\n3. MNE:")
print(f"   MNE version: {mne.__version__}")

print("\\n✅ All tests passed! Your environment is ready.")
print("\\nNext steps:")
print("  1. Run: python part1_forward_inverse.py")
print("  2. Run: python part2_motor_imagery.py")
print("  3. Run: python part3_source_space.py")
'''
    Path("results").mkdir(exist_ok=True, parents=True)
    Path("test_installation.py").write_text(test_code, encoding="utf-8")
    print("✅ Created: test_installation.py")
    print("\nRun it with: python test_installation.py")
    return True

def print_next_steps():
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)

    print("\n📚 Next Steps:")
    print("\n1. Test your installation:")
    print("   python test_installation.py")
    print("\n2. Run the pipeline:")
    print("   python part1_forward_reverse.py    # Forward/inverse problems")
    print("   python part2_motor_imagery.py      # Motor imagery classification")
    print("   python part3_source_space.py       # Source-space analysis")
    print("\n3. Explore the code:")
    print("   - Each script is well-documented")
    print("   - Read README.md for detailed information")
    print("   - Check docs/ for technical report template")
    print("\n4. View results:")
    print("   - Generated images will be in results/")
    print("   - Review visualizations and metrics")
    print("\n📧 Questions? Contact: xueli.xl1122@gmail.com")
    print("="*60)

def parse_args():
    p = argparse.ArgumentParser(description="EEG Motor Imagery Pipeline - Setup")
    p.add_argument(
        "--sample",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Manage MNE sample dataset download (auto: only if missing; yes: force; no: skip)",
    )
    p.add_argument(
        "--mne-data",
        default=None,
        help="Custom directory for MNE sample dataset (default: $MNE_DATA or ~/mne_data)",
    )
    return p.parse_args()

def main() -> bool:
    print(BANNER)

    args = parse_args()

    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Checking dependencies", check_dependencies),
    ]
    for name, fn in steps:
        if not fn():
            print(f"\n❌ Setup failed at: {name}")
            return False

    print("\n" + "="*60)
    print("OPTIONAL STEPS")
    print("="*60)

    # Download/skip sample data according to flags
    download_sample_data(sample_mode=args.sample, mne_data_dir=args.mne_data)

    create_test_script()
    print_next_steps()
    return True

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)