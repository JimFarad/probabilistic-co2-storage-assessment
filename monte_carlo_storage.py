"""
Monte Carlo Simulation for CO2 Storage Capacity Assessment
==========================================================

Author: Dimitris Farantouris
Date: 07 June 2026

Overview
--------
This script performs a Monte Carlo simulation to quantify uncertainty 
in total CO2 storage capacity for a multi-closure reservoir system.

The model propagates geological uncertainty in key petrophysical 
parameters through a probabilistic framework, generating a distribution 
of possible storage outcomes.

In addition to total storage uncertainty, the workflow analyses the 
relative contribution of individual closures and identifies the dominant 
storage structure within the system.

Methodology
-----------
A deterministic storage case (from Permedia) is used as a baseline.  
Uncertainty is introduced by sampling:

    • Porosity (ϕ)
    • CO2 saturation (S_CO2)

Each simulation scales the deterministic closure volumes according to 
these sampled parameters and converts the results into stored CO2 mass.

Monte Carlo sampling is performed using clipped normal distributions
to enforce physically realistic parameter bounds.

Key Assumptions
---------------
• Structural geometry (closure volumes, thickness, trap extent) is fixed  
• Only petrophysical properties (porosity, saturation) vary  
• Porosity and saturation follow truncated normal distributions  
• CO2 density is constant across all realizations  
• Closure behaviour is independent (no interaction between traps)  

Outputs
-------
The workflow produces:

• Probabilistic storage estimates:
    - P10 (optimistic/high case)
    - P50 (median case)
    - P90 (conservative/low case)

• Total storage statistics:
    - Mean
    - Standard deviation

• Largest closure analysis:
    - Identification of dominant closure
    - P10 / P50 / P90 values
    - Contribution to total storage (%)

• Closure-level results:
    - Median storage per closure
    - Percentage contribution to total storage

• Visual outputs:
    - Histogram of total storage distribution
    - Closure contribution bar chart

• Exported files:
    - CSV datasets (total & closure-level results)
    - Summary text file
    - Metadata describing simulation inputs
    - Figures (PNG always, optional PGF/PDF)

• LaTeX integration:
    - Optional PGF figure export (native LaTeX rendering)
    - Auto-generated LaTeX figure snippets

Technical Features
------------------
• Robust backend handling:
    - Automatically detects LaTeX availability
    - Falls back safely to PNG-only mode if unavailable

• Multi-format figure export:
    - PNG (always)
    - PGF + PDF (if LaTeX backend enabled)

• Reproducibility:
    - Fixed random seed
    - Time-stamped output directory
    - Complete metadata logging

Applications
------------
This analysis supports:

• CO2 storage screening and ranking  
• Early-phase Carbon Capture and Storage (CCS) feasibility studies  
• Quantification of uncertainty in subsurface storage capacity  
• Identification of dominant storage structures  
• Risk-informed decision-making under geological uncertainty  

Notes
-----
Results depend strongly on assumed parameter distributions and bounds.  
Care should be taken to ensure these inputs reflect:

• Available geological data  
• Reservoir characterization studies  
• Expert interpretation and calibration  

The largest closure often dominates total storage capacity and should 
be carefully evaluated in both technical and economic assessments.
"""

# ============================================================
# IMPORT LIBRARIES & Functions
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_figure(fig, name, output_directory, pgf_enabled):

    import matplotlib as mpl

    # Always save PNG:
    # → Ensures compatibility, quick viewing, and fallback output
    png_path = f"{output_directory}/{name}.png"
    fig.savefig(png_path, dpi=300)

    # Detect active matplotlib backend (critical for PGF safety)
    backend = mpl.get_backend().lower()

    # Only attempt PGF export if:
    # → PGF is explicitly enabled AND backend is actually 'pgf'
    if pgf_enabled and backend.startswith("pgf"):
        try:
            # Save LaTeX-native figure (vector + LaTeX fonts)
            fig.savefig(f"{output_directory}/{name}.pgf")

            # Save PDF backup (high-quality portable format)
            fig.savefig(f"{output_directory}/{name}.pdf")

            print(f"✓ Saved {name} as PGF/PDF")

        except Exception as e:
            # Fail-safe: never crash due to LaTeX issues
            print(f"! PGF export failed for {name}, skipping.")
            print(f"→ Reason: {e}")

    else:
        # Inform user that only PNG was generated
        print(f"\n→ PGF skipped for {name} (backend={backend})")

# ============================================================
# 1. INITIAL SETUP: OUTPUT DIRECTORY & PLOTTING BACKEND
# ============================================================
#
# This section:
# • Creates a unique output directory for reproducibility
# • Initializes matplotlib backend (PGF if LaTeX available)
# • Ensures robust fallback if LaTeX is not installed

# ------------------------------------------------------------
# 1a. CREATE UNIQUE OUTPUT DIRECTORY
# ------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Each run gets a unique folder → ensures reproducibility & no overwrites
output_directory = f"mc_results/run_{timestamp}"

os.makedirs(output_directory, exist_ok=True)

print(f"\n✓ Results will be saved in: {output_directory}")


# ------------------------------------------------------------
# 1b. MATPLOTLIB BACKEND SETUP (PGF WITH SAFE FALLBACK)
# ------------------------------------------------------------

import matplotlib as mpl
import subprocess

# Flag controlling whether LaTeX/PGF rendering is active
pgf_enabled = False

try:
    # Check if LaTeX engine (pdflatex) is available
    subprocess.run(
        ["pdflatex", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    # Activate PGF backend (must occur BEFORE pyplot import)
    mpl.use("pgf")

    pgf_enabled = True
    print("✓ PGF backend enabled (LaTeX detected)")

except Exception:
    # Fallback: use non-interactive backend (safe for scripts / servers)
    mpl.use("Agg")
    pgf_enabled = False

    print("\n! No LaTeX → using Agg backend (PNG only)\n")


# Import pyplot AFTER backend is set
import matplotlib.pyplot as plt

# Configure matplotlib style based on backend
if pgf_enabled:
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",  # Use pdflatex explicitly
        "text.usetex": True,          # Enable LaTeX text rendering
        "font.family": "serif",
        "font.size": 11,
    })
else:
    mpl.rcParams.update({
        "text.usetex": False,         # Disable LaTeX completely
        "font.family": "serif",
        "font.size": 11,
    })


# ============================================================
# 2. INPUT DATA (FROM PERMEDIA)
# ============================================================

# Fluid volumes per closure [m3]
# Already includes geometry + base petrophysical parameters
closure_volumes_m3 = np.array([
    34987288,
    6711115,
    4377690.5,
    2877019.75,
    1545083,
    1121914,
    4087630.75,
    1386234.25,
    2754063.75,
    691994.375
])

number_of_closures = len(closure_volumes_m3)

# ============================================================
# 3. BASE PARAMETERS (REFERENCE CASE)
# ============================================================

base_porosity = 0.20
base_saturation = 0.74
co2_density_kg_per_m3 = 630

kg_to_megaton = 1e-9

# ============================================================
# 4. MONTE CARLO SETTINGS
# ============================================================

number_of_simulations = 10000

# Uncertainty distributions
porosity_mean, porosity_std = 0.20, 0.03
saturation_mean, saturation_std = 0.74, 0.05

# Physical limits
porosity_min, porosity_max = 0.15, 0.30
saturation_min, saturation_max = 0.50, 0.90

np.random.seed(42)

# ============================================================
# 5. GENERATE RANDOM VARIABLES
# ============================================================

porosity_samples = np.clip(
    np.random.normal(porosity_mean, porosity_std, number_of_simulations),
    porosity_min, porosity_max
)

saturation_samples = np.clip(
    np.random.normal(saturation_mean, saturation_std, number_of_simulations),
    saturation_min, saturation_max
)

# ============================================================
# 6. RUN MONTE CARLO SIMULATION
# ============================================================

total_storage_results_mt = []
closure_storage_results_mt = np.zeros((number_of_simulations, number_of_closures))

for simulation_index in range(number_of_simulations):

    # Scaling factor relative to deterministic case
    scaling_factor = (
        porosity_samples[simulation_index] / base_porosity *
        saturation_samples[simulation_index] / base_saturation
    )

    # Apply scaling
    scaled_volumes = closure_volumes_m3 * scaling_factor

    # Convert to mass (Mt)
    closure_storage_mt = scaled_volumes * co2_density_kg_per_m3 * kg_to_megaton

    closure_storage_results_mt[simulation_index, :] = closure_storage_mt

    total_storage_mt = np.sum(closure_storage_mt)
    total_storage_results_mt.append(total_storage_mt)

total_storage_results_mt = np.array(total_storage_results_mt)

# ============================================================
# 7. TOTAL STATISTICS
# ============================================================

P10 = np.percentile(total_storage_results_mt, 90)
P50 = np.percentile(total_storage_results_mt, 50)
P90 = np.percentile(total_storage_results_mt, 10)

mean_storage = np.mean(total_storage_results_mt)
std_storage = np.std(total_storage_results_mt)

# ============================================================
# 8. CLOSURE ANALYSIS
# ============================================================

closure_median_mt = np.percentile(closure_storage_results_mt, 50, axis=0)

closure_contribution_percent = (
    closure_median_mt / np.sum(closure_median_mt) * 100
)

# ============================================================
# 9. LARGEST CLOSURE ANALYSIS
# ============================================================

# Largest closure based on deterministic input volume
largest_closure_index = np.argmax(closure_volumes_m3)
largest_closure_id = largest_closure_index + 1
largest_closure_volume_m3 = closure_volumes_m3[largest_closure_index]

# Monte Carlo storage results for the largest closure
largest_closure_results_mt = closure_storage_results_mt[:, largest_closure_index]

# Probabilistic statistics for the largest closure
largest_closure_P10 = np.percentile(largest_closure_results_mt, 90)
largest_closure_P50 = np.percentile(largest_closure_results_mt, 50)
largest_closure_P90 = np.percentile(largest_closure_results_mt, 10)

largest_closure_mean = np.mean(largest_closure_results_mt)
largest_closure_std = np.std(largest_closure_results_mt)

largest_closure_contribution_percent = closure_contribution_percent[largest_closure_index]

# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n=== TOTAL STORAGE RESULTS ===")
print(f"P90 (Conservative): {P90:.2f} Mt")
print(f"P50 (Median):       {P50:.2f} Mt")
print(f"P10 (Optimistic):   {P10:.2f} Mt")
print(f"Mean:               {mean_storage:.2f} Mt")
print(f"Std Dev:            {std_storage:.2f} Mt")

print("\n=== LARGEST CLOSURE RESULTS ===")
print(f"Largest Closure:    Closure {largest_closure_id}")
print(f"Input Volume:       {largest_closure_volume_m3:.2f} m3")
print(f"P90 (Conservative): {largest_closure_P90:.2f} Mt")
print(f"P50 (Median):       {largest_closure_P50:.2f} Mt")
print(f"P10 (Optimistic):   {largest_closure_P10:.2f} Mt")
print(f"Mean:               {largest_closure_mean:.2f} Mt")
print(f"Std Dev:            {largest_closure_std:.2f} Mt")
print(f"Contribution:       {largest_closure_contribution_percent:.1f}% of total P50 storage")

print("\n=== CLOSURE CONTRIBUTIONS ===")
for i in range(number_of_closures):
    print(f"Closure {i+1}: {closure_median_mt[i]:.2f} Mt "
          f"({closure_contribution_percent[i]:.1f}%)")

# ============================================================
# 11. SAVE OUTPUT FILES
# ============================================================

# ---- SUMMARY FILE ----
with open(f"{output_directory}/summary.txt", "w") as f:
    f.write("Monte Carlo CO2 Storage Results\n")
    f.write("================================\n\n")
    
    f.write("TOTAL STORAGE\n")
    f.write("----------------\n")
    f.write(f"P90: {P90:.2f} Mt\n")
    f.write(f"P50: {P50:.2f} Mt\n")
    f.write(f"P10: {P10:.2f} Mt\n")
    f.write(f"Mean: {mean_storage:.2f} Mt\n")
    f.write(f"Std Dev: {std_storage:.2f} Mt\n\n")

    f.write("LARGEST CLOSURE RESULTS\n")
    f.write("-----------------------\n")
    f.write(f"Largest Closure: Closure {largest_closure_id}\n")
    f.write(f"Input Volume: {largest_closure_volume_m3:.2f} m3\n")
    f.write(f"P90: {largest_closure_P90:.2f} Mt\n")
    f.write(f"P50: {largest_closure_P50:.2f} Mt\n")
    f.write(f"P10: {largest_closure_P10:.2f} Mt\n")
    f.write(f"Mean: {largest_closure_mean:.2f} Mt\n")
    f.write(f"Std Dev: {largest_closure_std:.2f} Mt\n")
    f.write(f"Contribution to total P50 storage: {largest_closure_contribution_percent:.1f}%\n\n")

    f.write("CLOSURE CONTRIBUTIONS (P50)\n")
    f.write("--------------------------\n")
    for i in range(number_of_closures):
        f.write(f"Closure {i+1}: {closure_median_mt[i]:.2f} Mt "
                f"({closure_contribution_percent[i]:.1f}%)\n")

# ---- RAW TOTAL RESULTS ----
np.savetxt(
    f"{output_directory}/total_results.csv",
    total_storage_results_mt,
    delimiter=",",
    header="Total_CO2_Mt",
    comments=""
)

# ---- RAW CLOSURE RESULTS ----
np.savetxt(
    f"{output_directory}/closure_results.csv",
    closure_storage_results_mt,
    delimiter=",",
    header="Closure1,Closure2,Closure3,Closure4,Closure5,Closure6,Closure7,Closure8,Closure9,Closure10",
    comments=""
)

# ---- LARGEST CLOSURE RAW RESULTS ----
np.savetxt(
    f"{output_directory}/largest_closure_results.csv",
    largest_closure_results_mt,
    delimiter=",",
    header=f"Closure{largest_closure_id}_CO2_Mt",
    comments=""
)

# ---- METADATA ----
with open(f"{output_directory}/run_info.txt", "w") as f:
    f.write("Monte Carlo Run Info\n")
    f.write("====================\n")
    f.write(f"Simulations: {number_of_simulations}\n")
    f.write(f"Porosity: mean={porosity_mean}, std={porosity_std}\n")
    f.write(f"Saturation: mean={saturation_mean}, std={saturation_std}\n")
    f.write(f"Porosity limits: {porosity_min}-{porosity_max}\n")
    f.write(f"Saturation limits: {saturation_min}-{saturation_max}\n")
    f.write(f"Largest closure ID: {largest_closure_id}\n")
    f.write(f"Largest closure input volume (m3): {largest_closure_volume_m3:.2f}\n")
    f.write(f"Largest closure P90 (Mt): {largest_closure_P90:.2f}\n")
    f.write(f"Largest closure P50 (Mt): {largest_closure_P50:.2f}\n")
    f.write(f"Largest closure P10 (Mt): {largest_closure_P10:.2f}\n")
    f.write(f"Largest closure mean (Mt): {largest_closure_mean:.2f}\n")
    f.write(f"Largest closure std dev (Mt): {largest_closure_std:.2f}\n")
    f.write(f"Largest closure contribution (%): {largest_closure_contribution_percent:.1f}\n")

# ============================================================
# 12. PLOT TOTAL STORAGE DISTRIBUTION (MONTE CARLO)
# ============================================================

# Create figure for histogram of total storage results
plt.figure(figsize=(10,6))

# Plot histogram of Monte Carlo total storage outcomes
# → bins define resolution of distribution
plt.hist(total_storage_results_mt, bins=50, color='skyblue', edgecolor='black')

# Plot percentile indicators (risk bounds)
# → P90 = conservative (low storage)
# → P50 = median
# → P10 = optimistic (high storage)
plt.axvline(P10, color='red', linestyle='--', label='P10')
plt.axvline(P50, color='green', linestyle='--', label='P50')
plt.axvline(P90, color='blue', linestyle='--', label='P90')

# Axis labels and title
plt.xlabel("CO2 Storage Capacity (Mt)")
plt.ylabel("Frequency")
plt.title("Monte Carlo CO2 Storage Capacity Distribution")

# Add legend and grid for readability
plt.legend()
plt.grid()

# Optimize spacing (prevents label overlap)
plt.tight_layout()

# Save figure using robust export function:
# → PNG always saved
# → PGF/PDF only if LaTeX backend is active
save_figure(plt.gcf(), "histogram", output_directory, pgf_enabled)

# Close figure to free memory (important for multiple plots)
plt.close()

# ============================================================
# 13. PLOT CLOSURE CONTRIBUTION
# ============================================================

# Create a new figure window for the bar chart
plt.figure(figsize=(10,6))

# Generate closure IDs (1 → N) for plotting on x-axis
closure_ids = np.arange(1, number_of_closures + 1)

# Plot contribution (%) of each closure (based on P50 case)
plt.bar(closure_ids, closure_contribution_percent)

# Axis labeling
plt.xlabel("Closure ID")
plt.ylabel("Contribution (%)")

# Title describes that this is based on the median (P50) scenario
plt.title("Closure Contribution to Total Storage (P50)")

# Add grid for readability
plt.grid()

# Adjust layout to avoid clipping of labels/titles
plt.tight_layout()

# Save figure using robust save function:
# → Always saves PNG
# → Saves PGF/PDF only if LaTeX backend is active
save_figure(plt.gcf(), "closure_contribution", output_directory, pgf_enabled)

# Close figure to free memory (important in batch workflows)
plt.close()

# ============================================================
# 14. EXPORT LATEX FIGURE SNIPPETS
# ============================================================

# Only generate LaTeX figure snippets if PGF backend is active
# (i.e., LaTeX rendering is available and .pgf files exist)
if pgf_enabled:

    # Define relative path to figures from LaTeX document
    # This allows portability across systems and servers
    # (IMPORTANT: adjust to match your LaTeX project structure)
    latex_relative_path = "figures/" + output_directory

    # Output file containing ready-to-use LaTeX figure environments
    latex_file_path = f"{output_directory}/latex_figures.txt"

    with open(latex_file_path, "w") as f:

        # Header explaining purpose of this file
        f.write("% =====================================================\n")
        f.write("% LaTeX Figure Snippets (PGF - Native Rendering)\n")
        f.write("% Paths are RELATIVE (adjust if needed)\n")
        f.write("% =====================================================\n\n")

        # --------------------------------------------------------
        # Histogram Figure
        # --------------------------------------------------------

        # This block creates a complete LaTeX figure environment
        # using PGF input (native LaTeX rendering, not an image)
        f.write("% --- Monte Carlo Histogram ---\n")
        f.write("\\begin{figure}[h!]\n")
        f.write("\\centering\n")

        # \input loads the PGF figure directly (vector + LaTeX fonts)
        f.write(f"\\input{{{latex_relative_path}/histogram.pgf}}\n")

        # Caption describes the probabilistic distribution
        f.write("\\caption{Monte Carlo distribution of total CO$_2$ storage capacity.}\n")

        # Label allows referencing in LaTeX text (\\ref{...})
        f.write("\\label{fig:mc_histogram}\n")
        f.write("\\end{figure}\n\n")

        # --------------------------------------------------------
        # Closure Contribution Figure
        # --------------------------------------------------------

        f.write("% --- Closure Contribution ---\n")
        f.write("\\begin{figure}[h!]\n")
        f.write("\\centering\n")

        # Load closure contribution figure (PGF format)
        f.write(f"\\input{{{latex_relative_path}/closure_contribution.pgf}}\n")

        # Caption explains that values correspond to P50 case
        f.write("\\caption{Closure contribution to total storage capacity (P50 case).}\n")

        f.write("\\label{fig:closure_contribution}\n")
        f.write("\\end{figure}\n\n")

    # Confirmation message for user
    print("\n✓ LaTeX figure snippets exported")

else:
    # If PGF not available → no LaTeX-native figures can be created
    print("\n→ Skipping LaTeX snippet export (PGF not enabled)")


# ============================================================
# DONE
# ============================================================

print("\n✓ All results saved successfully!")
