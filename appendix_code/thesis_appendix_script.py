"""
Monte Carlo Simulation for Probabilistic CO₂ Storage Capacity Assessment

This script propagates uncertainty in porosity and CO₂ saturation
through a Monte Carlo framework to estimate total storage capacity
for a multi-closure reservoir system.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 2. INPUT DATA
# ============================================================

# Deterministic closure volumes from reservoir assessment [m³]
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
# 3. REFERENCE PARAMETERS
# ============================================================

base_porosity = 0.20
base_saturation = 0.74

co2_density_kg_per_m3 = 630
kg_to_megaton = 1e-9

# ============================================================
# 4. MONTE CARLO SETTINGS
# ============================================================

number_of_simulations = 10000

# Porosity distribution
porosity_mean = 0.20
porosity_std = 0.03
porosity_min = 0.15
porosity_max = 0.30

# CO₂ saturation distribution
saturation_mean = 0.74
saturation_std = 0.05
saturation_min = 0.50
saturation_max = 0.90

np.random.seed(42)

# ============================================================
# 5. RANDOM SAMPLING
# ============================================================

porosity_samples = np.clip(
    np.random.normal(
        porosity_mean,
        porosity_std,
        number_of_simulations
    ),
    porosity_min,
    porosity_max
)

saturation_samples = np.clip(
    np.random.normal(
        saturation_mean,
        saturation_std,
        number_of_simulations
    ),
    saturation_min,
    saturation_max
)

# ============================================================
# 6. MONTE CARLO STORAGE CALCULATION
# ============================================================

def calculate_storage_capacity(porosity, saturation):
    """
    Scale deterministic closure volumes according to sampled
    porosity and saturation, then convert to stored CO₂ mass.
    """

    scaling_factor = (
        (porosity / base_porosity)
        * (saturation / base_saturation)
    )

    scaled_volumes = closure_volumes_m3 * scaling_factor

    closure_storage_mt = (
        scaled_volumes
        * co2_density_kg_per_m3
        * kg_to_megaton
    )

    return closure_storage_mt


total_storage_results_mt = np.zeros(number_of_simulations)
closure_storage_results_mt = np.zeros(
    (number_of_simulations, number_of_closures)
)

for i in range(number_of_simulations):

    closure_storage_mt = calculate_storage_capacity(
        porosity_samples[i],
        saturation_samples[i]
    )

    closure_storage_results_mt[i, :] = closure_storage_mt
    total_storage_results_mt[i] = np.sum(closure_storage_mt)

# ============================================================
# 7. PROBABILISTIC STATISTICS
# ============================================================

P10 = np.percentile(total_storage_results_mt, 90)
P50 = np.percentile(total_storage_results_mt, 50)
P90 = np.percentile(total_storage_results_mt, 10)

mean_storage = np.mean(total_storage_results_mt)
std_storage = np.std(total_storage_results_mt)

# ============================================================
# 8. CLOSURE-LEVEL ANALYSIS
# ============================================================

closure_median_mt = np.percentile(
    closure_storage_results_mt,
    50,
    axis=0
)

closure_contribution_percent = (
    closure_median_mt
    / np.sum(closure_median_mt)
    * 100
)

# Largest closure based on deterministic volume
largest_closure_index = np.argmax(closure_volumes_m3)

largest_closure_results_mt = (
    closure_storage_results_mt[:, largest_closure_index]
)

largest_closure_P10 = np.percentile(
    largest_closure_results_mt, 90
)

largest_closure_P50 = np.percentile(
    largest_closure_results_mt, 50
)

largest_closure_P90 = np.percentile(
    largest_closure_results_mt, 10
)

# ============================================================
# 9. RESULTS
# ============================================================

print("=== TOTAL STORAGE RESULTS ===")
print(f"P90:  {P90:.2f} Mt")
print(f"P50:  {P50:.2f} Mt")
print(f"P10:  {P10:.2f} Mt")
print(f"Mean: {mean_storage:.2f} Mt")
print(f"Std:  {std_storage:.2f} Mt")

print("\n=== LARGEST CLOSURE ===")
print(f"P90:  {largest_closure_P90:.2f} Mt")
print(f"P50:  {largest_closure_P50:.2f} Mt")
print(f"P10:  {largest_closure_P10:.2f} Mt")

# ============================================================
# 10. HISTOGRAM OF STORAGE CAPACITY
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    total_storage_results_mt,
    bins=50,
    color="skyblue",
    edgecolor="black"
)

plt.axvline(P10, color="red", linestyle="--", label="P10")
plt.axvline(P50, color="green", linestyle="--", label="P50")
plt.axvline(P90, color="blue", linestyle="--", label="P90")

plt.xlabel("CO₂ Storage Capacity (Mt)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Distribution of CO₂ Storage Capacity")

plt.legend()
plt.tight_layout()
plt.show()
