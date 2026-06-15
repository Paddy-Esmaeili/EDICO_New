import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from mesh_to_ply import parse_mesh

# -------------------------
# Files
# -------------------------

mesh_file = "Test_output_gradient.mesh"
car_file = "Test_corners_car(1).txt"

# -------------------------
# Load mesh
# -------------------------

mesh_points, _, _, voltages, _ = parse_mesh(mesh_file)

# -------------------------
# Load CAR points
# -------------------------

car_points = []

with open(car_file, "r") as f:

    for line in f:

        if line.startswith("P"):

            parts = line.split()

            x = float(parts[4])
            y = float(parts[5])
            z = float(parts[6])

            car_points.append([x, y, z])

car_points = np.array(car_points)

print("Mesh points:", len(mesh_points))
print("CAR points:", len(car_points))

# -------------------------
# Voltage statistics
# -------------------------

print("\nVoltage Statistics")
print("------------------")
print("Min:", np.min(voltages))
print("Max:", np.max(voltages))
print("Mean:", np.mean(voltages))

unique_voltages = np.unique(voltages)

print("Number of unique voltages:", len(unique_voltages))

if len(unique_voltages) <= 20:
    print("Unique voltages:", unique_voltages)
else:
    print("First 20 unique voltages:", unique_voltages[:20])

# -------------------------
# Distance computation
# -------------------------

tree = cKDTree(car_points)

distances, _ = tree.query(mesh_points)

max_dist = int(np.ceil(np.max(distances)))

print("\nMaximum mesh-to-CAR distance:", max_dist, "mm")

# -------------------------
# Threshold sweep
# -------------------------

thresholds = np.arange(0, max_dist + 1, 1)

retention_list = []
scar_list = []
mean_voltage_list = []
scar_count_list = []

for threshold in thresholds:

    keep = distances <= threshold

    retained = np.sum(keep)

    if retained == 0:

        retention_list.append(0)
        scar_list.append(np.nan)
        mean_voltage_list.append(np.nan)

        continue

    retention = (
        100 * retained / len(mesh_points)
    )

    scar = voltages[keep] < 0.5
    scar_count = np.sum(voltages[keep] < 0.5)

    scar_count_list.append(scar_count)
    scar_fraction = (
        100 * np.sum(scar) / retained
    )

    mean_voltage = np.mean(
        voltages[keep]
    )

    retention_list.append(retention)
    scar_list.append(scar_fraction)
    mean_voltage_list.append(mean_voltage)


# -------------------------
# Print 11 mm results
# -------------------------

if 11 <= max_dist:

    idx = np.where(thresholds == 11)[0][0]

    print("\n11 mm Results")
    print("------------------")
    print(f"Retention: {retention_list[idx]:.2f}%")
    print(f"Scar: {scar_list[idx]:.2f}%")
    print(f"Mean Voltage: {mean_voltage_list[idx]:.3f} mV")

# -------------------------
# Plot 1
# Retention vs Threshold
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    retention_list,
    linewidth=2
)

plt.axvline(
    x=11,
    linestyle="--",
    linewidth=1.5,
    label="11 mm"
)

plt.xlabel("Fill Threshold (mm)")
plt.ylabel("Retention (%)")
plt.title("Mesh Retention vs Fill Threshold")

plt.legend()

plt.grid(True)

plt.savefig(
    "retention_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Plot 2
# Scar vs Threshold
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    scar_list,
    linewidth=2
)

plt.axvline(
    x=11,
    linestyle="--",
    linewidth=1.5,
    label="11 mm"
)

plt.xlabel("Fill Threshold (mm)")
plt.ylabel("Scar Fraction (%)")
plt.title("Scar Fraction vs Fill Threshold")

plt.legend()

plt.grid(True)

plt.savefig(
    "scar_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Plot 3
# Mean Voltage vs Threshold
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    mean_voltage_list,
    linewidth=2
)

plt.axvline(
    x=11,
    linestyle="--",
    linewidth=1.5,
    label="11 mm"
)

plt.xlabel("Fill Threshold (mm)")
plt.ylabel("Mean Voltage (mV)")
plt.title("Mean Voltage vs Fill Threshold")

plt.legend()

plt.grid(True)

plt.savefig(
    "mean_voltage_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Plot 4
# Threshold vs Scar Count
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    scar_count_list,
    linewidth=2
)

plt.axvline(
    x=11,
    linestyle="--",
    linewidth=1.5,
    label="11 mm"
)

plt.xlabel("Fill Threshold (mm)")
plt.ylabel("Scar Count")
plt.title("Scar Count vs Fill Threshold")

plt.legend()

plt.grid(True)

plt.savefig(
    "scar_count_vs_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Summary
# -------------------------

print("\nSaved Figures")
print("------------------")
print("retention_vs_threshold.png")
print("scar_vs_threshold.png")
print("mean_voltage_vs_threshold.png")
print("scar_count_vs_threshold.png")