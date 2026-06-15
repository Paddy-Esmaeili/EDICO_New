import numpy as np
from scipy.spatial import cKDTree

from mesh_to_ply import parse_mesh


mesh_file = "Test_output_gradient.mesh"
car_file = "Test_corners_car(1).txt"

# -------------------------
# Load mesh
# -------------------------

mesh_points, _, _, voltages, valid = parse_mesh(mesh_file)


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

print("CAR points:", len(car_points))     # Output how many car points were loaded

# -------------------------
# Nearest-neighbor search
# -------------------------

tree = cKDTree(car_points)

distances, _ = tree.query(mesh_points)

# -------------------------
# Threshold study
# -------------------------

thresholds = [5, 8, 11, 15, 20]      # Fill thresholds in mm to analyze 

print()
print("Threshold Analysis")
print("----------------------------")

for threshold in thresholds:

    keep = distances <= threshold

    retained = np.sum(keep)

    percent_retained = (
        100 * retained / len(mesh_points)
    )

    mean_voltage = np.mean(
        voltages[keep]
    )

    scar = voltages[keep] < 0.5    # CARTO mapping system defined a scar tissue as a tissue with 
                                   # a bipolar voltage of < 0.5 mV
    

    valid_retained = valid & keep

    n_valid = np.sum(valid_retained)

    if n_valid == 0:
        scar_fraction = np.nan
    else:
        scar_fraction = (
            100 * np.sum((voltages < 0.5) & valid_retained)
            / n_valid
        )

    print(
        f"{threshold:>2} mm | "
        f"{percent_retained:6.2f}% retained | "
        f"Mean V = {mean_voltage:.3f} mV | "
        f"Scar = {scar_fraction:.2f}%"
    )
print("Mesh points:", len(mesh_points))
print("CAR points:", len(car_points))
print("Voltage stats:")
print("  min:", np.min(voltages))
print("  max:", np.max(voltages))
print("  % invalid (-10000):", 100 * np.mean(voltages <= -9999), "%")
print("  % zero:", 100 * np.mean(voltages == 0), "%")
print("  % valid (>0 and not -10000):", 100 * np.mean((voltages > 0) & (voltages < 9999)), "%")

print("\nCAR point bounding box:")
print("  X:", car_points[:,0].min(), "to", car_points[:,0].max())
print("\nMesh point bounding box:")
print("  X:", mesh_points[:,0].min(), "to", mesh_points[:,0].max())