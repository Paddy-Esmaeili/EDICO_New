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

mesh_points, _, _, voltages, valid= parse_mesh(mesh_file)

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
# Distance computation
# -------------------------

tree = cKDTree(car_points)

distances, _ = tree.query(mesh_points)

# -------------------------
# Choose fill threshold
# -------------------------

threshold = 60

keep = distances <= threshold

# -------------------------
# Scar classification
# -------------------------

scar = (voltages < 0.5) & valid & keep
healthy = (voltages >= 0.5) & valid & keep

outside_fill = ~keep

inside_fill_invalid = keep & (~valid)

print()
print("Threshold:", threshold, "mm")
print("Retained:", np.sum(keep))
print("Scar points:", np.sum(scar))
print("Healthy points:", np.sum(healthy))
print("Invalid points:", np.sum(inside_fill_invalid))

# -------------------------
# 3D Visualization
# -------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Invalid region (outside fill threshold)

# outside fill threshold 

ax.scatter(
    mesh_points[outside_fill, 0],
    mesh_points[outside_fill, 1],
    mesh_points[outside_fill, 2],
    c="lightgray",
    s=50,
    alpha=0.2,
    label="Outside Fill Threshold"
)
# Inside fill threshold but invalid 

ax.scatter(
    mesh_points[inside_fill_invalid, 0],
    mesh_points[inside_fill_invalid, 1],
    mesh_points[inside_fill_invalid, 2],
    c="orange",
    s=50,
    alpha=0.2,
    label="Inside Fill Threshold but Invalid"
)


# Healthy tissue

ax.scatter(
    mesh_points[healthy, 0],
    mesh_points[healthy, 1],
    mesh_points[healthy, 2],
    c="green",
    s=50,
    label="Healthy (>= 0.5 mV)"
)

# Scar tissue

ax.scatter(
    mesh_points[scar, 0],
    mesh_points[scar, 1],
    mesh_points[scar, 2],
    c="red",
    s=50,
    label="Scar (< 0.5 mV)"
)

# Catheter points

ax.scatter(
    car_points[:, 0],
    car_points[:, 1],
    car_points[:, 2],
    c="blue",
    s=50,
    label="Catheter Points"
)

ax.legend()

ax.set_title(f"Scar Map ({threshold} mm Fill Threshold)")

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")

plt.savefig(
    f"scar_map_{threshold}mm.png",
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved scar_map_{threshold}mm.png")