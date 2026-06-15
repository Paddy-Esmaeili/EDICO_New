import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from mesh_to_ply import parse_mesh

# -------------------------
# Load data
# -------------------------
mesh_file = "Test_output_gradient.mesh"
car_file = "Test_corners_car(1).txt"

mesh_points, _, _, voltages, _ = parse_mesh(mesh_file)

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

# -------------------------
# Distance computation
# -------------------------
tree = cKDTree(car_points)
distances, _ = tree.query(mesh_points)

threshold = 20  # mm

# -------------------------
# Mask
# -------------------------
trusted = distances <= threshold

# -------------------------
# 3D plot
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot trusted region (green)
ax.scatter(
    mesh_points[trusted, 0],
    mesh_points[trusted, 1],
    mesh_points[trusted, 2],
    c='green',
    s=0.2
)

# plot rejected region (red)
ax.scatter(
    mesh_points[~trusted, 0],
    mesh_points[~trusted, 1],
    mesh_points[~trusted, 2],
    c='red',
    s=0.2
)

# catheter points (blue)
ax.scatter(
    car_points[:, 0],
    car_points[:, 1],
    car_points[:, 2],
    c='blue',
    s=50
)

ax.set_title("Fill Threshold Coverage Map (20 mm)")
plt.savefig("figure.png", dpi=300)