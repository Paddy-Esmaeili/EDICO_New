"""
CARTO3 .mesh parser — surfaces + bipolar voltage coloring
Outputs a proper triangle mesh PLY with per-vertex colors.

Usage:
    python parse_mesh.py input.mesh output.ply [input.car]

Dependencies:
    pip install numpy open3d scipy

"""

import sys
from matplotlib import colors
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INPUT_FILE  = "PMMA_saline.mesh"
CAR_FILE    = "PMMA_saline_car.txt"  
OUTPUT_FILE = "PMMA_saline.ply"
""
# Distance threshold: mesh vertices further than this from any
# catheter contact point are considered unmapped → colored gray.

UNMAPPED_THRESHOLD_MM = 0.0

# CARTO3 bipolar voltage color scale (mV → RGB)
CARTO3_SCALE = [
    (0.50, (1.00, 0.00, 0.00)),  # Red
    (0.70, (1.00, 0.50, 0.00)),  # Orange
    (0.90, (1.00, 1.00, 0.00)),  # Yellow
    (1.10, (0.00, 1.00, 0.00)),  # Green
    (1.30, (0.00, 1.00, 1.00)),  # Cyan
    (1.40, (0.00, 0.00, 1.00)),  # Blue
    (1.50, (1.00, 0.00, 1.00)),  # Purple/Magenta
]
SCAR_THRESHOLD  = 0.50   # mV — below this = scar 
HEALTHY_VOLTAGE = 1.50   # mV — at or above = healthy 

# Which color column to use (within VerticesColorsSection)
# 0=Unipolar, 1=Bipolar, 2=LAT, 3=Impedance …
COLOR_COLUMN = 1   # Bipolar

INVALID_VALUE = -10000.0  


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
GRAY = np.array([0.6, 0.6, 0.6])  # unmapped only

def carto3_color(mv: float) -> np.ndarray:
    # Scar threshold: red, NOT gray
    # Gray is reserved for unmapped (handled outside this function)
    if mv <= 0.50:
        return np.array([1.0, 0.0, 0.0])  # red = scar/low voltage
    if mv >= 1.50:
        return np.array(CARTO3_SCALE[-1][1])  # purple = healthy
    for i in range(len(CARTO3_SCALE) - 1):
        v0, c0 = CARTO3_SCALE[i]
        v1, c1 = CARTO3_SCALE[i+1]
        if v0 <= mv <= v1:
            t = (mv - v0) / (v1 - v0)
            return np.array(c0)*(1-t) + np.array(c1)*t
    return np.array(CARTO3_SCALE[-1][1])


# ─────────────────────────────────────────────
# CAR file parser
# ─────────────────────────────────────────────
def parse_car(path: str):
    """
    Parse a CARTO3 .car file and return a list of catheter contact points.
    Each entry: {'xyz': np.array([x,y,z]), 'bipolar_mv': float, 'valid': bool}
    Bipolar voltage is at column index 19 (0-based after the leading 'P').
    """
    points = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not (line.startswith("P\t") or line.startswith("P ")):
                continue
            parts = line.split()
            if len(parts) < 20:
                continue
            try:
                x        = float(parts[4])
                y        = float(parts[5])
                z        = float(parts[6])
                bipolar  = float(parts[19])
                valid    = abs(bipolar - INVALID_VALUE) > 1.0 and bipolar > -100.0
                points.append({
                    "xyz":        np.array([x, y, z]),
                    "bipolar_mv": bipolar if valid else None,
                    "valid":      valid,
                })
            except (ValueError, IndexError):
                continue
    return points


def compute_unmapped_mask(mesh_vertices: np.ndarray,
                          car_points: list,
                          threshold_mm: float = 10.0):
    """
    For every mesh vertex compute distance to the nearest catheter contact
    point.  Vertices farther than threshold_mm are flagged as unmapped.

    Returns
    -------
    unmapped_mask : bool array, shape (N,)   True = unmapped
    distances     : float array, shape (N,)  distance to nearest contact
    """
    print("CAR points loaded:", len(car_points))

    car_xyz = np.array([p["xyz"] for p in car_points])

    print("CAR centroid:", car_xyz.mean(axis=0))
    print("Mesh centroid:", mesh_vertices.mean(axis=0))

    valid_pts = [p for p in car_points if p["valid"]]
    if not valid_pts:
        print("      WARNING: no valid CAR points found — "
              "distance masking skipped.")
        return np.zeros(len(mesh_vertices), dtype=bool), \
               np.zeros(len(mesh_vertices), dtype=np.float64)

    car_xyz = np.array([p["xyz"] for p in valid_pts])
    tree    = cKDTree(car_xyz)
    distances, _ = tree.query(mesh_vertices, k=1)
    unmapped_mask = distances > threshold_mm
    return unmapped_mask, distances


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────
def parse_mesh(path: str):
    points    = []   # [x, y, z]
    normals   = []   # [nx, ny, nz]
    triangles = []   # [v0, v1, v2]
    voltages  = []   # one float per vertex
    valid     = []   # True if voltage data exists for this vertex

    section = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            # ── Skip comments and empty lines ──────────────────
            if not line or line.startswith(";"):
                continue

            # ── Section headers ────────────────────────────────
            if line.startswith("["):
                if   "[VerticesSection]"       in line: section = "vertices"
                elif "[TrianglesSection]"      in line: section = "triangles"
                elif "[VerticesColorsSection]" in line: section = "colors"
                elif "[VerticesAttributesSection]" in line: section = "attributes"
                else:                                    section = "other"
                continue

            if "=" not in line:
                continue

            rhs = line.split("=", 1)[1].split()

            # ── Vertices ───────────────────────────────────────
            if section == "vertices":
                if len(rhs) >= 3:
                    try:
                        xyz = list(map(float, rhs[:3]))
                        points.append(xyz)
                        if len(rhs) >= 6:
                            normals.append(list(map(float, rhs[3:6])))
                        else:
                            normals.append([0.0, 0.0, 0.0])
                    except ValueError:
                        pass

            # ── Triangles ──────────────────────────────────────
            elif section == "triangles":
                if len(rhs) >= 3:
                    try:
                        v0, v1, v2 = int(rhs[0]), int(rhs[1]), int(rhs[2])
                        # GroupID == -1000000 marks degenerate/null triangles
                        group = int(rhs[6]) if len(rhs) >= 7 else 0
                        if group != -1000000 and not (v0 == 0 and v1 == 0 and v2 == 0):
                            triangles.append([v0, v1, v2])
                    except (ValueError, IndexError):
                        pass

            # ── Voltage colors ─────────────────────────────────
            elif section == "colors":
                if len(rhs) > COLOR_COLUMN:
                    try:
                        val = float(rhs[COLOR_COLUMN])
                        if val == INVALID_VALUE:
                            voltages.append(0.0)
                            valid.append(False)
                        else:
                            voltages.append(val)
                            valid.append(True)
                    except ValueError:
                        voltages.append(0.0)
                        valid.append(False)
                else:
                    voltages.append(0.0)
                    valid.append(False)

    return (
        np.array(points,    dtype=np.float64),
        np.array(normals,   dtype=np.float64),
        np.array(triangles, dtype=np.int32),
        np.array(voltages,  dtype=np.float64),
        np.array(valid,     dtype=bool),
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    in_file  = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    out_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    car_file = sys.argv[3] if len(sys.argv) > 3 else CAR_FILE

    print(f"\n[1/4] Parsing  →  {in_file}")
    points, normals, triangles, voltages, valid = parse_mesh(in_file)

    n_pts  = len(points)
    n_tri  = len(triangles)
    n_volt = len(voltages)

    print(f"      Vertices  : {n_pts:,}")
    print(f"      Triangles : {n_tri:,}")
    print(f"      Voltages  : {n_volt:,}  ({valid.sum():,} valid)")

    if n_pts == 0:
        print("ERROR: no vertices parsed — check section headers in file.")
        sys.exit(1)

    # ── Bounding box ───────────────────────────────────────────
    lo, hi = points.min(axis=0), points.max(axis=0)
    dims   = hi - lo
    print(f"\n[2/4] Bounding box")
    print(f"      X: {lo[0]:.1f} → {hi[0]:.1f}  ({dims[0]:.1f} mm)")
    print(f"      Y: {lo[1]:.1f} → {hi[1]:.1f}  ({dims[1]:.1f} mm)")
    print(f"      Z: {lo[2]:.1f} → {hi[2]:.1f}  ({dims[2]:.1f} mm)")
    print(f"      Centroid: {points.mean(axis=0).round(1)}")

    # ── Voltage stats (valid only) ─────────────────────────────
    if valid.any():
        v_valid = voltages[valid]
        print(f"\n      Bipolar voltage (valid points)")
        print(f"      Min : {v_valid.min():.4f} mV")
        print(f"      Max : {v_valid.max():.4f} mV")
        print(f"      Mean: {v_valid.mean():.4f} mV")
        pct_scar = (v_valid < SCAR_THRESHOLD).mean() * 100
        print(f"      Scar (<{SCAR_THRESHOLD} mV): {pct_scar:.1f}%")

    # ── CAR-based unmapped mask ────────────────────────────────
    unmapped_mask = np.zeros(n_pts, dtype=bool)   # default: all mapped

    if car_file:
        print(f"\n      CAR file  →  {car_file}")
        car_points = parse_car(car_file)
        n_car_valid = sum(1 for p in car_points if p["valid"])
        print(f"      CAR points: {len(car_points):,}  ({n_car_valid:,} with valid voltage)")

        unmapped_mask, distances = compute_unmapped_mask(
            points, car_points, threshold_mm=UNMAPPED_THRESHOLD_MM
        )

        # Print coverage stats to help tune UNMAPPED_THRESHOLD_MM
        for t in [5.0, 8.0, 10.0, 12.0, 15.0]:
            pct = (distances < t).mean() * 100
            marker = " ◄ current" if t == UNMAPPED_THRESHOLD_MM else ""
            print(f"        {t:5.1f} mm threshold → {pct:5.1f}% of vertices mapped{marker}")

        pct_unmapped = unmapped_mask.mean() * 100
        print(f"      Unmapped (gray) at {UNMAPPED_THRESHOLD_MM} mm: "
              f"{unmapped_mask.sum():,} vertices ({pct_unmapped:.1f}%)")
    else:
        print(f"\n      No CAR file provided — skipping distance-based unmapping.")
        print(f"      (Pass a .car file as 3rd argument to enable this feature)")

    # ── Color mapping ──────────────────────────────────────────
    print(f"\n[3/4] Coloring vertices (CARTO3 bipolar scale)")
    colors = np.zeros((n_pts, 3), dtype=np.float64)
    GRAY   = np.array([0.6, 0.6, 0.6])

    # Pad voltage/valid arrays to match vertex count
    if n_volt < n_pts:
        voltages = np.pad(voltages, (0, n_pts - n_volt))
        valid    = np.pad(valid,    (0, n_pts - n_volt))
    elif n_volt > n_pts:
        voltages = voltages[:n_pts]
        valid    = valid[:n_pts]

    for i in range(n_pts):
        # Distance-based unmapped takes priority over everything
        if unmapped_mask[i]:
            colors[i] = GRAY
        elif valid[i]:
            colors[i] = carto3_color(voltages[i])
        else:
            colors[i] = GRAY  # no voltage data in mesh either

    # ── Build Open3D mesh ──────────────────────────────────────
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices       = o3d.utility.Vector3dVector(points)
    mesh.vertex_colors  = o3d.utility.Vector3dVector(colors)

    if len(normals) == n_pts:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    if n_tri > 0:
        # Clamp triangle indices to valid range
        tri_arr = np.array(triangles, dtype=np.int32)
        mask    = (tri_arr >= 0).all(axis=1) & (tri_arr < n_pts).all(axis=1)
        tri_arr = tri_arr[mask]
        print(f"      Valid triangles after index check: {len(tri_arr):,}")
        mesh.triangles = o3d.utility.Vector3iVector(tri_arr)

        # Compute normals if none were in the file
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
    else:
        # No triangles — fall back to point cloud
        print("      WARNING: No triangles found. Saving as point cloud.")
        pcd        = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        o3d.io.write_point_cloud(out_file, pcd, write_ascii=False)
        print(f"\n[4/4] Saved point cloud  →  {out_file}")
        return

    # ── Write PLY ──────────────────────────────────────────────
    print(f"\n[4/4] Writing  →  {out_file}")
    o3d.io.write_triangle_mesh(
        out_file, mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    print("Done")
    print("\nMeshLab tip: after opening, go to")
    print("  Render → Color → Per Vertex   (to show vertex colors)")
    print("  Render → Smooth / Flat shading (to control surface look)")
    v_valid = voltages[valid]

    print(f">1.5 mV : {(v_valid >= 1.5).mean()*100:.1f}%")
    print(f">2.0 mV : {(v_valid >= 2.0).mean()*100:.1f}%")
    print(f">2.5 mV : {(v_valid >= 2.5).mean()*100:.1f}%")
    print("invalid count =", (~valid).sum())

if __name__ == "__main__":
    main()