"""
Ram-air (double-skin, caisson) kites from a SurfPlan export.

A ram-air kite has no LE tube, no struts and no bridle: SurfPlan only exports the
`3d rib positions` table and open single-skin profiles, and the flying lines
attach straight onto the wing tips. `process_wing.main_process_wing` needs the
`LE tube` section such an export does not have, so the ribs are read here, with
the same rib sorting and SurfPlan -> VSM transform, so that `ribs_data` follows
the convention of an LEI kite:

  - the ribs, sorted tip to tip and moved into the VSM frame (x chord, y span to
    the left, z up), with the tip ribs that fold back inboard optionally dropped;
  - each profile closed for a closed-airfoil polar model (`regularize_profile`:
    the mono-skin sail behind the box is given a thin, tapering thickness);
  - the chord fraction where the box (double-skin caisson) closes;
  - the tow point of each tip: where the leading edge and the box closure, both
    extended outboard, meet;
  - optionally, the fabric mass and CG, from areal densities the caller supplies.

`main` writes `profiles/prof_N.dat`, `aero_geometry.yaml` and `ramair_wing.yaml`.
"""

import re
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline

from SurfplanAdapter.process_wing.main_process_wing import _sort_ribs_by_proximity
from SurfplanAdapter.utils import line_parser, transform_coordinate_system_surfplan_to_VSM


def read_ramair_ribs(surfplan_txt_file_path):
    """Ribs of the `3d rib positions` table, sorted tip to tip, in the VSM frame.

    Each rib holds `LE`, `TE`, `VUP`, `chord`, `rib_index` and `airfoil_id`; the
    profiles are numbered symmetrically around the centre rib, the way SurfPlan
    names the prof_*.dat files: N, ..., 2, 1, 2, ..., N.
    """
    ribs, section = [], None
    for raw_line in Path(surfplan_txt_file_path).read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("3d rib positions"):
            section = "ribs"
            continue
        if section != "ribs":
            continue
        if not line:
            break
        if not any(char.isdigit() for char in line) or line.isdigit():
            continue  # Column header and rib count
        values = line_parser(line)
        if len(values) == 9:
            ribs.append({
                "LE": np.array(values[0:3]),
                "TE": np.array(values[3:6]),
                "VUP": np.array(values[6:9]),
            })

    if not ribs:
        raise ValueError(f"No '3d rib positions' table found in {surfplan_txt_file_path}")

    n_ribs = len(ribs)
    k = n_ribs // 2
    for i, rib in enumerate(ribs):
        if n_ribs % 2 == 1:
            rib["airfoil_id"] = 1 + abs(i - k)
        else:
            rib["airfoil_id"] = k - i if i < k else i - k + 1
        rib["rib_index"] = i
        rib["chord"] = float(np.linalg.norm(rib["TE"] - rib["LE"]))

    ribs_data = _sort_ribs_by_proximity(ribs)
    for rib in ribs_data:
        for key in ("LE", "TE", "VUP"):
            rib[key] = transform_coordinate_system_surfplan_to_VSM(rib[key])
    return ribs_data


def trim_folded_tips(ribs_data):
    """Keep only the spanwise-monotonic run of ribs.

    On a strongly arched ram-air wing the SurfPlan tip ribs fold back inboard, so
    the outermost rib is not the last one. VSM's mesh refinement walks the
    section list in order, and that fold makes it emit a spurious side force on a
    geometry that is exactly symmetric; the folded ribs are dropped. They carry
    the actual wing tips, so the tow point is taken before this trimming.
    """
    spanwise_stations = [rib["LE"][1] for rib in ribs_data]
    i_max, i_min = int(np.argmax(spanwise_stations)), int(np.argmin(spanwise_stations))
    return ribs_data[min(i_max, i_min) : max(i_max, i_min) + 1]


def read_profile(dat_file_path):
    """(n, 2) coordinates of a .dat profile: every line holding exactly two numbers."""
    coordinates = []
    for line in Path(dat_file_path).read_text().splitlines():
        items = [s for s in re.split(r"[;,\s]", line) if s]
        if len(items) != 2:
            continue
        try:
            coordinates.append([float(items[0]), float(items[1])])
        except ValueError:
            continue
    if not coordinates:
        raise ValueError(f"Could not read any coordinates from {dat_file_path}")
    return np.array(coordinates)


def box_closure_station(dat_file_path):
    """(x/c, y/c) of the point where the closed cell gives way to the mono-skin sail.

    A ram-air profile is exported as a single contour: trailing edge -> rear
    sail -> nose upper skin -> leading edge -> nose lower skin -> closing panel
    -> rear sail -> trailing edge. Walking the lower surface from the leading
    edge aft, the closing panel is the one large upward step in y, so its top
    end -- the point sought here -- is the station where the box ends.
    """
    coordinates = read_profile(dat_file_path)
    i_le = int(np.argmin(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2))
    lower = coordinates[i_le:]  # LE -> TE

    steps = np.diff(lower[:, 1])
    i_jump = int(np.argmax(steps)) + 1
    if steps[i_jump - 1] < 0.02:  # No closing panel: not a ram-air section
        raise ValueError(
            f"{Path(dat_file_path).name} has no box-closing panel (largest upward step on "
            f"the lower surface is {steps[i_jump - 1]:.4f} c). A ram-air profile is a closed "
            "nose cell followed by a mono-skin rear sail."
        )
    return lower[i_jump]


def rib_point(rib, station):
    """Lift a (x/c, y/c) profile station onto a rib, in the VSM frame.

    SurfPlan's VUP is the rib's up direction carrying the chord's length
    (|VUP| = |TE - LE| and VUP . (TE - LE) = 0), so it is exactly the profile's
    y axis and no normalisation is needed.
    """
    x_over_c, y_over_c = station
    return rib["LE"] + x_over_c * (rib["TE"] - rib["LE"]) + y_over_c * rib["VUP"]


def tip_intersection(rib_outer, rib_inner, profiles_dir):
    """Intersect the extended leading edge with the extended box closure at one tip.

    Takes the two outermost ribs, draws the line through their leading edges and
    the line through their box-closure points, and intersects them inside the
    best-fit plane of those four points (coplanar only to within a residual,
    reported as `planarity`).
    """
    leading_edges = np.array([rib_outer["LE"], rib_inner["LE"]])
    closures = np.array([
        rib_point(rib, box_closure_station(Path(profiles_dir) / f"prof_{rib['airfoil_id']}.dat"))
        for rib in (rib_outer, rib_inner)
    ])

    points = np.vstack([leading_edges, closures])
    centroid = points.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(points - centroid)
    e1, e2 = right_vectors[0], right_vectors[1]  # In-plane basis
    planarity = float(singular_values[2] / singular_values[0])  # 0 = exactly coplanar

    to_plane = lambda p: np.array([np.dot(p - centroid, e1), np.dot(p - centroid, e2)])
    a, b = to_plane(leading_edges[0]), to_plane(leading_edges[1])
    c, d = to_plane(closures[0]), to_plane(closures[1])

    le_dir, closure_dir = b - a, d - c
    denominator = le_dir[0] * closure_dir[1] - le_dir[1] * closure_dir[0]
    if abs(denominator) < 1e-12:
        raise ValueError("Leading edge and box closure are parallel at the tip: "
                         "no tow point can be constructed.")

    t = ((c[0] - a[0]) * closure_dir[1] - (c[1] - a[1]) * closure_dir[0]) / denominator
    intersection_uv = a + t * le_dir

    return {
        "leading_edges": leading_edges,
        "closures": closures,
        "intersection": centroid + intersection_uv[0] * e1 + intersection_uv[1] * e2,
        "planarity": planarity,
        "t_leading_edge": float(t),  # <0 : outboard of the outermost rib
        # Basis of the plane the intersection was solved in, so the construction can
        # be drawn flat: uv = (point - plane_origin) @ plane_basis.T
        "plane_origin": centroid,
        "plane_basis": np.vstack([e1, e2]),
    }


def tow_point_construction(ribs_data, profiles_dir):
    """`tip_intersection` on both tips. `ribs_data` is sorted tip to tip, so each
    tip's two outermost ribs are the first and last pairs -- not the ribs of
    largest |y|, since the tip of an arched wing curls back inboard."""
    return [
        tip_intersection(ribs_data[0], ribs_data[1], profiles_dir),
        tip_intersection(ribs_data[-1], ribs_data[-2], profiles_dir),
    ]


def tow_point(construction):
    """Tow point from the tip construction: the two tips averaged, then put on the
    symmetry plane y = 0, where the resultant tether force applies."""
    point = np.mean([tip["intersection"] for tip in construction], axis=0)
    point[1] = 0.0
    return point


def regularize_profile(dat_file_path, sail_start_x=0.50, sail_thickness=0.004,
                       n_profile_points=200):
    """Close an exported ram-air profile so a closed-airfoil polar model can digest it.

    The SurfPlan contour is, in Selig order: trailing edge -> mono-skin rear sail
    -> nose upper skin -> leading edge -> nose lower skin -> closing ramp ->
    mono-skin rear sail -> trailing edge. The closing ramp runs from the lower
    skin up onto the upper skin, and the sail behind it has a thickness of ~1e-4
    chord: that pair breaks the CST/Kulfan fit closed-airfoil models rely on.

    The upper skin -- the suction side, hence most of the lift -- is left
    untouched, and so is the nose lower skin up to the start of the closing ramp.
    Behind it the lower side is the upper side minus a thickness t(x), a cubic
    Hermite spline through three knots:

    - x0, the box end: t0 is the real thickness there and its slope makes the
      lower side leave the box with the slope of the lower skin (no kink);
    - `sail_start_x`: the sail has reached `sail_thickness` (fraction of chord);
    - the trailing edge: t = 0, a sharp trailing edge. The slope at the middle
      knot is the secant of the last segment, a straight taper.

    t decreases monotonically (checked), so the lower side only ever gets closer
    to the upper side. Returns the closed contour as an (n, 2) array in Selig order.
    """
    coordinates = read_profile(dat_file_path)

    i_le = int(np.argmin(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2))
    upper = coordinates[: i_le + 1][::-1]  # LE -> TE
    lower = coordinates[i_le:]             # LE -> TE

    slopes = np.diff(lower[:, 1]) / (np.diff(lower[:, 0]) + 1e-12)

    # The closing ramp is the straight run that ends at the big upward jump; the
    # box ends where that run starts.
    i_jump = int(np.argmax(np.diff(lower[:, 1]))) + 1
    ramp_slope = slopes[i_jump - 1]
    i_start = i_jump - 1
    while i_start > 1 and abs(slopes[i_start - 1] - ramp_slope) < 0.02:
        i_start -= 1

    x0, y0 = lower[i_start]
    m0 = (lower[i_start, 1] - lower[i_start - 2, 1]) / (lower[i_start, 0] - lower[i_start - 2, 0])
    if not x0 < sail_start_x < 1.0:
        raise ValueError(f"sail_start_x = {sail_start_x} must lie between the box end "
                         f"(x/c = {x0:.3f}) and the trailing edge")

    x = np.linspace(0.0, 1.0, n_profile_points) ** 1.3  # Denser near the LE
    y_upper = np.interp(x, upper[:, 0], upper[:, 1])

    eps = 1e-3
    upper_slope_0 = (np.interp(x0 + eps, upper[:, 0], upper[:, 1])
                     - np.interp(x0 - eps, upper[:, 0], upper[:, 1])) / (2 * eps)
    t0 = np.interp(x0, upper[:, 0], upper[:, 1]) - y0
    taper = -sail_thickness / (1.0 - sail_start_x)

    thickness = CubicHermiteSpline(
        [x0, sail_start_x, 1.0],
        [t0, sail_thickness, 0.0],
        [upper_slope_0 - m0, taper, taper],
    )(np.clip(x, x0, 1.0))

    if np.any(np.diff(thickness) > 1e-12):
        raise ValueError(
            f"{Path(dat_file_path).name}: the sail thickness is not monotone between the box "
            f"end (x/c = {x0:.3f}, t = {t0:.3f}) and the trailing edge; move sail_start_x "
            f"({sail_start_x}) or lower sail_thickness ({sail_thickness})."
        )

    y_box = np.interp(x, lower[: i_start + 1, 0], lower[: i_start + 1, 1])
    y_lower = np.where(x <= x0, y_box, y_upper - thickness)

    return np.vstack([
        np.column_stack([x, y_upper])[::-1],  # TE -> LE
        np.column_stack([x, y_lower])[1:],    # LE -> TE
    ])


def _quad_area_and_centroid(a, b, c, d):
    """Area and centroid of the quadrilateral abcd, split into triangles abc and acd."""
    def triangle(p, q, r):
        return 0.5 * np.linalg.norm(np.cross(q - p, r - p)), (p + q + r) / 3.0

    area_1, centroid_1 = triangle(a, b, c)
    area_2, centroid_2 = triangle(a, c, d)
    if area_1 + area_2 < 1e-16:
        return 0.0, 0.25 * (a + b + c + d)
    return area_1 + area_2, np.average([centroid_1, centroid_2], axis=0, weights=[area_1, area_2])


def fabric_mass_cg(ribs_data, profiles, skin_areal_density, rib_areal_density=None,
                   n_skins=2, skin_surface_factor=1.0, rib_surface_factor=1.0):
    """Fabric-only mass [kg] and CG: `n_skins` canopy skins plus the internal ribs.

    No material data is held here: the areal densities [kg/m^2] (and the surface
    factors that account for seams and overlaps) are the caller's. The skins are
    the quadrilateral panels between consecutive ribs; each rib panel is the area
    enclosed by its closed profile, `profiles[airfoil_id]` (an (n, 2) array in
    chord fractions, e.g. from `regularize_profile`), scaled by the chord squared.
    `rib_areal_density` defaults to `skin_areal_density`.
    """
    rib_areal_density = skin_areal_density if rib_areal_density is None else rib_areal_density

    skin_masses, skin_cgs = [], []
    for rib_1, rib_2 in zip(ribs_data[:-1], ribs_data[1:]):
        area, centroid = _quad_area_and_centroid(rib_1["LE"], rib_1["TE"], rib_2["TE"], rib_2["LE"])
        skin_masses.append(n_skins * skin_areal_density * skin_surface_factor * area)
        skin_cgs.append(centroid)
    skin_mass = np.sum(skin_masses)
    skin_cg = np.average(skin_cgs, axis=0, weights=skin_masses)

    rib_masses, rib_cgs = [], []
    for rib in ribs_data:
        x, y = profiles[rib["airfoil_id"]][:, 0], profiles[rib["airfoil_id"]][:, 1]
        enclosed_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # Shoelace
        rib_masses.append(rib_areal_density * rib_surface_factor * enclosed_area * rib["chord"] ** 2)
        rib_cgs.append((rib["LE"] + rib["TE"]) / 2)
    rib_mass = np.sum(rib_masses)
    rib_cg = np.average(rib_cgs, axis=0, weights=rib_masses)

    masses = np.array([skin_mass, rib_mass])
    return float(np.sum(masses)), np.average([skin_cg, rib_cg], axis=0, weights=masses)


def _as_list(v):
    return [float(x) for x in np.asarray(v, dtype=float)]


def _write_yaml(path, data):
    """UTF-8 with LF line endings whatever the platform."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main(surfplan_txt_file_path, profile_load_dir, output_dir, reynolds=1.5e6,
         sail_start_x=0.50, sail_thickness=0.004, n_profile_points=200,
         keep_monotonic_span=True, mass_model=None):
    """Write a ram-air kite's adapter files into `output_dir`:

      - `profiles/prof_N.dat`  the regularized, closed profiles of the kept ribs;
      - `aero_geometry.yaml`   kept ribs + profiles, sorted +y to -y, with profile
                               paths relative to `output_dir`;
      - `ramair_wing.yaml`     every rib (kept or not), the box closure of each
                               profile, the tip tow points and, when `mass_model`
                               is given, the fabric mass and CG.

    `mass_model`, the keyword arguments of `fabric_mass_cg` other than the ribs
    and profiles (at least `skin_areal_density`), is left out by default: no
    material data ships with the adapter. Returns the ramair_wing dict.
    """
    profile_load_dir, output_dir = Path(profile_load_dir), Path(output_dir)
    every = read_ramair_ribs(surfplan_txt_file_path)
    kept = trim_folded_tips(every) if keep_monotonic_span else every
    kept_idx = {rib["rib_index"] for rib in kept}
    construction = tow_point_construction(every, profile_load_dir)

    (output_dir / "profiles").mkdir(parents=True, exist_ok=True)
    ids = sorted({rib["airfoil_id"] for rib in kept})
    profiles, box = {}, {}
    for airfoil_id in ids:
        source = profile_load_dir / f"prof_{airfoil_id}.dat"
        profiles[airfoil_id] = regularize_profile(source, sail_start_x, sail_thickness,
                                                  n_profile_points)
        with open(output_dir / "profiles" / f"prof_{airfoil_id}.dat", "w",
                  encoding="utf-8", newline="\n") as f:
            np.savetxt(f, profiles[airfoil_id], fmt="%.7f", header=f"prof_{airfoil_id}",
                       comments="")
        box[airfoil_id] = _as_list(box_closure_station(source))

    sections = sorted(kept, key=lambda rib: -rib["LE"][1])
    aero = {
        "wing_sections": {
            "headers": ["airfoil_id", "LE_x", "LE_y", "LE_z", "TE_x", "TE_y", "TE_z"],
            "data": [[int(rib["airfoil_id"]), *_as_list(rib["LE"]), *_as_list(rib["TE"])]
                     for rib in sections]},
        "wing_airfoils": {
            "reynolds": reynolds,
            "headers": ["airfoil_id", "type", "info_dict"],
            "data": [[airfoil_id, "neuralfoil",
                      {"dat_file_path": f"profiles/prof_{airfoil_id}.dat"}]
                     for airfoil_id in ids]},
    }
    _write_yaml(output_dir / "aero_geometry.yaml", aero)

    wing = {
        "source": Path(surfplan_txt_file_path).name,
        "frame": "VSM: x chord (LE to TE), y span to the left, z up",
        "ribs": [{"rib_index": int(rib["rib_index"]), "airfoil_id": int(rib["airfoil_id"]),
                  "kept": rib["rib_index"] in kept_idx, "chord": float(rib["chord"]),
                  "LE": _as_list(rib["LE"]), "TE": _as_list(rib["TE"]),
                  "VUP": _as_list(rib["VUP"])}
                 for rib in every],
        "box_closure": {int(k): v for k, v in box.items()},
        "tow_point_tips": [{"intersection": _as_list(tip["intersection"]),
                            "planarity": float(tip["planarity"]),
                            "t_leading_edge": float(tip["t_leading_edge"])}
                           for tip in construction],
        "tow_point": _as_list(tow_point(construction)),
    }
    if mass_model is not None:
        mass, cg = fabric_mass_cg(kept, profiles, **mass_model)
        wing["fabric_mass"] = mass
        wing["fabric_cg"] = _as_list(cg)
    _write_yaml(output_dir / "ramair_wing.yaml", wing)
    return wing
