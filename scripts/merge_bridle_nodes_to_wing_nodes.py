from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

from SurfplanAdapter.generate_yaml import utils
from SurfplanAdapter.plotting import plot_struct_geometry_all_in_surfplan_yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def _build_coord_map(data_rows: Iterable[Iterable]) -> Dict[int, np.ndarray]:
    coord_map: Dict[int, np.ndarray] = {}
    for row in data_rows:
        if not row:
            continue
        node_id = int(row[0])
        coord_map[node_id] = np.array([float(row[1]), float(row[2]), float(row[3])])
    return coord_map


def _find_nearest_node(
    target_coord: np.ndarray,
    candidates: Dict[int, np.ndarray],
    used_ids: set,
) -> Tuple[int | None, float]:
    nearest_id = None
    nearest_distance = float("inf")
    for cand_id, coord in candidates.items():
        if cand_id in used_ids:
            continue
        distance = np.linalg.norm(coord - target_coord)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = cand_id
    return nearest_id, nearest_distance


def _remap_connection_row(row: List, mapping: Dict[int, int]) -> List:
    def remap_value(value):
        if isinstance(value, int):
            return mapping.get(value, value)
        if isinstance(value, float) and float(value).is_integer():
            int_value = int(value)
            remapped = mapping.get(int_value)
            return remapped if remapped is not None else value
        return value

    return [row[0], *[remap_value(val) for val in row[1:]]]


def _augment_strut_tubes_with_bridle_nodes(config: dict) -> None:
    strut_tubes = config.get("strut_tubes")
    if not strut_tubes:
        return

    wing_particles = config.get("wing_particles", {})
    bridle_particles = config.get("bridle_particles", {})
    wing_data = wing_particles.get("data", [])
    bridle_data = bridle_particles.get("data", [])

    if not wing_data:
        return

    wing_coords = _build_coord_map(wing_data)
    bridle_coords = _build_coord_map(bridle_data)

    headers = list(strut_tubes.get("headers", []))
    if not headers:
        return

    if "node_indices" in headers:
        headers.remove("node_indices")
    headers.append("node_indices")
    strut_tubes["headers"] = headers
    core_len = len(headers) - 1

    updated_rows = []
    for row in strut_tubes.get("data", []):
        row = list(row)
        if len(row) < core_len:
            row.extend([None] * (core_len - len(row)))
        core_values = row[:core_len]

        try:
            le_id = int(core_values[1])
            te_id = int(core_values[2])
            le_coord = wing_coords[le_id]
            te_coord = wing_coords[te_id]
        except (KeyError, ValueError):
            node_chain = [core_values[1], core_values[2]]
            updated_rows.append(core_values + [node_chain])
            continue

        chord_vec = te_coord - le_coord
        chord_len = np.linalg.norm(chord_vec)
        if chord_len == 0:
            node_chain = [le_id, te_id]
            updated_rows.append(core_values + [node_chain])
            continue

        chord_unit = chord_vec / chord_len
        max_radial = 0.05 * chord_len

        candidate_nodes: List[Tuple[int, float]] = []
        for node_id, coord in bridle_coords.items():
            rel_vec = coord - le_coord
            proj_length = float(np.dot(rel_vec, chord_unit))
            if proj_length < 0 or proj_length > chord_len:
                continue
            perpendicular_vec = rel_vec - proj_length * chord_unit
            radial_distance = float(np.linalg.norm(perpendicular_vec))
            if radial_distance <= max_radial:
                candidate_nodes.append((node_id, proj_length))

        candidate_nodes.sort(key=lambda item: item[1])
        node_chain = [le_id] + [node_id for node_id, _ in candidate_nodes] + [te_id]
        updated_rows.append(core_values + [node_chain])

    strut_tubes["data"] = updated_rows


def merge_nodes(
    config: dict,
) -> Tuple[dict, Dict[int, int]]:
    wing_particles = config.get("wing_particles", {})
    strut_tubes = config.get("strut_tubes", {})
    bridle_particles = config.get("bridle_particles", {})
    bridle_connections = config.get("bridle_connections", {})

    wing_data = wing_particles.get("data", [])
    strut_data = strut_tubes.get("data", [])
    bridle_data = bridle_particles.get("data", [])
    bridle_connection_rows = bridle_connections.get("data", [])

    wing_coords = _build_coord_map(wing_data)
    bridle_coords = _build_coord_map(bridle_data)

    if not wing_coords or not bridle_coords or not strut_data:
        raise ValueError("Missing wing, bridle, or strut data required for merging.")

    used_bridle_ids = set()
    bridle_to_wing: Dict[int, int] = {}

    for entry in strut_data:
        if len(entry) < 3:
            continue
        le_id = int(entry[1])
        te_id = int(entry[2])

        for node_id in (le_id, te_id):
            if node_id not in wing_coords:
                print(f"Warning: Wing node {node_id} not found in wing_particles.")
                continue

            nearest_id, nearest_distance = _find_nearest_node(
                wing_coords[node_id], bridle_coords, used_bridle_ids
            )
            if nearest_id is None:
                print(
                    f"Warning: No available bridle node found for wing node {node_id}."
                )
                continue

            used_bridle_ids.add(nearest_id)
            bridle_to_wing[nearest_id] = node_id
            print(
                f"Merged bridle node {nearest_id} -> wing node {node_id} "
                f"(distance {nearest_distance:.4f} m)"
            )

    # Also merge first and last trailing-edge nodes (tips) with bridle nodes
    wing_ids_sorted = sorted(wing_coords.keys())
    te_nodes = [node_id for node_id in wing_ids_sorted if node_id % 2 == 0]
    tip_te_candidates = []
    if te_nodes:
        tip_te_candidates.append(te_nodes[0])
        if len(te_nodes) > 1:
            tip_te_candidates.append(te_nodes[-1])

    for node_id in tip_te_candidates:
        if node_id not in wing_coords:
            continue
        # Skip if already paired through strut merging
        if node_id in bridle_to_wing.values():
            continue

        nearest_id, nearest_distance = _find_nearest_node(
            wing_coords[node_id], bridle_coords, used_bridle_ids
        )
        if nearest_id is None:
            print(f"Warning: No available bridle node found for wing node {node_id}.")
            continue

        used_bridle_ids.add(nearest_id)
        bridle_to_wing[nearest_id] = node_id
        print(
            f"Merged bridle node {nearest_id} -> wing tip node {node_id} "
            f"(distance {nearest_distance:.4f} m)"
        )

    # Merge outer leading-edge nodes in the same way
    le_nodes = [node_id for node_id in wing_ids_sorted if node_id % 2 == 1]
    tip_le_candidates = []
    if le_nodes:
        tip_le_candidates.append(le_nodes[0])
        if len(le_nodes) > 1:
            tip_le_candidates.append(le_nodes[-1])

    for node_id in tip_le_candidates:
        if node_id not in wing_coords:
            continue
        if node_id in bridle_to_wing.values():
            continue

        nearest_id, nearest_distance = _find_nearest_node(
            wing_coords[node_id], bridle_coords, used_bridle_ids
        )
        if nearest_id is None:
            print(f"Warning: No available bridle node found for wing node {node_id}.")
            continue

        used_bridle_ids.add(nearest_id)
        bridle_to_wing[nearest_id] = node_id
        print(
            f"Merged bridle node {nearest_id} -> wing tip node {node_id} "
            f"(distance {nearest_distance:.4f} m)"
        )

    if not bridle_to_wing:
        print("No bridle nodes were merged.")
        return config, bridle_to_wing

    # Filter bridle_particles data to remove merged nodes
    filtered_bridle_data = [
        row for row in bridle_data if int(row[0]) not in bridle_to_wing
    ]
    config["bridle_particles"]["data"] = filtered_bridle_data

    # Update bridle connections to reference wing nodes
    remapped_connections = [
        _remap_connection_row(list(row), bridle_to_wing)
        for row in bridle_connection_rows
    ]
    config["bridle_connections"]["data"] = remapped_connections

    _augment_strut_tubes_with_bridle_nodes(config)

    return config, bridle_to_wing


def main(
    kite_name: str = "TUDELFT_V3_KITE",
    input_path: Path | None = None,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> Path:
    project_dir = Path(__file__).resolve().parents[1]

    if input_path is None:
        input_path = (
            project_dir
            / "processed_data"
            / kite_name
            / "struc_geometry_all_in_surfplan.yaml"
        )
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(
            "struc_geometry_all_in_surfplan_merged_nodes.yaml"
        )
    else:
        output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input structural geometry file not found: {input_path}"
        )

    config = _load_yaml(input_path)
    updated_config, mapping = merge_nodes(config)

    if mapping:
        print(
            f"Reassigned {len(mapping)} bridle nodes to wing nodes. "
            f"Saving merged structure to {output_path}"
        )
    else:
        print(
            "No bridle nodes reassigned. Saving a copy of the original configuration."
        )

    utils.save_to_yaml(updated_config, output_path)
    plot_struct_geometry_all_in_surfplan_yaml(output_path, show_plot=show_plot)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge bridle nodes into wing nodes for Surfplan geometry."
    )
    parser.add_argument("--kite-name", default="TUDELFT_V3_KITE", help="Kite name.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional override for the input structural geometry YAML path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional override for the output YAML path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive plotting of the merged geometry.",
    )
    args = parser.parse_args()

    main(
        kite_name=args.kite_name,
        input_path=args.input,
        output_path=args.output,
        show_plot=not args.no_plot,
    )
