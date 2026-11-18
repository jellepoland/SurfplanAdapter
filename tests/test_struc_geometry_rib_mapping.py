"""Struts and tube diameters land on the right ribs in the structural geometry.

The outermost sections are removed before the wing nodes are numbered; every lookup
by rib index (struts, LE/TE tube diameters) must still find that rib's nodes.
"""

import numpy as np
import pytest
import yaml

from SurfplanAdapter.generate_yaml.create_struc_geometry_yaml import (
    _build_leading_edge_tubes,
    _build_strut_tubes,
    _build_wing_particles_and_mapping,
    _remove_outer_sections,
    create_struc_geometry_all_in_surfplan_yaml,
)
from SurfplanAdapter.process_wing import generate_wing_sections_data


def _surfplan(point_vsm):
    """VSM point -> SurfPlan frame (inverse of transform_coordinate_system_surfplan_to_VSM)."""
    x, y, z = point_vsm
    return [-y, z, -x]


def _ribs(span, struts_at, tube=0.05):
    """Synthetic ribs, +y first: chord 2 m along x, LE at z = 1. Each strut has
    SurfPlan's three samples: the root inside the LE tube with diameter 0, then 0.08 m
    near the LE and 0.06 m near the TE."""
    ribs = []
    for i, y in enumerate(span):
        ribs.append({
            "LE": [0.0, y, 1.0], "TE": [2.0, y, 1.0], "VUP": [0.0, 0.0, 1.0],
            "airfoil_id": 1, "chord": 2.0, "d_tube_from_surfplan_txt": tube * (1 + i),
            "is_strut": i in struts_at,
            "strut_samples": [
                {"center": _surfplan([0.0, y, 1.0]), "diameter": 0.0},
                {"center": _surfplan([0.2, y, 1.0]), "diameter": 0.08},
                {"center": _surfplan([1.9, y, 1.0]), "diameter": 0.06},
            ] if i in struts_at else [],
            "vsm_y_rotation_rad": 0.0,
        })
    return ribs


# Folded tip: the tip ribs (|y| = 2.8) lie inboard of their neighbours (|y| = 2.9),
# so the outermost sections are the SECOND and the second-last ribs.
FOLDED = [2.8, 2.9, 2.5, 1.0, -1.0, -2.5, -2.9, -2.8]
# Ordinary tip: the outermost sections are the first and last ribs.
ORDINARY = [2.9, 2.8, 2.5, 1.0, -1.0, -2.5, -2.8, -2.9]


def _structure(span, struts_at):
    ribs = _ribs(span, struts_at)
    sections, kept = _remove_outer_sections(generate_wing_sections_data.main(ribs))
    particles, node_map = _build_wing_particles_and_mapping(sections, kept)
    y_of = {row[0]: row[2] for row in particles["data"]}
    return ribs, kept, node_map, y_of


@pytest.mark.parametrize("span", [FOLDED, ORDINARY], ids=["folded tip", "ordinary tip"])
def test_struts_sit_on_their_own_ribs_on_both_sides(span):
    ribs, _, node_map, y_of = _structure(span, struts_at={2, 5})
    struts = _build_strut_tubes(ribs, node_map)["data"]
    assert [y_of[row[1]] for row in struts] == [2.5, -2.5]
    assert [y_of[row[2]] for row in struts] == [2.5, -2.5]


@pytest.mark.parametrize("span", [FOLDED, ORDINARY], ids=["folded tip", "ordinary tip"])
def test_only_the_outermost_sections_are_removed(span):
    _, kept, node_map, y_of = _structure(span, struts_at=set())
    removed = sorted(set(range(len(span))) - set(kept))
    assert sorted(abs(span[i]) for i in removed) == [2.9, 2.9]
    assert list(node_map) == kept
    assert sorted(y_of[n["LE"]] for n in node_map.values()) == sorted(span[i] for i in kept)


def test_leading_edge_tube_diameters_belong_to_the_ribs_they_join():
    ribs, kept, node_map, y_of = _structure(FOLDED, struts_at=set())
    tubes = {row[0]: row for row in _build_leading_edge_tubes(ribs, node_map)["data"]}
    le_d = {r["LE"][1]: r["d_tube_from_surfplan_txt"] * r["chord"] for r in ribs}
    # le_run_1 joins the first two kept ribs (y = 2.8 and 2.5)
    _, ci, cj, d = tubes["le_run_1"]
    assert (y_of[ci], y_of[cj]) == (2.8, 2.5)
    assert d == pytest.approx(0.5 * (le_d[2.8] + le_d[2.5]))


def test_strut_root_diameter_skips_the_zero_diameter_sample():
    ribs, _, node_map, _ = _structure(ORDINARY, struts_at={2, 5})
    for row in _build_strut_tubes(ribs, node_map)["data"]:
        assert row[3] == pytest.approx(0.08)  # strut_diam_le
        assert row[4] == pytest.approx(0.06)  # strut_diam_te


def test_all_in_surfplan_file_is_mirror_symmetric_on_a_folded_tip(tmp_path):
    ribs = _ribs(FOLDED, struts_at={2, 5}, tube=0.05)
    for rib in ribs:  # symmetric diameters for a symmetric check
        rib["d_tube_from_surfplan_txt"] = 0.05
    create_struc_geometry_all_in_surfplan_yaml(ribs, [], tmp_path / "config_kite.yaml")
    doc = yaml.safe_load(open(tmp_path / "struc_geometry_all_in_surfplan.yaml"))
    y_of = {row[0]: row[2] for row in doc["wing_particles"]["data"]}
    strut_y = [y_of[row[1]] for row in doc["strut_tubes"]["data"]]
    np.testing.assert_allclose(strut_y, [2.5, -2.5])
    assert all(row[3] > 0 for row in doc["strut_tubes"]["data"])
