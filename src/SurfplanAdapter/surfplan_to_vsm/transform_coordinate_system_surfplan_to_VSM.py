import numpy as np


def transform_coordinate_system_surfplan_to_VSM(coord_surfplan):
    """
    Transform coordinate from Surfplan reference frame to VSM reference frame

    Surfplan reference frame :
    # z: along the chord / parallel to flow direction
    # x: left
    # y: upwards

    VSM reference frame :
    # Body EastNorthUp (ENU) Reference Frame (aligned with Earth direction)
    # x: along the chord / parallel to flow direction
    # y: left
    # z: upwards

    Parameters:
    coord_surfplan (tuple): a tuple of three floats representing the x, y, and z coordinates of the rib endpoint in Surfplan reference frame.

    Returns:
    coord_vsm (tuple): a tuple of three floats representing the x, y, and z coordinates of the rib endpoint in VSM reference frame.
    """

    # Rotation matrix
    R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    # R = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
    coord_vsm = np.dot(R, coord_surfplan)

    return coord_vsm
