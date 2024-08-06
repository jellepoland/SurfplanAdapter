from reading_surfplan_txt import read_from_txt
from transforming_coordinate_system import rotate_surfplan_to_VSM
import transforming_coordinate_system
import plotting
from VSM import Wing, WingAerodynamics


def generate_VSM_input(filepath):
    """
    Generate Input for the Vortex Step Method

    Parameters:
    filepath (string) : path of the file to extract the wing data from
                        it should be a txt file exported from Surfplan

    Returns:
    None: This function return an instance of WingAerodynamics which represent the wing described by the txt file
    """
    ribs_data = read_from_txt(filepath)
    n_panels = len(ribs_data) - 1

    #Create wing geometry
    wing = Wing(n_panels, "unchanged")
    for rib in ribs_data:
        wing.add_section(
            rotate_surfplan_to_VSM(rib["LE"]),
            rotate_surfplan_to_VSM(rib["LE"]),
            ["lei_airfoil_breukels", [rib["d_tube"], rib["camber"]]],
        )

    return WingAerodynamics([wing])
    # or return wing ?