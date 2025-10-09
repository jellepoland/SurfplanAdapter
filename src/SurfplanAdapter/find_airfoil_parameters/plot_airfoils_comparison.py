#!/usr/bin/env python3
"""
Plot all airfoils defined in a YAML configuration file.

This script reads a kite configuration YAML file and plots all the airfoils
defined in the wing_airfoils section. Each airfoil is plotted in a separate
subplot arranged in a single column, with optional CAD-sliced airfoil overlays.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from VSM.plot_styling import set_plot_style
from SurfplanAdapter.find_airfoil_parameters import utils_lei_parametric_copy


def load_cad_airfoil(dat_file_path):
    """
    Load airfoil coordinates from a .dat file.

    Parameters:
    -----------
    dat_file_path : Path or str
        Path to the .dat file containing airfoil coordinates

    Returns:
    --------
    numpy.ndarray or None
        Nx2 array of (x, y) coordinates, or None if file doesn't exist
    """
    dat_path = Path(dat_file_path)
    if not dat_path.exists():
        return None

    try:
        # Load data (space-separated x,y coordinates, skip first line which is airfoil name)
        data = np.loadtxt(dat_path, skiprows=1)
        return data
    except Exception as e:
        print(f"  Warning: Could not load {dat_path}: {e}")
        return None


def extract_airfoils_from_yaml(yaml_path):
    """
    Extract all airfoil definitions from a YAML configuration file.

    Parameters:
    -----------
    yaml_path : Path or str
        Path to the YAML configuration file

    Returns:
    --------
    list
        List of dictionaries, each containing airfoil parameters
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    airfoils = []

    if "wing_airfoils" in config and "data" in config["wing_airfoils"]:
        wing_airfoils_data = config["wing_airfoils"]["data"]

        for airfoil_entry in wing_airfoils_data:
            airfoil_id, airfoil_type, params = airfoil_entry

            if airfoil_type == "masure_regression":
                airfoil_info = {
                    "id": airfoil_id,
                    "type": airfoil_type,
                    "t": params.get("t", 0.1),
                    "eta": params.get("eta", 0.3),
                    "kappa": params.get("kappa", 0.1),
                    "delta": params.get("delta", 0.0),
                    "lambda": params.get("lambda", 0.5),
                    "phi": params.get("phi", 0.7),
                }
                airfoils.append(airfoil_info)

    return airfoils


def plot_all_airfoils(yaml_path, output_path=None, surfplan_airfoils_dir=None):
    """
    Plot all airfoils from a YAML configuration file.

    Parameters:
    -----------
    yaml_path : Path or str
        Path to the YAML configuration file
    output_path : Path or str, optional
        Path to save the output figure
    surfplan_airfoils_dir : Path or str, optional
        Directory containing CAD-sliced airfoil .dat files (named prof_1.dat, prof_2.dat, etc.)
    """
    # Apply plot styling
    set_plot_style()

    # Extract airfoils from YAML
    airfoils = extract_airfoils_from_yaml(yaml_path)

    if not airfoils:
        print("No airfoils found in the YAML file.")
        return

    n_airfoils = len(airfoils)
    print(f"Found {n_airfoils} airfoil(s) in {yaml_path}")

    # Count available Surfplan airfoil files if directory is provided
    n_surfplan_files = 0
    last_surfplan_id = None
    if surfplan_airfoils_dir is not None and Path(surfplan_airfoils_dir).exists():
        # Find all prof_*.dat files and sort them numerically by ID
        surfplan_files = list(Path(surfplan_airfoils_dir).glob("prof_*.dat"))
        # Sort by extracting numeric ID from filename
        surfplan_files = sorted(surfplan_files, key=lambda f: int(f.stem.split("_")[1]))
        n_surfplan_files = len(surfplan_files)
        if n_surfplan_files > 0:
            # Extract ID from last file (e.g., prof_12.dat -> 12)
            last_file = surfplan_files[-1]
            last_surfplan_id = int(last_file.stem.split("_")[1])
            print(
                f"Found {n_surfplan_files} Surfplan airfoil file(s), last ID: {last_surfplan_id}"
            )
            if n_surfplan_files < n_airfoils:
                print(
                    f"  → Will use prof_{last_surfplan_id}.dat for last airfoil comparison"
                )

    # Create figure with subplots
    fig_width = 12
    fig_height = 2 * n_airfoils
    fig, axes = plt.subplots(n_airfoils, 1, figsize=(fig_width, fig_height))

    # Handle single airfoil case
    if n_airfoils == 1:
        axes = [axes]

    # Plot each airfoil
    for idx, (ax, airfoil) in enumerate(zip(axes, airfoils)):
        print(f"\nGenerating airfoil {idx + 1}/{n_airfoils}:")
        print(f"  ID: {airfoil['id']}")
        print(
            f"  Parameters: t={airfoil['t']:.4f}, η={airfoil['eta']:.4f}, "
            f"κ={airfoil['kappa']:.4f}, δ={airfoil['delta']:.1f}°, "
            f"λ={airfoil['lambda']:.4f}, φ={airfoil['phi']:.4f}"
        )

        try:
            # Generate airfoil geometry using complete implementation
            result = utils_lei_parametric_copy.generate_profile(
                t_val=airfoil["t"],
                eta_val=airfoil["eta"],
                kappa_val=airfoil["kappa"],
                delta_val=airfoil["delta"],
                lambda_val=airfoil["lambda"],
                phi_val=airfoil["phi"],
            )

            # Unpack result (returns tuple: all_points, profile_name, seam_a)
            if result is None or len(result) != 3:
                raise ValueError("generate_profile returned invalid result")

            points, _, _ = result

            # Plot generated airfoil (black)
            ax.plot(
                points[:, 0],
                points[:, 1],
                "k-",
                linewidth=2,
                label="Generated (masure_regression)",
            )
            ax.fill(points[:, 0], points[:, 1], color="lightgray", alpha=0.3)

            # Try to load and plot Surfplan airfoil if available (blue)
            surfplan_points = None
            surfplan_file = None
            if surfplan_airfoils_dir is not None and n_surfplan_files > 0:
                # Determine which Surfplan file to use
                if airfoil["id"] < last_surfplan_id:
                    # Use matching ID for profiles 1 through (last_surfplan_id - 1)
                    surfplan_file = (
                        Path(surfplan_airfoils_dir) / f"prof_{airfoil['id']}.dat"
                    )
                elif airfoil["id"] == n_airfoils:
                    # For the LAST YAML profile, use the last Surfplan file
                    surfplan_file = (
                        Path(surfplan_airfoils_dir) / f"prof_{last_surfplan_id}.dat"
                    )
                    print(
                        f"  Note: Using prof_{last_surfplan_id}.dat for last airfoil {airfoil['id']}"
                    )
                else:
                    # For middle profiles (from last_surfplan_id to second-to-last), no Surfplan overlay
                    surfplan_file = None

                if surfplan_file is not None and surfplan_file.exists():
                    surfplan_points = load_cad_airfoil(surfplan_file)

                if surfplan_points is not None:
                    ax.plot(
                        surfplan_points[:, 0],
                        surfplan_points[:, 1],
                        "b-",
                        linewidth=2,
                        label="Surfplan (sliced)",
                        alpha=0.8,
                    )
                    print(f"  ✓ Surfplan airfoil loaded from {surfplan_file.name}")
                    max_y = max(np.concatenate((points[:, 1], surfplan_points[:, 1])))
                    min_y = min(np.concatenate((points[:, 1], surfplan_points[:, 1])))
                else:
                    max_y = max(points[:, 1])
                    min_y = min(points[:, 1])
            else:
                max_y = max(points[:, 1])
                min_y = min(points[:, 1])

            # Formatting
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("$x/c$ (-)")
            ax.set_ylabel("$y/c$ (-)")

            # Add legend if Surfplan data was plotted
            if surfplan_points is not None:
                ax.legend(loc="upper right", fontsize=8)

            # Title with parameters
            title = (
                f"Airfoil {airfoil['id']}: "
                f"$t$={airfoil['t']:.3f}, $\\eta$={airfoil['eta']:.3f}, "
                f"$\\kappa$={airfoil['kappa']:.3f}, $\\delta$={airfoil['delta']:.1f}°, "
                f"$\\lambda$={airfoil['lambda']:.3f}, $\\phi$={airfoil['phi']:.3f}"
            )
            ax.set_title(title, pad=10)

            # Set reasonable axis limits
            y_range = max_y - min_y
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)

            print(f"  ✓ Successfully generated")

        except Exception as e:
            print(f"  ✗ Error generating airfoil: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"\nFigure saved to: {output_path}")

    plt.close(fig)
