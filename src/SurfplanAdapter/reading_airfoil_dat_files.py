import matplotlib.pyplot as plt
import math

# Read main caracteristics of an ILE profile 
# /!\ The .dat file should respect XFoil norm, the points start at the TE and go to the LE through the extrado and come back to TE through the intrado
def read_profile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    profile_name = lines[0].strip() # Name of the profile
    tube_diameter = None            # LE tube diameter of the airfoil, in % of the chord 
    depth = -float('inf')           # Depth of the profile, in % of the chord 
    x_depth = None                  # Position of the maximum depth of the profile, in % of the chord 
    TE_angle_deg = None           # Angle of the TE

    # Read profile points
    for line in lines[1:]:
        x, y = map(float, line.split())
        if y > depth:
            depth = y
            x_depth = x

    # Read TE angle 
    # TE angle is defined here as the angle between the horizontal and the TE extrado line
    # The TE extrado line is going from the TE to the 3rd point of the extrado from the TE
    if len(lines) > 4:
        (x1, y1) = map(float, lines[1].split())
        (x2, y2) = map(float, lines[3].split())
        delta_x = x2 - x1
        delta_y = y2 - y1
        TE_angle_rad = math.atan2(delta_y, delta_x)
        TE_angle_deg = 180 - math.degrees(TE_angle_rad)
    else:
        TE_angle_deg = None  # Not enough points to calculate the angle

    return profile_name, tube_diameter, 100*depth, 100*x_depth, TE_angle_deg

#Plot the profile and display caracteristcs
def plot_profiles(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    points = []  # List to store (x, y) points
    # Store profile points
    for line in lines[1:]:
        x, y = map(float, line.split())
        points.append((x, y))
    # Unzip points into x and y coordinates
    x_points, y_points = zip(*points)
    #Read profile caracteristics
    profile_name, tube_diameter, depth, x_depth, TE_angle = read_profile(filename)
    
    # Plot the profile points
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker='', linestyle='-', color='b')
    plt.scatter([x_depth/100], [depth/100], color='r', zorder=5, label='Highest Point')
    
    # Annotate the highest point
    plt.annotate(f'({x_depth/100}, {depth/100})', xy=(x_depth/100, depth/100), xytext=(x_depth/100, depth/100 + 0.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
   
    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(profile_name)
    # Label with profile caracteristics
    legend = f"Depth: {depth:.2f}%\nx_depth = {x_depth:.2f}%\nTE Angle: {TE_angle:.2f}°"
    plt.legend([legend], loc='best')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Set equal aspect ratio

    # Show the plot
    plt.show()


# Example usage:
filename = 'data/default_kite/profiles/prof_1.dat'
profile_name, tube_diameter, depth, x_depth, TE_angle = read_profile(filename)
plot_profiles(filename)
print(f"Profile Name: {profile_name}")
print(f"Highest Point X Coordinate (x_depth): {x_depth} %")
print(f"Highest Point Y Coordinate (depth): {depth} %")
print(f"TE angle: {TE_angle:.2f}°")