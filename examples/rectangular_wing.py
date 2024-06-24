from VSM.WingAerodynamicModel import WingAerodynamics
from VSM.WingGeometry import Wing, Section
from VSM.Solver import Solver

# Use example
################# CAREFULL WITH REFERENCE FRAMES, CHANGING FROM ORIGINAL CODE #################
# Aircraft reference frame
# x: forward
# y: right
# z: down
# Create a wing object
wing = Wing(n_panels=10)

# Add sections to the wing
# arguments are: (leading edge position [x,y,z], trailing edge position [x,y,z], airfoil data)
# airfoil data can be:
# ['inviscid']
# ['lei_airfoil_breukels', [tube_diameter, chamber_height]]
wing.add_section([0, -1, 0], [-1, -1, 0], "inviscid")
wing.add_section([0, 1, 0], [-1, 1, 0], "inviscid")

# Initialize wing aerodynamics
# Default parameters are used (elliptic circulation distribution, 5 filaments per ring)
wing_aero = WingAerodynamics([wing])

# Plotting the wing
wing_aero.plot()

# Initialize solver
# Default parameters are used (VSM, no artificial damping)
VSM = Solver()

# Define inflow conditions
wing_aero.va = [-10, 0, 0]

# solve the aerodynamics
results, wing_aero = VSM.solve(wing_aero)

# Print
print(results)
