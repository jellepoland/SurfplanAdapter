### The main script that will
# 1. Load the data
# 2. Process the data and store in processed_data
# 3. Run the solver, iteratively for multiple angles of attack
# 4. Store the results in results folder

import VSM

# Load the data
# get data from txt
data = load_data("data.txt")
# get data from .dat
data.update(load_data("data2.data"))

wing = VSM.Wing(n_panels=10)
for dict_i in dict:
    wing.add_section(dict_i[TE], dict_i[LE], dict_i["aero_input")
va = [-10, 0, 0]
# create object wing that has all the aerodynamic data
wing_aero = WingAerodynamics([wing])
wing_aero.va = va

VSM = Solver("VSM")
result, wing_aero = VSM.solve(wing_aero)

wing_aero.plot('cl,cd,alpha')