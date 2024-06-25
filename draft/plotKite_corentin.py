import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Rib:
    def __init__(self):
        self.LE = []    #LE 3d coordinate
        self.TE = []    #TE 3d coordinate
        self.VUP = []   #
        self.profile3d = []
    
    def __init__(self, LE, TE, VUP):
        self.LE = LE
        self.TE = TE
        self.VUP = VUP
        self.profile3d = []
    
    #read the .dat file of the 2d profile and switch it into 3d in the kite reference
    def read_dat_file(self, dat_filename):
        points2d = []

        #read .dat file
        with open(dat_filename, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('prof'): #name of the profile
                print('skip line')
                continue
            else  :
                values = line.split()
                if len(values) == 2:
                    try:
                        x, y = float(values[0]), float(values[1])
                        xy = np.array([x, y, 0], dtype=float)
                        print(xy)
                        points2d.append(xy)
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
        
        #Project 2d profile into the 3d reference of the kite
        # Compute the first basis vector (x-axis in the plane)
        x_basis = self.TE - self.LE
        x_basis = x_basis / np.linalg.norm(x_basis)
        print('x norm =', np.linalg.norm(x_basis))
        # Compute the second basis vector (y-axis in the plane)
        y_basis = self.VUP / np.linalg.norm(self.VUP)
        print('y norm =', np.linalg.norm(y_basis))
        # Calculate the normal vector of the plane containing the profile by taking the cross product of LE_to_TE and VUP
        normal_vector = np.cross(x_basis , y_basis )
        # Ensure normal vector is a unit vector
        print(np.linalg.norm(normal_vector))
        # Create the transformation matrix
        # transformation_matrix = np.vstack([x_basis, y_basis, normal_vector]).T

        # points3d = [self.LE + ]                
        # self.profile3d = points3d
                        

    
    def print(self):
        print(f"Leading Edge (LE): {self.LE}")
        print(f"Trailing Edge (TE): {self.TE}")
        print(f"Up vector (VUP): {self.VUP}")

class LETubeSection:
    def __init__(self, centre, diam):
        self.centre = centre
        self.diam = diam

    def print(self):
        print(f"Centre: {self.centre}")
        print(f"Diameter: {self.diam}")

class StrutSection:
    def __init__(self, centre, diam):
        self.centre = centre
        self.diam = diam

    def print(self):
        print(f"Centre: {self.centre}")
        print(f"Diameter: {self.diam}")

class Strut:
    def __init__(self):
        self.sections = []

    #def __init__(self, strut_sections):
    #    self.sections = strut_sections

    def add_section(self, section):
        self.sections.append(section)
    
    def print(self):
        for section in self.sections:
            section.print()

class Bridle:
    def __init__(self, top, bottom, name, length, material):
        self.top = top
        self.bottom = bottom
        self.name = name
        self.length = length
        self.material = material

    def print(self):
        print(f"Top: {self.top}")
        print(f"Bottom: {self.bottom}")
        print(f"Name: {self.name}")
        print(f"Length: {self.length}")
        print(f"Material: {self.material}")

class Kite:
    def __init__(self):
        self.ribs = []
        self.le_tube = []
        self.struts = []
        self.bridle = []

    def read_from_txt(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        section = None
        current_strut = Strut()

        for line in lines:
            line = line.strip()
            if line.startswith('3d rib positions'):
                section = 'ribs'
                continue
            elif line.startswith('LE tube'):
                section = 'le_tube'
                continue
            elif line.startswith('Strut'):
                section = 'strut'
                continue
            elif line.startswith('3d Bridle'):
                section = 'bridle'
                continue
            #Read kite ribs
            if section == 'ribs':
                if not line or line.isdigit() or not any(char.isdigit() for char in line):
                    continue  # Skip empty or comments lines
                values = list(map(float, line.replace(',', '.').split(';')))
                if len(values) == 9:
                    le = np.array(values[0:3])
                    te = np.array(values[3:6])
                    vup = np.array(values[6:9])
                    self.ribs.append(Rib(le, te, vup))
            #Read Kite LE tube
            elif section == 'le_tube':
                if not line or line.isdigit() or not any(char.isdigit() for char in line):
                    continue  # Skip empty or comments lines
                values = list(map(float, line.replace(',', '.').split(';')))
                if len(values) == 4:
                    centre = np.array(values[0:3])
                    diam = values[3]
                    self.le_tube.append(LETubeSection(centre, diam))
            #Read Struts
            elif section == 'strut':
                if not line: #end of the strut section
                    self.struts.append(current_strut) #add the strut to the struts list of the kite
                    current_strut = Strut()
                    continue
                if line.isdigit() or not any(char.isdigit() for char in line):
                    continue  # Skip comments lines
                values = list(map(float, line.replace(',', '.').split(';')))
                if len(values) == 4:
                    centre = np.array(values[0:3])
                    diam = values[3]
                    strut_section = StrutSection(centre, diam)
                    current_strut.add_section(strut_section)

            #Read Bridle
            elif section == 'bridle':
                if not line or line.isdigit() or not any(char.isdigit() for char in line):
                    continue  # Skip empty or comments lines
                parts = line.replace(',', '.').split(';')
                top = np.array(list(map(float, parts[0:3])))
                bottom = np.array(list(map(float, parts[3:6])))
                name = parts[6]
                length = float(parts[7].replace(',', '.'))
                material = parts[8] if len(parts) > 8 else ""
                self.bridle.append(Bridle(top, bottom, name, length, material))


def plot_data(rib_data, le_tube_data, struts_data, bridle_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(rib_data) == 0 and le_tube_data.size == 0 and len(struts_data) == 0 and len(bridle_data) == 0:
        print("No data to plot.")
        return

    first_rib = True
    for rib in rib_data:
        label = 'ribs' if first_rib else ''
        ax.scatter(rib.LE[0], rib.LE[1], rib.LE[2], c='c', label=label, marker='.')
        ax.scatter(rib.TE[0], rib.TE[1], rib.TE[2], c='c', marker='.')
        ax.scatter(rib.VUP[0], rib.VUP[1], rib.VUP[2], c='c', marker='.')
        first_rib = False

    first_le_section = True
    for le_section in le_tube_data:
        label = 'LE Tube' if first_le_section else ''
        ax.scatter(le_section.centre[0], le_section.centre[1], le_section.centre[2], c='g', label=label, marker='^')
        first_le_section = False

    first_strut = True
    for strut in struts_data:
        for section in strut.sections:
            label = 'Struts' if first_strut else ''
            ax.scatter(section.centre[0], section.centre[1], section.centre[2], c='b', label=label, marker='o')
            first_strut = False
    
    first_bridle = True
    for bridle in bridle_data:
        label = '3d bridle' if first_bridle else ''
        ax.plot([bridle.top[0], bridle.bottom[0]], [bridle.top[1], bridle.bottom[1]], [bridle.top[2], bridle.bottom[2]], c='r', label=label)
        first_bridle = False

    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.show()

#Usage 
# Initialize Kite instance
kite = Kite()
# Read the data from the file
filename = 'data/default_kite/default_kite_3d.txt'
#filename = 'test_cases/Seakite50_VH/SK50-VH_3d.txt'
kite.read_from_txt(filename)

# Plot the data
plot_data(kite.ribs, kite.le_tube, kite.struts, kite.bridle)

