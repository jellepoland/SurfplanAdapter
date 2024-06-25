import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the path to the file
file_path = 'enter_file_path'

# loading file, this is used to read some info in the file useful for importing (such as number of rows in matrix)
with open(file_path, 'r') as file:
    content = file.read()   # load file
    content = content.strip()   # remove whitespace
lines = content.split('\n')     # split into a list

hn_ribs = ['LE_X', 'LE_Y', 'LE_Z', 'TE_X', 'TE_Y', 'TE_Z', 'VU_X', 'VU_Y', 'VU_Z']  # manual entry for the names
hn_struts = ['X', 'Y', 'Z', 'D']
hn_bridles = ['XT', 'YT', 'ZT', 'XB', 'YB', 'ZB', 'Name', 'Length', 'Material']
df = {}

for idx, line in enumerate(lines):  # read csv and save to dataframe
    if line.startswith('3d rib'):
        df[str(line)] = pd.read_csv(file_path, sep=';', decimal=',', names=hn_ribs, skiprows=idx + 3,
                                    nrows=int(lines[idx + 2]))
    elif line.startswith('Strut'):
        df[str(line)] = pd.read_csv(file_path, sep=';', decimal=',', names=hn_struts, skiprows=idx + 3,
                                    nrows=int(lines[idx + 2]))
    elif line.startswith('3d Bridle'):
        df[str(line)] = pd.read_csv(file_path, sep=';', decimal=',', names=hn_bridles, skiprows=idx + 3)


print(df.get('3d rib positions'))   # for example

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_box_aspect([1, 1, 1])
ax.plot3D(df['3d rib positions']['LE_X'], df['3d rib positions']['LE_Y'], df['3d rib positions']['LE_Z'], 'blue')
ax.plot3D(df['3d rib positions']['TE_X'], df['3d rib positions']['TE_Y'], df['3d rib positions']['TE_Z'], 'red')
for i in range(len(df['3d rib positions']['LE_X'])):
    ax.plot3D([df['3d rib positions']['LE_X'][i], df['3d rib positions']['TE_X'][i]],
              [df['3d rib positions']['LE_Y'][i], df['3d rib positions']['TE_Y'][i]],
              [df['3d rib positions']['LE_Z'][i], df['3d rib positions']['TE_Z'][i]], 'yellow')
for i in range(len(df['3d Bridle']['XT'])):
    ax.plot3D([df['3d Bridle']['XT'][i], df['3d Bridle']['XB'][i]],
              [df['3d Bridle']['YT'][i], df['3d Bridle']['YB'][i]],
              [df['3d Bridle']['ZT'][i], df['3d Bridle']['ZB'][i]], 'gray')
    ax.plot3D([df['3d Bridle']['XT'][i]*(-1), df['3d Bridle']['XB'][i]*(-1)],
              [df['3d Bridle']['YT'][i], df['3d Bridle']['YB'][i]],
              [df['3d Bridle']['ZT'][i], df['3d Bridle']['ZB'][i]], 'gray')
ax.plot
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

