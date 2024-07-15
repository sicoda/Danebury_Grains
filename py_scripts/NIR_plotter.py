import numpy as np
import matplotlib.pyplot as plt

# read in data from the .txt file
filename = "..." #must contain only homogeneous TXT comma-separated files
#data = np.loadtxt(directory)

with open(filename,'r') as f:
    raw = f.readlines()
file_data = []
for line in raw:
    line = line.replace('\n','')
    line = line.split(',')
    file_data.append(line)
file_data = np.array(file_data,dtype=np.float64)

x = file_data[:, 0]
y = file_data[:, 1]

# Create plot
plt.figure()
plt.plot(x, y, linestyle='-')

# Add titles and labels
plt.title('4S.16 NIR Spectrum')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Reflectance')

plt.savefig("...",dpi=300,bbox_inches='tight')
plt.show()

