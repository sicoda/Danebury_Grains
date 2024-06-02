#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt

# read in data from the .txt file
filename = "/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/NIR/NIR_Abs/4S.16.txt" #must contain only homogeneous TXT comma-separated files
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

plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/NIR/NIR_Specs/4S.16.png",dpi=300,bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:




