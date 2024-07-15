#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import scipy.stats as spstat
from sklearn.decomposition import PCA
import scipy.optimize as scpo


######## inputs
directory = "..." #must contain only homogeneous TXT comma-separated files
selected_species = ['Einkorn'] #[] or 'all'
colorscale = mpl.colormaps['rainbow'].resampled(40) #color scale for categorizing scatter points
pca_components = 2 #number of PCA components to use
mode = 'plot_PCA' #'plot_data' to plot original data, 'plot_PCA' to plot PCA-transformed results
save = True

##################################################################################

#Read excel file
Coff = pd.read_excel(directory,sheet_name="Sheet1",header=0,index_col=None)

#Get total number of excel rows
temp = Coff['temp']
N = temp.shape[0]

#collate selected data
x = []
y = []
z = []
if selected_species == 'all':
    selected_species = list(np.unique(species))

for i in range(N):
    if species[i] in selected_species:
        x.append(Coff['temp'][i])
        y.append(Coff['duration'][i])
        z.append(Coff['offset'][i])
        
x = np.array(x)
y = np.array(y)
z = np.array(z)

if mode == 'plot_data':
    #normalise c
    z = (z - np.min(z))/(np.max(z)-np.min(z))

    plt.scatter(x,z,s=y, c='g')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("$\delta$$^{13}$C Offset")
    if save: plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/Initial data/LitData_all.png", bbox_inches='tight')
    plt.show()


elif mode == 'plot_PCA':
    data = np.zeros((x.shape[0],3))
    data[:,0] = x
    data[:,1] = y
    data[:,2] = z
    
    ####Perform PCA

    pca_C = PCA(n_components=pca_components)
    pca_C.fit(data)

    ####Prepare and display results of pca
    fig,axs = plt.subplots(1)

    print("[INFO] PCA fitting complete:")
    print("Explained variance:", pca_C.explained_variance_ratio_)

    #Transform spectra
    transf_spec_C = pca_C.transform(data)

    #Normalise offsets
    off_norm = (z - np.min(z))/(np.max(z)-np.min(z))

    plt.scatter(transf_spec_C[:,0],transf_spec_C[:,1],c=colorscale(off_norm))
    if save: plt.savefig("LitDataPCA_einkorn.png")
    plt.show()


# In[31]:


"hello there".title()


# In[63]:


coff_np = np.array(Coff)
coff_np[:,2:]


# In[ ]:





# In[ ]:





# In[ ]:




