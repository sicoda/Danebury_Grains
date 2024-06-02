#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Initialise variables
directory = "/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_Homogenized/" #must contain only homogeneous TXT comma-separated files
pca_components = 2 #number of PCA components to use
wn_start = 1000
wn_end = 2000
cpn = 5000 #density of plotting grid for kmean clustering
cluster = True #whether or not to plot clusters

#Read in data
filenames_raw =  os.listdir(directory)
filenames = []
for filename in filenames_raw:
    if filename.endswith(".txt"):
        filenames.append(filename)
wns = []
all_spec_B = []
all_spec_S = []
filenames_B = []
filenames_S = []
finished_wns = False
for filename in filenames:
    file_data = []
    with open(directory+'/'+filename,'r') as f:
        raw = f.readlines()
    for line in raw:
        line = line.replace('\n','')
        line = line.split(',')
        if not(finished_wns):
            wns.append(line[0])
        file_data.append(line[1])
    file_data = np.array(file_data,dtype=np.float64)
    if 'B' in filename:
        all_spec_B.append(file_data)
        filenames_B.append(filename)
    elif 'S' in filename:
        all_spec_S.append(file_data)
        filenames_S.append(filename)
    if not(finished_wns):
        finished_wns = True
wns = np.array(wns,dtype=np.float64)
all_spec_B = np.array(all_spec_B)
all_spec_S = np.array(all_spec_S)
print("[INFO] Finished reading in data.")

#Determine start and end wavenumber indices
wn_range_args = np.argwhere(np.logical_and(np.greater(wns,wn_start),np.less(wns,wn_end)))
wn_start_arg = wn_range_args[0][0]
wn_end_arg = wn_range_args[-1][0]

#Perform PCA
pca_B = PCA(n_components=pca_components)
pca_B.fit(all_spec_B[:,wn_start_arg:wn_end_arg])
pca_S = PCA(n_components=pca_components)
pca_S.fit(all_spec_S[:,wn_start_arg:wn_end_arg])
print("[INFO] PCA fitting complete:")
print("Explained variance for barley:", pca_B.explained_variance_ratio_)
print("Explained variance for spelt:", pca_S.explained_variance_ratio_)

#Transform spectra
transf_spec_B = pca_B.transform(all_spec_B[:,wn_start_arg:wn_end_arg])
transf_spec_S = pca_S.transform(all_spec_S[:,wn_start_arg:wn_end_arg])

##Perform k-means cluster fit
if cluster:
    #Run k mean fitting
    kmeans_B = KMeans(n_clusters=4).fit(transf_spec_B)
    kmeans_S = KMeans(n_clusters=4).fit(transf_spec_S)

    #get plotting points
    ts_B_min_x, ts_B_max_x = np.min(transf_spec_B[:,0]), np.max(transf_spec_B[:,0])
    ts_B_min_y, ts_B_max_y = np.min(transf_spec_B[:,1]), np.max(transf_spec_B[:,1])
    ts_S_min_x, ts_S_max_x = np.min(transf_spec_S[:,0]), np.max(transf_spec_S[:,0])
    ts_S_min_y, ts_S_max_y = np.min(transf_spec_S[:,1]), np.max(transf_spec_S[:,1])

    ts_B_x = np.linspace(ts_B_min_x,ts_B_max_x,cpn)
    ts_B_y = np.linspace(ts_B_min_y,ts_B_max_y,cpn)
    ts_S_x = np.linspace(ts_S_min_x,ts_S_max_x,cpn)
    ts_S_y = np.linspace(ts_S_min_y,ts_S_max_y,cpn)

    ts_B_xx,ts_B_yy = np.meshgrid(ts_B_x,ts_B_y)
    ts_S_xx,ts_S_yy = np.meshgrid(ts_S_x,ts_S_y)

    ts_B_flat = np.c_[ts_B_xx.ravel(),ts_B_yy.ravel()]
    ts_S_flat = np.c_[ts_S_xx.ravel(),ts_S_yy.ravel()]

    #Predict space
    km_B_z = kmeans_B.predict(ts_B_flat)
    km_S_z = kmeans_S.predict(ts_S_flat)
    km_B_z = km_B_z.reshape(ts_B_xx.shape)
    km_S_z = km_S_z.reshape(ts_S_xx.shape)

#Plot spectra
fig,axs = plt.subplots(2)
if cluster:
    axs[0].imshow(km_B_z,interpolation="nearest",extent=(ts_B_xx.min(), ts_B_xx.max(), ts_B_yy.min(), ts_B_yy.max()),cmap="Greens",aspect="auto",origin="lower")
    axs[1].imshow(km_S_z,interpolation="nearest",extent=(ts_S_xx.min(), ts_S_xx.max(), ts_S_yy.min(), ts_S_yy.max()),cmap="Greens",aspect="auto",origin="lower")
fig.tight_layout()
colors = ['limegreen','greenyellow','yellow','sandybrown','firebrick']
for i,filename in enumerate(filenames_B):
    color_index = int(filename[0])-1
    #filename = filename[:-4]
    sample_id = filename[3:-4]
    axs[0].scatter(transf_spec_B[i,0],transf_spec_B[i,1],c=colors[color_index])
    axs[0].text(transf_spec_B[i,0],transf_spec_B[i,1],sample_id,fontsize='xx-small')
axs[0].set_xlabel("Component B1")
axs[0].set_ylabel("Component B2")
for i,filename in enumerate(filenames_S):
    color_index = int(filename[0])-1
    #filename = filename[:-4]
    sample_id = filename[3:-4]
    axs[1].scatter(transf_spec_S[i,0],transf_spec_S[i,1],c=colors[color_index])
    axs[1].text(transf_spec_S[i,0],transf_spec_S[i,1],sample_id,fontsize='xx-small')
axs[1].set_xlabel("Component S1")
axs[1].set_ylabel("Component S2")
if cluster:
    plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_PCA/PCA_distribution_clustering_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
else:
    plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_PCA/PCA_distribution_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
plt.show()

comp_B = pca_B.components_
comp_S = pca_S.components_
fig,axs = plt.subplots(2,sharex=True)
fig.tight_layout()
colors = ['r','b']
for i in range(pca_components):
    axs[0].plot(wns[wn_start_arg:wn_end_arg],np.abs(comp_B[i]),c=colors[i])
    axs[1].plot(wns[wn_start_arg:wn_end_arg],np.abs(comp_S[i]),c=colors[i])
plt.xlim((4000,400))
plt.xlabel("Wavenumber")
plt.ylabel("Component Magnitude")
plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_PCA/PCA_transform_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
plt.show()
print()


# In[ ]:





# In[ ]:





# In[ ]:




