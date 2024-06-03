#!/usr/bin/env python
# coding: utf-8

# In[65]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.optimize as scpo
import pandas as pd
import math
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#Initialise variables
# GENERAL ########################################################################
directory = "/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_Homogenized/" #must contain only homogeneous TXT comma-separated files
pca_components = 2 #number of PCA components to use
wn_start = 1000
wn_end = 3400
colorscale = mpl.colormaps['rainbow'].resampled(40) #color scale for categorising scatter points

# PCA PLOTTING ###################################################################
cpn = 5000 #density of plotting grid for kmean clustering
plot_pca = True #plot PCA component-space distribution
plot_cluster = True #whether or not to plot clusters
plot_text_labels = False

# MORPH VS PCA PLOTTING ##########################################################
plot_morph_vs_pca = True #plot morphological characteristic against first PCA component
morph_category = 'cracks' #plot PCA first component vs this morphological property
mvp_points = True #morph_vs_pca: show individual points
mvp_boxplots = True #morph_vs_pca: show boxplots
mvp_funct_fit = False
def mvp_funct_B(x,m,c): #function to fit to barley mvp data
    return m*x+c
mvp_funct_initial_params_B = [-0.5,2.5]
def mvp_funct_S(x,m,c): #function to fit to spelt mvp data
    return m*x+c
mvp_funct_initial_params_S = [0.5,3]

# MORPH PCA ######################################################################
plot_morph_pca = True #carry out PCA analysis of the morphological score
morph_pca_cats = ['luster', 'glume impressions', 'cracks', 'endosperm expansion', 'VF visibility', 'shape'] #categories to include in morphological PCA analysis
morph_pca_components_B = 4 #numher of components to output in PCA fitting
morph_pca_components_S = 4
chem_pca_plot_component = 0 #which chemical PCA component to plot
morph_pca_plot_component = 0 #which morph PCA component to plot

##################################################################################

#Read in spectrum data
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
print("[INFO] Finished reading in spectrum data.")

#Read in morphology data
morph = pd.read_excel("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/morphology.xlsx",sheet_name="Barley",header=0,index_col=0)

##Get morph sum values
#Read in and store
morph_sum_B = np.zeros(len(filenames_B))
morph_sum_S = np.zeros(len(filenames_S))
for i,filename in enumerate(filenames_B):
    morph_sum_B[i] = morph.loc[filename[:-4]]['SUM']
for i,filename in enumerate(filenames_S):
    morph_sum_S[i] = morph.loc[filename[:-4]]['SUM']
#Renormalise
morph_sum_B = (morph_sum_B-np.min(morph_sum_B))/(np.max(morph_sum_B)-np.min(morph_sum_B))
morph_sum_S = (morph_sum_S-np.min(morph_sum_S))/(np.max(morph_sum_S)-np.min(morph_sum_S))

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
if plot_cluster:
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
if plot_pca:
    fig,axs = plt.subplots(2)
    if plot_cluster:
        axs[0].imshow(km_B_z,interpolation="nearest",extent=(ts_B_xx.min(), ts_B_xx.max(), ts_B_yy.min(), ts_B_yy.max()),cmap="Greens",aspect="auto",origin="lower")
        axs[1].imshow(km_S_z,interpolation="nearest",extent=(ts_S_xx.min(), ts_S_xx.max(), ts_S_yy.min(), ts_S_yy.max()),cmap="Greens",aspect="auto",origin="lower")
    fig.tight_layout()
    colors = ['limegreen','greenyellow','yellow','sandybrown','firebrick']
    for i,filename in enumerate(filenames_B):
        color_index = int(filename[0])-1
        #filename = filename[:-4]
        sample_id = filename[3:-4]
        axs[0].scatter(transf_spec_B[i,0],transf_spec_B[i,1],c=colorscale(morph_sum_B[i]))#colors[color_index])
        if plot_text_labels: axs[0].text(transf_spec_B[i,0],transf_spec_B[i,1],sample_id,fontsize='xx-small')
    axs[0].set_xlabel("Component B1")
    axs[0].set_ylabel("Component B2")
    for i,filename in enumerate(filenames_S):
        color_index = int(filename[0])-1
        #filename = filename[:-4]
        sample_id = filename[3:-4]
        axs[1].scatter(transf_spec_S[i,0],transf_spec_S[i,1],c=colorscale(morph_sum_S[i]))#c=colors[color_index])
        if plot_text_labels: axs[1].text(transf_spec_S[i,0],transf_spec_S[i,1],sample_id,fontsize='xx-small')
    axs[1].set_xlabel("Component S1")
    axs[1].set_ylabel("Component S2")
    if plot_cluster:
        plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_spec_PCA/PCA_distribution_clustering_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
    else:
        plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_spec_PCA/PCA_distribution_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
    plt.show()

    comp_B = pca_B.components_
    comp_S = pca_S.components_
    fig,axs = plt.subplots(2,sharex=True)
    fig.tight_layout()
    colors = ['r','b']
    for i in range(pca_components):
        axs[0].plot(wns[wn_start_arg:wn_end_arg],comp_B[i],c=colors[i])
        axs[1].plot(wns[wn_start_arg:wn_end_arg],comp_S[i],c=colors[i])
    plt.xlim((4000,400))
    plt.xlabel("Wavenumber [1/cm]")
    axs[0].set_ylabel("Component Magnitude_B")
    axs[1].set_ylabel("Component Magnitude_S")
    plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_spec_PCA/PCA_transform_%s_%s.png"%(str(wn_start),str(wn_end)), dpi=300, bbox_inches='tight')
    plt.show()

if plot_morph_vs_pca:
    fig,axs = plt.subplots(2)
    fig.tight_layout()
    colors = ['limegreen','greenyellow','yellow','sandybrown','firebrick']
    mvp_B_x = []
    mvp_B_y = []
    mvp_S_x = []
    mvp_S_y = []
    sorted_pca_vals_B = [[],[],[],[],[]]
    for i in range(len(filenames_B)):
        pca_first_val = transf_spec_B[i,0]
        morph_val = morph.loc[filenames_B[i][:-4]][morph_category]
        sorted_pca_vals_B[morph_val-1].append(pca_first_val)
        mvp_B_x.append(pca_first_val)
        mvp_B_y.append(morph_val)
        if mvp_points:
            axs[0].scatter(pca_first_val,morph_val,c=colorscale(morph_sum_B[i])) #c=colors[int(filenames_B[i][0])-1])
    sorted_pca_vals_S = [[],[],[],[],[]]
    for i in range(len(filenames_S)):
        pca_first_val = transf_spec_S[i,0]
        morph_val = morph.loc[filenames_S[i][:-4]][morph_category]
        sorted_pca_vals_S[morph_val-1].append(pca_first_val)
        mvp_S_x.append(pca_first_val)
        mvp_S_y.append(morph_val)
        if mvp_points:
            axs[1].scatter(pca_first_val,morph_val,c=colorscale(morph_sum_S[i]))#c=colors[int(filenames_S[i][0])-1])
    axs[0].set_xlabel("Component B1")
    axs[1].set_xlabel("Component S1")
    axs[0].set_ylabel("B_%s"%(morph_category.capitalize()))
    axs[1].set_ylabel("S_%s"%(morph_category.capitalize()))
    if mvp_boxplots:
        axs[0].boxplot(sorted_pca_vals_B,vert=False)
        axs[1].boxplot(sorted_pca_vals_S,vert=False)

    if mvp_funct_fit:
        #Ensure each group contains equal numbers of points
        group_nums_B = []
        for sublist in sorted_pca_vals_B:
            group_nums_B.append(len(sublist))
        lcm_B = math.lcm(*group_nums_B)
        mvp_B_x = []
        mvp_B_y = []
        for i,sublist in enumerate(sorted_pca_vals_B):
            mvp_B_x += (int(lcm_B/len(sublist)))*sublist
            mvp_B_y += lcm_B*[i+1]

        group_nums_S = []
        for sublist in sorted_pca_vals_S:
            group_nums_S.append(len(sublist))
        lcm_S = math.lcm(*group_nums_S)
        mvp_S_x = []
        mvp_S_y = []
        for i,sublist in enumerate(sorted_pca_vals_S):
            mvp_S_x += (int(lcm_S/len(sublist)))*sublist
            mvp_S_y += lcm_S*[i+1]

        #Perform optimisation to fit data
        popt_B, pcov_B, ignore, msg_B, flag_B = scpo.curve_fit(mvp_funct_B,mvp_B_x,mvp_B_y,p0=mvp_funct_initial_params_B,full_output=True)
        perr_B = np.sqrt((lcm_B/np.min(np.array(group_nums_B)))*np.diag(pcov_B))
        popt_S, pcov_S, ignore, msg_S, flag_S = scpo.curve_fit(mvp_funct_S,mvp_S_x,mvp_S_y,p0=mvp_funct_initial_params_S,full_output=True)
        perr_S = np.sqrt((lcm_S/np.min(np.array(group_nums_S)))*np.diag(pcov_S))

        #Print optimisation results
        print('Curve fitting results for barley:')
        print('Message:', msg_B)
        print('Flag:', flag_B)
        print()
        print('Curve fitting results for spelt:')
        print('Message:', msg_S)
        print('Flag:', flag_S)

        #Get optimum curve data
        mvp_B_x_opt = np.linspace(np.min(mvp_B_x),np.max(mvp_B_x),100)
        mvp_S_x_opt = np.linspace(np.min(mvp_S_x),np.max(mvp_S_x),100)
        mvp_B_y_opt = mvp_funct_B(mvp_B_x_opt,*popt_B)
        mvp_S_y_opt = mvp_funct_S(mvp_S_x_opt,*popt_S)

        #Get standard deviation data
        pupper_B = list(np.array(popt_B)+np.array(perr_B))
        plower_B = list(np.array(popt_B)-np.array(perr_B))
        pupper_S = list(np.array(popt_S)+np.array(perr_S))
        plower_S = list(np.array(popt_S)-np.array(perr_S))
        mvp_B_y_upper = mvp_funct_B(mvp_B_x_opt,*pupper_B)
        mvp_B_y_lower = mvp_funct_B(mvp_B_x_opt,*plower_B)
        mvp_S_y_upper = mvp_funct_S(mvp_S_x_opt,*pupper_S)
        mvp_S_y_lower = mvp_funct_S(mvp_S_x_opt,*plower_S)

        #Plot optimisation results
        axs[0].plot(mvp_B_x_opt,mvp_B_y_opt,c='r',linestyle='--')
        axs[1].plot(mvp_S_x_opt,mvp_S_y_opt,c='r',linestyle='--')
        #axs[0].fill_between(mvp_B_x_opt,mvp_B_y_lower,mvp_B_y_upper,alpha=0.5,color='g')
        #axs[1].fill_between(mvp_S_x_opt,mvp_S_y_lower,mvp_S_y_upper,alpha=0.5,color='g')
        
    plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/PCA_1.1/PCA_morphology_comparison_%s.png"%(morph_category))
    plt.show()

if plot_morph_pca:
    ##Perform morph PCA
    #Construct array of morph scores
    morph_B_scores = np.zeros((len(filenames_B),len(morph_pca_cats)))
    morph_S_scores = np.zeros((len(filenames_S),len(morph_pca_cats)))
    for i,filename in enumerate(filenames_B):
        for j,catname in enumerate(morph_pca_cats):
            morph_B_scores[i,j] = morph.loc[filename[:-4]][catname]
    for i,filename in enumerate(filenames_S):
        for j,catname in enumerate(morph_pca_cats):
            morph_S_scores[i,j] = morph.loc[filename[:-4]][catname]

    #Perform PCA
    morph_PCA_B = PCA(n_components=morph_pca_components_B)
    morph_PCA_S = PCA(n_components=morph_pca_components_S)
    morph_PCA_B.fit(morph_B_scores)
    morph_PCA_S.fit(morph_S_scores)
    print("[INFO] Morph PCA fitting complete:")
    print("Explained variance for barley:", morph_PCA_B.explained_variance_ratio_)
    print("Explained variance for spelt:", morph_PCA_S.explained_variance_ratio_)
    morph_PCA_B_comps = morph_PCA_B.components_
    morph_PCA_S_comps = morph_PCA_S.components_
    morph_B_scores_transform = morph_PCA_B.transform(morph_B_scores)
    morph_S_scores_transform = morph_PCA_S.transform(morph_S_scores)

    #Plot morph PCA vs chem PCA
    fig,axs = plt.subplots(2,2)
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    axs[0,0].scatter(transf_spec_B[:,chem_pca_plot_component],morph_B_scores_transform[:,morph_pca_plot_component],c='b')
    axs[0,1].scatter(transf_spec_S[:,chem_pca_plot_component],morph_S_scores_transform[:,morph_pca_plot_component],c='g')
    axs[1,0].bar(np.arange(len(morph_pca_cats)),morph_PCA_B_comps[morph_pca_plot_component],color='b')
    axs[1,0].set_xticks(np.arange(len(morph_pca_cats)),labels=morph_pca_cats, rotation='vertical')
    axs[1,1].bar(np.arange(len(morph_pca_cats)),morph_PCA_S_comps[morph_pca_plot_component],color='g')
    axs[1,1].set_xticks(np.arange(len(morph_pca_cats)),labels=morph_pca_cats, rotation='vertical')
    axs[0,0].set_xlabel("B_Spectrum PCA #%s"%(str(chem_pca_plot_component+1)))
    axs[0,1].set_xlabel("S_Spectrum PCA #%s"%(str(chem_pca_plot_component+1)))
    axs[0,0].set_ylabel("B_Morph PCA #%s"%(str(morph_pca_plot_component+1)))
    axs[0,1].set_ylabel("S_Morph PCA #%s"%(str(morph_pca_plot_component+1)))
    axs[1,0].set_ylabel("B_Morph Comp Coeff")
    axs[1,1].set_ylabel("S_Morph Comp Coeff")

    plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/FTIR_morph_PCA/PCA_morphology_comparison_%s.png"%(morph_category), bbox_inches='tight')

    plt.show()


print()


# In[ ]:





# In[ ]:




