import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstat

#Initialise variables
directory = "FTIR Spelt Samples/" #"FTIR Barley Samples/"
groupnames = ['1S', '2S', '3S', '4S'] #['1B','2B','3B', '4B']
colors = ['r', 'g', 'b', 'k']
compare = [0,3]
sample_num_threshold = 15 #how many samples to use from each group. Wavenumbers with fewer than this are excluded

#Determine number of samples to use per group, so that number is same across all groups
min_num_samples = 1e10
hc_min_wn = 0
lc_max_wn = 1e10
wn_step = None
for group_index,groupname in enumerate(groupnames):
    #List files in group
    group_filenames = os.listdir(directory+groupname+'/')

    #Select CSV files
    selected_filenames = []
    for filename in group_filenames:
        if filename[-3:] == "CSV":
            selected_filenames.append(filename)

    #Find min number of spectra in a group
    num = len(selected_filenames)
    if num < min_num_samples:
        min_num_samples = num

print("[INFO] Number of samples per group to consider:", min_num_samples)

#Read in data
all_spec = []
for group_index,groupname in enumerate(groupnames):
    group_spec = []
    group_filenames = os.listdir(directory+groupname+'/')

    selected_filenames = []
    for filename in group_filenames:
        if filename[-3:] == "CSV":
            selected_filenames.append(filename)

    for file_index,filename in enumerate(selected_filenames):
        file_data = []
        with open(directory+groupname+'/'+filename,'r') as f:
            raw = f.readlines()
        for line in raw:
            line = line.replace('\n','')
            line = line.split(',')
            file_data.append(line)
        file_data = np.array(file_data,dtype=np.float64)
        group_spec.append(file_data)
    all_spec.append(group_spec)
print("[INFO] Finished reading in data.")

#Create array of wavenumbers
all_wns = []
for i in range(len(all_spec)):
    group_spec = all_spec[i]
    for j in range(len(group_spec)):
        all_wns = all_wns + list(group_spec[j][:,0])
wns = np.unique(all_wns)
print("[INFO] Created array of wavenumbers.")

#Create arrays of intensities
all_ints = np.zeros((wns.shape[0],len(groupnames),min_num_samples))-1
for i,wn in enumerate(wns):
    for j in range(len(groupnames)):
        int_vals = []
        for dataset in all_spec[j][:min_num_samples]:
            dataset = np.array(dataset,dtype=np.float64)
            wn_index = np.argwhere(np.equal(dataset[:,0],wn))
            if wn_index.size != 0:
                int_vals.append(dataset[wn_index[0,0],1])
        all_ints[i,j,0:len(int_vals)] = np.array(int_vals,dtype=np.float64)
missing_data = ((np.argwhere(np.equal(all_ints,-1))).shape[0]/(all_ints.size))*100
print("[INFO] Created homogeneous array of intensity data. Amount of missing data was %s percent."%(str(missing_data)))

##Compute mean and standard deviation of selected groups
#Select wavenumbers where both have intensity values
num_sample_ints = np.sum(np.logical_not(np.equal(all_ints,-1)),axis=2)
compare_wn_indices = np.argwhere(np.logical_and(num_sample_ints[:,compare[0]]>=sample_num_threshold,num_sample_ints[:,compare[1]]>=sample_num_threshold))
compare_wns = np.zeros((compare_wn_indices.shape[0]))
compare_means = np.zeros((compare_wn_indices.shape[0],2))
compare_std = np.zeros((compare_wn_indices.shape[0],2))
for i in range(compare_wn_indices.shape[0]):
    compare_wns[i] = wns[compare_wn_indices[i]]
    for j,g in enumerate(compare):
        compare_means[i,j] = np.mean(all_ints[compare_wn_indices[i],g,:sample_num_threshold])
        compare_std[i,j] = np.std(all_ints[compare_wn_indices[i],g,:sample_num_threshold])
print("[INFO] Computed mean and standard deviation at %s wavenumber points (%s percent of total) with %s samples per point"%(str(compare_wn_indices.shape[0]),str(100*(compare_wn_indices.shape[0]/wns.shape[0])),str(sample_num_threshold)))

#Set up plot
fig, axs = plt.subplots(2,sharex=True)

#Plot mean and std dev of chosen groups
for i,g in enumerate(compare):
    axs[0].plot(compare_wns,compare_means[:,i],c=colors[g],label=groupnames[g])
    axs[0].fill_between(compare_wns,compare_means[:,i]-compare_std[:,i],compare_means[:,i]+compare_std[:,i],alpha=0.5,color=colors[g])
axs[0].legend()

p_values = np.zeros((compare_wn_indices.shape[0]))
for i in range(compare_wn_indices.shape[0]):
    #p_values[i] = spstat.ttest_ind(all_intensities[ttest_index1][:,i], all_intensities[ttest_index2][:,i], axis=0, equal_var=False, nan_policy='propagate', alternative='two-sided', trim=0).pvalue
    t_val = (np.sqrt(sample_num_threshold)*np.abs(compare_means[i,0]-compare_means[i,1]))/np.mean(np.array([compare_std[i,0],compare_std[i,1]]))
    p_values[i] = 1-spstat.t.cdf(t_val,sample_num_threshold-2)

axs[1].plot(compare_wns,p_values)
axs[1].set_xlabel(r"Wavenumber [cm$^{-1}$]")
axs[0].set_ylabel("FTIR Intensity")
axs[1].set_ylabel("P-value")

confidence_level = 0.05
axs[1].plot([compare_wns[0],compare_wns[-1]],[confidence_level,confidence_level],c='r',linestyle='--')

plt.xlim((4000,400))


#plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/3S_4S.png",dpi=300,bbox_inches='tight')
plt.show()



