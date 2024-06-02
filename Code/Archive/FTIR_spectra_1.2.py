import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstat

#Initialise variables
directory = "spectra" #must contain only homogeneous TXT comma-separated files
colors = ['r', 'g']
compare = ['1B','4B'] #specify prefixes of groups to compare

#Read in data
filenames =  os.listdir(directory)
wns = []
all_spec = []
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
    all_spec.append(file_data)
    if not(finished_wns):
        finished_wns = True
wns = np.array(wns,dtype=np.float64)
print("[INFO] Finished reading in data.")

##Compute mean and standard deviation of selected groups
means = np.zeros((wns.shape[0],2))
stds = np.zeros((wns.shape[0],2))
N = np.zeros((2))
for m,compare_prefix in enumerate(compare):
    sets = []
    set_names = []
    for i,name in enumerate(filenames):
        if name[:len(compare_prefix)] == compare_prefix:
            sets.append(all_spec[i])
            set_names.append(name)
    print("Data for %s will be computed from %s"%(str(compare_prefix),str(set_names)))
    sets = np.array(sets)
    means[:,m] = np.mean(sets,axis=0)
    stds[:,m] = np.std(sets,axis=0)
    N[m] = len(set_names)
print("[INFO] Computed mean and standard deviation.")

#Set up plot
fig, axs = plt.subplots(2,sharex=True)

#Plot mean and std dev of chosen groups
for i,g in enumerate(compare):
    axs[0].plot(wns,means[:,i],c=colors[i],label=g)
    axs[0].fill_between(wns,means[:,i]-stds[:,i],means[:,i]+stds[:,i],alpha=0.5,color=colors[i])
axs[0].legend()

p_values = np.zeros((wns.shape[0]))
for i in range(wns.shape[0]):
    t_val = (np.sqrt(np.mean(N))*np.abs(means[i,0]-means[i,1]))/np.mean(np.array([stds[i,0],stds[i,1]]))
    p_values[i] = 1-spstat.t.cdf(t_val,np.mean(N)-2)

axs[1].plot(wns,p_values)
axs[1].set_xlabel(r"Wavenumber [cm$^{-1}$]")
axs[0].set_ylabel("FTIR Intensity")
axs[1].set_ylabel("P-value")

confidence_level = 0.05
axs[1].plot([wns[0],wns[-1]],[confidence_level,confidence_level],c='r',linestyle='--')

plt.xlim((4000,400))


#plt.savefig("/Users/daniellesicotte/Library/Mobile Documents/com~apple~CloudDocs/Cambridge/MPhil Thesis/DATA/FTIR/3S_4S.png",dpi=300,bbox_inches='tight')
plt.show()

###Proposed Extensions
#Filter anomalies
#Plot individual samples
#PCA
