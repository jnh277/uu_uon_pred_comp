
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import truncnorm
#from scipy.special import expit
import progressbar as pb


#%% Load data

y = np.load('sb_best_neighbors_mat.npy')
coords = np.load('coords.npy')
npg_id = y.shape[0]
Tm = y.shape[1]

#%%


fig = plt.figure(figsize=(20,17))
ax = plt.subplot(111)
ypl = np.copy(y[:,-1,0])
y_scale = np.log(1+ypl)
pt = np.array([0,1,5,10,20,30,60])
ptt = np.log(pt+1)
y_scale[6801] = 100
plt.scatter(coords[:,1],coords[:,0],c=y_scale,s=27,marker='s',alpha=1,cmap='rainbow',vmin=ptt[0],vmax=ptt[-1])
ax.set_xlim((-20,53))
ax.set_ylim((-37,39))
cbar = plt.colorbar(ticks=ptt)
cbar.ax.set_yticklabels(['0','1','5','10','20','30','60'])
plt.title('State-based conflict fatalities; 2018-9')

#%%

# somalia = 8
# algeria = 15


np.random.seed(23)
ynz = np.sum(y[:,:,0],axis=1)
ynzi= np.nonzero(ynz)
pid = np.random.choice(ynzi[0],1)
pid = pid[0]


fig = plt.figure(figsize=(20,3))
ax = plt.subplot(111)
plt.plot(y[pid,:,0])
plt.xlim([0,356])
plt.xticks(np.arange(12,350,12*5),labels=('1990','1995','2000','2005','2010','2015'))
plt.show()
#
fig = plt.figure(figsize=(20,17))
ax = plt.subplot(111)
yi = np.zeros(y_scale.shape)
yi[pid] = 100
plt.scatter(coords[:,1],coords[:,0],c=yi,s=27,marker='s',alpha=1,cmap='rainbow',vmin=ptt[0],vmax=ptt[-1])

cbar = plt.colorbar(ticks=ptt)
cbar.ax.set_yticklabels(['0','1','5','10','20','30','60'])