
import matplotlib.pyplot as plt
import numpy as np
from model5_predict_function import predict_by_particle_filter_one
from matplotlib import gridspec
from scipy.special import ndtri

#%% Load data

sb_best_neighbors_mat = np.load('sb_best_neighbors_mat.npy')
population = np.load('population.npy')
ln_capdist = np.load('ln_capdist.npy')
ln_bdist3 = np.load('ln_bdist3.npy')
coords = np.load('coords.npy')
npg_id = sb_best_neighbors_mat.shape[0]
Tm = sb_best_neighbors_mat.shape[1]

#%%
np.random.seed(28) #28 #1863 #11 #19

nonzeros, = np.nonzero(np.sum(sb_best_neighbors_mat[:,:,0],axis=1))

ind = np.random.choice(nonzeros,size=1,replace=False)

y = sb_best_neighbors_mat[ind,:,0].T
ua = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[2],[4],[5],[7]]]+1),axis=0).T-0.02)/0.26 #adjacent, transformed to have mean 0 etc
ud = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[1],[3],[6],[8]]]+1),axis=0).T-0.02)/0.26 #diagonal
up = (np.log(population[0:357,ind]+1)-9.44)/2.28
uc = (ln_capdist[0:357,ind]-6.22)/0.79
ub = (ln_bdist3[0:357,ind]-4.41)/1.19
u = np.dstack((ua,ud,up,uc,ub))

plt.plot(y)

#%% predict
#np.random.seed(1)
a = .999
b = .995
s2e = np.exp(-1.3)
c = np.array([ 1.17923074e-05, -8.29116770e-06,  4.30480663e-04,  4.75196731e-05,  -8.48756895e-05])
d = np.array([ 4.50801929e-05, -4.49146198e-05,  8.55532456e-04,  6.11445942e-04,  -4.46232983e-04])
alpha = 2.3
beta = 0.005

#predict y(t) using particle filter
np.random.seed(0)
xm,xP,s,ypred,ypred_mean = predict_by_particle_filter_one((a,b,c,d,s2e,alpha,beta),y,u,50000)


#%%


fig = plt.figure(figsize=(20,12))
gs = gridspec.GridSpec(3, 2, width_ratios=[3, 2]) 
ax = plt.subplot(gs[0])
#for q in np.arange(0.05,0.5,0.05):
#    u = np.quantile(ypred,1-q,axis=1)
#    l = np.quantile(ypred,q,axis=1)
#    ax.fill_between(x=np.arange(1989,2018.8,1/12),y1=u,y2=l,alpha=0.2,facecolor='red')
ax.plot(np.arange(1989,2018.7,1/12),y,'k',linewidth=2)
ax.set_ylim([0,1.1*np.max(y)])
ax.set_xlim([1990,2018])
ax.set_title('Data')

ax = plt.subplot(gs[1])
txt = ax.text(0, 0, 'Risk of violence (death > 0) next month: ' + str(sum(ypred[-2,:]>0)/ypred[-2,:].size),fontsize=15)
ax.axis('off')

ax = plt.subplot(gs[3])
ax.hist(ypred[-2,:],range=(1,np.max(ypred[-2,:])),orientation="horizontal",color='red',density=True)
ax.set_title('Nonzero predictions for next month')

ax3 = plt.subplot(gs[2])
xmm = np.mean(xm,axis=1)
xPm = np.mean(xP,axis=1)
for q in np.arange(0.05,0.5,0.05):
    u = xmm+ndtri((1-2*q))*3*np.sqrt(xPm)
    l = xmm-ndtri((1-2*q))*3*np.sqrt(xPm)
    ax3.fill_between(x=np.arange(1989,2018.8,1/12),y1=l,y2=u,alpha=0.2,facecolor='red')
ax3.set_xlim([1990,2018])
ax3.set_title('Inferred state 1')

ax2 = plt.subplot(gs[4])
for q in np.arange(0.05,0.5,0.05):
    u = np.quantile(s,1-q,axis=1)
    l = np.quantile(s,q,axis=1)
    ax2.fill_between(x=np.arange(1989,2018.8,1/12),y1=l,y2=u,alpha=0.2,facecolor='red')
ax2.set_xlim([1990,2018])
ax2.set_title('Inferred state 2')
