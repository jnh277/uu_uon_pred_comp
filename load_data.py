import numpy as np
import matplotlib.pyplot as plt

#%% Load data

sb_best_neighbors_mat = np.load('./data/sb_best_neighbors_mat.npy')
population = np.load('./data/population.npy')
ln_capdist = np.load('./data/ln_capdist.npy')
ln_bdist3 = np.load('./data/ln_bdist3.npy')
coords = np.load('./data/coords.npy')
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