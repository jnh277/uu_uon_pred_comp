import numpy as np
import matplotlib.pyplot as plt
import pystan
from helpers import plot_trace

#%% Load data

# define some things
num_countries = 5
max_poly = 4
num_inputs = 5

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

# sb_best_neighbours has [regions, time, surrounding 3x3 grid centered on region]
# from teh surrounding 0 is the region, {2,4,5,7} are the adjacent and {1,3,5,8} are the diagonals


ind = np.random.choice(nonzeros,size=num_countries,replace=False)

y = sb_best_neighbors_mat[ind,:,0]
ua = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[2],[4],[5],[7]]]+1),axis=0)-0.02)/0.26 #adjacent, transformed to have mean 0 etc
ud = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[1],[3],[6],[8]]]+1),axis=0)-0.02)/0.26 #diagonal
up = (np.log(population[0:357,ind].T+1)-9.44)/2.28
uc = (ln_capdist[0:357,ind].T-6.22)/0.79
ub = (ln_bdist3[0:357,ind].T-4.41)/1.19

# build u matrix
# u = np.zeros((num_inputs*max_poly,Tm))

u = np.zeros((Tm,num_countries,num_inputs*max_poly))
for i in range(Tm):
    for j in range(max_poly):
        u[i,:,j*num_inputs] = np.power(ua[:,i],j+1)
        u[i,:,j* num_inputs + 1] = np.power(ud[:,i], j + 1)
        u[i,:,j* num_inputs + 2] = np.power(up[:,i], j + 1)
        u[i,:,j* num_inputs + 3] = np.power(uc[:,i], j + 1)
        u[i,:,j* num_inputs + 4] = np.power(ub[:,i], j + 1)


def init_function():
    output = dict(
                  # a=0.999,
                  # b=0.995,
                  # sig_e=np.sqrt(0.27),
                  x=np.log(y+1)/2.3,
                  s=np.zeros((num_countries,Tm + 1))
                  )
    return output

model = pystan.StanModel(file='stan/views_mc.stan')

stan_data = {'no_obs':Tm,
             'max_poly':max_poly,
             'num_inputs':num_inputs,
             'num_countries':num_countries,
             'u':u,
             'y':y,
             'beta':0.005,
             'alpha':2.3,
             'b':0.995,
             'a':0.999,
             'sig_e':np.sqrt(0.27)
             }

fit = model.sampling(data=stan_data, iter=2000, chains=4)


