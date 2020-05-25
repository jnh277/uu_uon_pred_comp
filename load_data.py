import numpy as np
import matplotlib.pyplot as plt
import pystan
from helpers import plot_trace

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

# sb_best_neighbours has [regions, time, surrounding 3x3 grid centered on region]
# from teh surrounding 0 is the region, {2,4,5,7} are the adjacent and {1,3,5,8} are the diagonals

ind = np.random.choice(nonzeros,size=1,replace=False)

y = sb_best_neighbors_mat[ind,:,0]
ua = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[2],[4],[5],[7]]]+1),axis=0)-0.02)/0.26 #adjacent, transformed to have mean 0 etc
ud = (np.sum(np.log(sb_best_neighbors_mat[ind,:,[[1],[3],[6],[8]]]+1),axis=0)-0.02)/0.26 #diagonal
up = (np.log(population[0:357,ind].T+1)-9.44)/2.28
uc = (ln_capdist[0:357,ind].T-6.22)/0.79
ub = (ln_bdist3[0:357,ind].T-4.41)/1.19

# define maximum polynomial coefficient
max_poly = 2
num_inputs = 5


# build u matrix
u = np.zeros((num_inputs*max_poly,Tm))
for i in range(max_poly):
    u[i*num_inputs,:] = np.power( ua,i+1)
    u[i*num_inputs+1,:] = np.power(ud,i+1)
    u[i*num_inputs+2,:] = np.power(up,i+1)
    u[i*num_inputs+3,:] = np.power(uc,i+1)
    u[i*num_inputs+4,:] = np.power(ub,i+1)

def init_function():
    output = dict(
                  # a=0.999,
                  b=0.995,
                  sig_e=np.sqrt(0.27),
                  x=np.log(y[0,:]+1)/2.3,
                  s=np.zeros((Tm + 1))
                  )
    return output

model = pystan.StanModel(file='stan/views.stan')

stan_data = {'no_obs':Tm,
             'max_poly':max_poly,
             'num_inputs':num_inputs,
             'u':u,
             'y':y[0,:],
             'beta':0.005,
             'alpha':2.3,
             'b':0.995
             }

fit = model.sampling(data=stan_data, iter=2000, chains=4)

traces = fit.extract()
a = traces['a']
# b = traces['b']
sig_e = traces['sig_e']
lam = traces['lambda']
s = traces['s']
x = traces['x']
# c = traces['c']
# d = traces['d']


plt.plot(np.mean(s,axis=0))
plt.show()


plot_trace(a, 2, 1, 'a')
plot_trace(sig_e, 2, 2, 'sig_e')
plt.show()
#
# plt.subplot(2,5,1)
# plt.hist(c[:,0])
# plt.title('Coeff of ua')
#
# plt.subplot(2,5,2)
# plt.hist(c[:,1])
# plt.title('Coeff of ud')
#
# plt.subplot(2,5,3)
# plt.hist(c[:,2])
# plt.title('Coeff of up')
#
# plt.subplot(2,5,4)
# plt.hist(c[:,3])
# plt.title('Coeff of uc')
#
# plt.subplot(2,5,5)
# plt.hist(c[:,4])
# plt.title('Coeff of ub')
#
# plt.subplot(2,5,6)
# plt.hist(c[:,5])
# plt.title('Coeff of ua^2')
#
# plt.subplot(2,5,7)
# plt.hist(c[:,6])
# plt.title('Coeff of ud^2')
#
# plt.subplot(2,5,8)
# plt.hist(c[:,7])
# plt.title('Coeff of up^2')
#
# plt.subplot(2,5,9)
# plt.hist(c[:,8])
# plt.title('Coeff of uc^2')
#
# plt.subplot(2,5,10)
# plt.hist(c[:,9])
# plt.title('Coeff of ub^2')
# plt.show()

# predicted log ys verse real log ys
threshed = np.zeros(np.shape(lam[:,1:]))
threshed[lam[:,1:] >=0.5] = 1

log_pred = np.mean(threshed * np.maximum(x[:,1:], 0),axis=0)

plt.plot(np.log(y[0,:]+1)/2.3)
plt.plot(np.mean(np.maximum(x[:,1:], 0),axis=0))
plt.plot(log_pred)
plt.title('Estimated')
plt.ylabel('log(y+1)/alpha')
plt.legend(('true','x','x with thresholded probabilities'))
plt.show()

## one step ahead prediction
x_p1 = traces['x_p1']
lam_p1 = traces['lambda_p1']
threshed_p1 = np.zeros(np.shape(lam_p1))
threshed_p1[lam_p1 >=0.5] = 1
log_pred_p1 = np.mean(threshed_p1 * np.maximum(x_p1, 0),axis=0)


plt.plot(np.log(y[0,:]+1)/2.3)
plt.plot(np.mean(np.maximum(x_p1, 0),axis=0))
plt.plot(log_pred_p1)
plt.title('1 step forecast')
plt.ylabel('log(y+1)/alpha')
plt.legend(('true','x','x with thresholded probabilities'))
plt.show()

