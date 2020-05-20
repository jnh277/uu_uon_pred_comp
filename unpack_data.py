# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:51:43 2018

@author: andsv164
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import progressbar

# load data
data = pd.read_csv("./data/ucdp_month_priogrid.csv")

#%%
def retrieve_coord(pg_id):
    xt = pg_id-129961
    lat2 = np.round(xt/720)
    long2 = xt-720*lat2
    lat = lat2/2
    long = long2/2
    return np.array((lat, long))

pg_ids = data.loc[(data['month_id'] == 109)]['pg_id']
npg_id = len(pg_ids)

coords = np.empty((npg_id,2))

for i in range(0,npg_id):
    coords[i,:] = retrieve_coord(pg_ids.iloc[i])
#%%
    
#lats = np.unique(coords[:,0])
#longs = np.unique(coords[:,1])

cols = ['C','NW','N','NE','W','E','SW','S','SE']
neighbors = pd.DataFrame(columns=cols)

pb = progressbar.ProgressBar(max_value=npg_id*2)

# for a given gridpoint, find its neighbors
for i in range(0,npg_id):
    coordi = coords[i,:]
    
    #1 northwest neighbor
    nwi = np.where((coords[:,0]==(coordi[0]+.5))&(coords[:,1]==(coordi[1]-.5)))
    if len(nwi[0])>0:
        nw = pg_ids.iloc[nwi[0][0]]
    else:
        nw = np.NaN
    
    #2 north neighbor
    ni = np.where((coords[:,0]==(coordi[0]+.5))&(coords[:,1]==(coordi[1])))
    if len(ni[0])>0:
        n = pg_ids.iloc[ni[0][0]]
    else:
        n = np.NaN
    
    #3 northeast neighbor
    nei = np.where((coords[:,0]==(coordi[0]+.5))&(coords[:,1]==(coordi[1]+.5)))
    if len(nei[0])>0:
        ne = pg_ids.iloc[nei[0][0]]
    else:
        ne = np.NaN
        
    #4 west neighbor
    wi = np.where((coords[:,0]==(coordi[0]))&(coords[:,1]==(coordi[1]-.5)))
    if len(wi[0])>0:
        w = pg_ids.iloc[wi[0][0]]
    else:
        w = np.NaN
        
    #5 east neighbor
    ei = np.where((coords[:,0]==(coordi[0]))&(coords[:,1]==(coordi[1]+.5)))
    if len(ei[0])>0:
        e = pg_ids.iloc[ei[0][0]]
    else:
        e = np.NaN
        
    #6 southwest neighbor
    swi = np.where((coords[:,0]==(coordi[0]-.5))&(coords[:,1]==(coordi[1]-.5)))
    if len(swi[0])>0:
        sw = pg_ids.iloc[swi[0][0]]
    else:
        sw = np.NaN
        
    #7 south neighbor
    si = np.where((coords[:,0]==(coordi[0]-.5))&(coords[:,1]==(coordi[1])))
    if len(si[0])>0:
        s = pg_ids.iloc[si[0][0]]
    else:
        s = np.NaN
        
    #8 southeast neighbor
    sei = np.where((coords[:,0]==(coordi[0]-.5))&(coords[:,1]==(coordi[1])+.5))
    if len(sei[0])>0:
        se = pg_ids.iloc[sei[0][0]]
    else:
        se = np.NaN
        
    neighbors = neighbors.append(pd.Series(data=[pg_ids.iloc[i],nw,n,ne,w,e,sw,s,se],index=cols,name=str(i)))
    
    pb.update(i)


#%% read out the data

sb_best_neighbors_mat = np.zeros([npg_id,357,9])
for i in range(0,npg_id):
    sb_best_neighbors_mat[i,:,0] = data.loc[(data['pg_id'].values == neighbors['C' ][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['NW'][i]):
        sb_best_neighbors_mat[i,:,1] = data.loc[(data['pg_id'].values == neighbors['NW'][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['N'][i]):  
        sb_best_neighbors_mat[i,:,2] = data.loc[(data['pg_id'].values == neighbors['N' ][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['NE'][i]):
        sb_best_neighbors_mat[i,:,3] = data.loc[(data['pg_id'].values == neighbors['NE'][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['W'][i]):
        sb_best_neighbors_mat[i,:,4] = data.loc[(data['pg_id'].values == neighbors['W' ][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['E'][i]):
        sb_best_neighbors_mat[i,:,5] = data.loc[(data['pg_id'].values == neighbors['E' ][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['SW'][i]):
        sb_best_neighbors_mat[i,:,6] = data.loc[(data['pg_id'].values == neighbors['SW'][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['S'][i]):
        sb_best_neighbors_mat[i,:,7] = data.loc[(data['pg_id'].values == neighbors['S' ][i])]['ged_best_sb'].values
    if not np.isnan(neighbors['SE'][i]):
        sb_best_neighbors_mat[i,:,8] = data.loc[(data['pg_id'].values == neighbors['SE'][i])]['ged_best_sb'].values
        
    pb.update(i+npg_id)
#%%
np.save('./data/sb_best_neighbors_mat',sb_best_neighbors_mat)
np.save('./data/coords',coords)

