# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:17:20 2019

@author: andsv164
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from particle_filter_functions import systematic_resampling
from particle_filter_functions import ESS
import progressbar as pb

#There are two different particle filter functions for predictions below. The only difference (?) is what data they return.

def predict_by_particle_filter(th,y,u,N):
    
    Tm = y.shape[0]
    sg = y.shape[1]
    
    progress = pb.ProgressBar(max_value=sg)
    
    logyp1 = np.log(y+1)
    
    (a,b,c,d,s2e,alpha,beta) = th
        
    # parameters
    s2x = (1-a**2)*(1-s2e)
    s2s = 1-b**2
    
    cs2e = 1/np.sqrt(2*np.pi*s2e**2)
    
    ypred_mat_upper = np.zeros(shape=y.shape)
    ypred_mat_lower = np.zeros(shape=y.shape)
    ypred_mat_mean = np.zeros(shape=y.shape)
    ypred_mat_bool = np.zeros(shape=y.shape)
    
    margin = .1
    mind = int(np.rint(margin*N))
    
    p0d = lambda xmt, pred_std : norm.cdf((np.log(1.5)/a-xmt)/pred_std)
    
    for i in range(sg):
        progress.update(i)
        xm = np.zeros((Tm+1,N))
        xP = np.zeros((Tm+1,N))
        s = np.zeros((Tm+1,N))
        
        #initalize
        xm[0,:] = np.zeros(N)
        xP[0,:] = np.ones(N)-s2e
        s[0,:] = np.random.normal(size=N)
        
        w = np.ones(N)
        
        
        for t in range(0,Tm):
            if t == 0: #new data series, restart. Restart with 3 times as many positive s, take account in weight
                xm[t,:] = np.zeros(N)
                xP[t,:] = np.ones(N)-s2e
                Nh = int(np.round(N/4))
                bp = 1
                s[t,:Nh] = truncnorm.rvs(a=-np.inf,b=bp,size=Nh)
                s[t,Nh:] = truncnorm.rvs(a=bp,b=np.inf,size=3*Nh)
                
            #predict!
            pred_std = np.sqrt(s2e + xP[t,:])
            zp = 1*((beta*(np.exp(np.exp(s[t,:]-1)-1)-1))>np.random.uniform(size=N))
            if t > 5:
                yvp = np.maximum(np.round(np.exp(alpha*(xm[t,:]+pred_std*np.random.normal(size=N)))-1),0)
                yp = zp[prediction_resampling]*yvp[prediction_resampling]
                ypred_mat_mean[t,i] = np.sum(w/np.sum(w)*np.minimum(np.maximum(0,(beta*(np.exp(np.exp(s[t,:]-1)-1)-1))),1)*np.maximum(np.exp(alpha*xm[t,:]+((alpha*pred_std)**2)/2)-1,0))
                yp.sort()
                ypred_mat_upper[t,i] = yp[-mind]
                ypred_mat_lower[t,i] = yp[mind]
                ypred_mat_bool[t,i] = np.mean(yp>0)
                
            #weighting
            if t == 0:
                z = zp
                w = w*(1*(z==0)*(y[t,i]==0)*cs2e + (z==1)*norm.pdf((logyp1[t,i]/alpha-xm[t,:])/pred_std)/pred_std)
                zp = z
            else:
                pred_std = np.sqrt(s2e + xP[t,:])
                if y[t,i]>0:
                    # weights if z = 1
                    z = np.ones(N)
                    w = w*(norm.pdf((logyp1[t,i]/alpha-xm[t,:])/pred_std)/pred_std)*np.minimum(np.maximum(0,(beta*(np.exp(np.exp(s[t,:]-1)-1)-1))),1)
                else:
                    z = zp
                    w = w*(1*(z==0)*(y[t]==0) + (z==1)*p0d(xm[t,:], pred_std))
                    w = w/np.sum(w)
            
            #measurement update
            K = xP[t,:]/(xP[t,:]+s2e)
            xm[t,:] =  xm[t,:] + z*K*(logyp1[t,i]/alpha-xm[t,:])
            xP[t,:] = xP[t,:] - z*(K**2)*(xP[t,:]+s2e)
            
            if (y[t,i] > 0) & (ESS(w)/N<.1): #
                # resample (systematic)
                resampling_indices = systematic_resampling(w,N)
                prediction_resampling = np.arange(N)
                w = np.ones(N)
            else:
                resampling_indices = np.arange(N)
                prediction_resampling = systematic_resampling(w,N)
            
            
            if t<Tm-1:
                # propagate
                xm[t+1,:] = a*xm[t,resampling_indices] + np.sum(c*u[t,i,:])
                xP[t+1,:] = (a**2)*xP[t,resampling_indices] + s2x
                s[t+1,:] = b*s[t,resampling_indices] + np.sum(d*u[t,i,:]) + np.sqrt(s2s)*np.random.normal(size=N)
 
    progress.update(sg)
    print('')
    return (ypred_mat_mean,ypred_mat_upper,ypred_mat_lower,ypred_mat_bool)


def predict_by_particle_filter_one(th,y,u,N):
    
    Tm = y.shape[0]
    sg = y.shape[1]
    
    progress = pb.ProgressBar(max_value=sg)
    
    logyp1 = np.log(y+1)
    
    (a,b,c,d,s2e,alpha,beta) = th
        
    # parameters
    s2x = (1-a**2)*(1-s2e)
    s2s = 1-b**2
    
    cs2e = 1/np.sqrt(2*np.pi*s2e**2)
    
    p0d = lambda xmt, pred_std : norm.cdf((np.log(1.5)/a-xmt)/pred_std)
    
    for i in range(sg):
        progress.update(i)
        xm = np.zeros((Tm+1,N))
        xP = np.zeros((Tm+1,N))
        s = np.zeros((Tm+1,N))
        sr = np.zeros((Tm+1,N))
        ypred = np.zeros((Tm+1,N))
        ypred_mean = np.zeros((Tm+1,N))
        
        #initalize
        xm[0,:] = np.zeros(N)
        xP[0,:] = np.ones(N)-s2e
        s[0,:] = np.random.normal(size=N)
        
        w = np.ones(N)
        
        for t in range(0,Tm):
            if t == 0: #new data series, restart. Restart with 3 times as many positive s, take account in weight
                xm[t,:] = np.zeros(N)
                xP[t,:] = np.ones(N)-s2e
                Nh = int(np.round(N/4))
                bp = 1
                s[t,:Nh] = truncnorm.rvs(a=-np.inf,b=bp,size=Nh)
                s[t,Nh:] = truncnorm.rvs(a=bp,b=np.inf,size=3*Nh)
                
            #predict!
            pred_std = np.sqrt(s2e + xP[t,:])
            zp = 1*((beta*(np.exp(np.exp(s[t,:]-1)-1)-1))>np.random.uniform(size=N))
            if t > 5:
                yvp = np.maximum(np.round(np.exp(alpha*(xm[t,:]+pred_std*np.random.normal(size=N)))-1),0)
                yp = zp[prediction_resampling]*yvp[prediction_resampling]
                ypred_mean[t,i] = np.sum(w/np.sum(w)*np.minimum(np.maximum(0,(beta*(np.exp(np.exp(s[t,:]-1)-1)-1))),1)*np.maximum(np.exp(alpha*xm[t,:]+((alpha*pred_std)**2)/2)-1,0))
                yp.sort()
                ypred[t,:] = np.copy(yp)
                
            #weighting
            if t == 0:
                z = zp
                w = w*(1*(z==0)*(y[t,i]==0)*cs2e + (z==1)*norm.pdf((logyp1[t,i]/alpha-xm[t,:])/pred_std)/pred_std)
                zp = z
            else:
                pred_std = np.sqrt(s2e + xP[t,:])
                if y[t,i]>0:
                    # weights if z = 1
                    z = np.ones(N)
                    w = w*(norm.pdf((logyp1[t,i]/alpha-xm[t,:])/pred_std)/pred_std)*np.minimum(np.maximum(0,(beta*(np.exp(np.exp(s[t,:]-1)-1)-1))),1)
                else:
                    z = zp
                    w = w*(1*(z==0)*(y[t]==0) + (z==1)*p0d(xm[t,:], pred_std))
                    w = w/np.sum(w)

            #measurement update
            if y[t]>0:
                K = xP[t,:]/(xP[t,:]+s2e)
                xm[t,:] =  xm[t,:] + z*K*(logyp1[t]/alpha-xm[t,:])
                xP[t,:] = xP[t,:] - z*(K**2)*(xP[t,:]+s2e)
            else:
                q = (np.log(1.5)/a-xm[t,:])/np.sqrt(xP[t,:]+s2e)
                xm[t,:] =  xm[t,:] - z*np.sqrt(xP[t,:]+s2e)*norm.pdf(q)/norm.cdf(q)
                xP[t,:] =  np.maximum(xP[t,:] - z*(xP[t,:]*(q*norm.pdf(q)/norm.cdf(q)+(norm.pdf(q)/norm.cdf(q))**2)+s2e),0)
            
            if (y[t,i] > 0) & (ESS(w)/N<.1): #
                # resample (systematic)
                resampling_indices = systematic_resampling(w,N)
                prediction_resampling = np.arange(N)
                w = np.ones(N)
            else:
                resampling_indices = np.arange(N)
                prediction_resampling = systematic_resampling(w,N)
            
            
            if t<Tm-1:
                # propagate
                xm[t+1,:] = a*xm[t,resampling_indices] + np.sum(c*u[t,i,:])
                xP[t+1,:] = (a**2)*xP[t,resampling_indices] + s2x
                s[t+1,:] = b*s[t,resampling_indices] + np.sum(d*u[t,i,:]) + np.sqrt(s2s)*np.random.normal(size=N)
                sr[t+1,:] = s[t+1,prediction_resampling]
 
    progress.update(sg)
    print('')
    return (xm,xP,sr,ypred,ypred_mean)

def evaluate_prediction(y,ypred,ypred_bool=None):
    y_bool = y>0
    if ypred_bool is None:
        ypred_bool = ypred>0
    TP = np.sum(y_bool*ypred_bool)
#    TN = np.sum((1-y_bool)*(1-ypred_bool))
    FP = np.sum((1-y_bool)*ypred_bool)
    FN = np.sum(y_bool*(1-ypred_bool))
    precision = TP/(TP+FN)
    if TP+FP>0:
        recall = TP/(TP+FP)
    else:
        recall = np.nan
    MAEt = np.abs(y[y_bool]-ypred[y_bool])
    MAE = np.mean(MAEt)
    return (precision,recall,MAE,MAEt)