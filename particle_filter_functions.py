import numpy as np

def systematic_resampling(W,N):
    if np.sum(W)==0:
        W = np.ones(N)
    W = W/np.sum(W)
    u = 1/N*np.random.uniform()
    idx = np.zeros(N,dtype=np.int32)
    q = 0
    n = -1
    for i in range(0,N):
        while q < u:
            n = n+1
            q = q + W[n];
        idx[i] = n
        u = u + 1/N
    return idx

def ESS(W):
    return np.sum(W)**2/np.sum(W**2)

