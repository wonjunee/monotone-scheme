#!/usr/bin/env python
# coding: utf-8

# # Eikonal equation on N-dimensional graphs
# 
# Given an bounded domain $\Omega \subset \mathbb{R}^d$ ($d\geq 2$) we are interested to solve, 
# \begin{align}
# |\nabla u(x)| &= f(x) = 0, &&  x \in \Omega\\
# u(x) &= 0, && x \in \partial \Omega
# \end{align}

# ## Importing libraries
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import graphlearning as gl
from scipy import sparse
from monotonescheme import EikonalGraph

# make folder if not exists
save_fig_path = 'figures'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
save_data_path = 'data'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)


# ## Generating data samples


n = 5000 # number of points
dim = 2 # dimensions

def generate_random_point_cloud_unit_square(n, dim=2, k=20):
    X = np.random.rand(n, dim)
    W = gl.weightmatrix.knn(X,k,kernel='distance')
    return X, W
def generate_random_point_cloud_circle(n, dim=2, k=20):
    X = [[0.5]*dim]
    count = 1
    R = 0.4
    while count < n:
        z = np.random.rand(dim)
        if  np.sum((z-0.5)**2)  <= R**2:
            X.append(z)
            count += 1
    X = np.array(X)

    W = gl.weightmatrix.knn(X,k,kernel='distance')
    return X, W

def process_sparse_data(X, W):
    #Coordinates of sparse matrix for passing to C code
    I,J,V = sparse.find(W)
    K = np.array((J[1:] - J[:-1]).nonzero()) + 1
    K = np.append(0,np.append(K,len(J)))
    X = np.ascontiguousarray(X, np.float64)
    V = np.ascontiguousarray(V, np.float64)
    I = np.ascontiguousarray(I, np.int32)
    K = np.ascontiguousarray(K, np.int32)
    print(X.shape, V.shape, K.shape, I.shape)
    return X,V,I,K

X, W = generate_random_point_cloud_circle(n,dim,30) # first param: number of points # second param: number of neighbord points
mask    = (np.sum((X-0.5)**2,axis=1) > 0.36**2)
bdy_pts = np.arange(n)[mask]

X,V,I,K = process_sparse_data(X,W)

print("data samples generated.")


# ## Initialize PDE solver


f = np.ones((n))
u = np.zeros((n))
max_it = 100
tol    = 1e-5
solver = EikonalGraph(X, V, I, K, f, bdy_pts)


# ### Running the algorithm


t = tqdm.tqdm(range(max_it), position=0)
for i in t:
    error = solver.iterate(u)
    t.set_description_str(f'Error: {error:0.2e}')
    if error < tol:
        break
print(f"Took {t.format_dict['elapsed']:0.2f} seconds to complete.")


# ### Plotting the result


fig, ax = plt.subplots(1,1,figsize=(6,6))
im = ax.scatter(X[:,0], X[:,1], c=u)
ax.set_aspect('equal')
plt.show()

