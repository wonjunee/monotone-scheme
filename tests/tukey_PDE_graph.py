#!/usr/bin/env python
# coding: utf-8

# # Tukey depth eikonal equation on N-dimensional graphs
# 
# Given an bounded domain $\Omega \subset \mathbb{R}^d$ ($d\geq 2$) and a data distribution $\rho \in \mathcal{P}(\Omega)$ we are interested to solve, 
# \begin{align}
# |\nabla u(x)| &= \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y), &&  x \in \Omega\\
# u(x) &= 0, && x \in \partial \Omega
# \end{align}

# ## Importing libraries
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import graphlearning as gl
from scipy import sparse
import monotonescheme as ms
import plotly.graph_objects as go

# make folder if not exists
save_fig_path = 'figures'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
save_data_path = 'data'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)


# ## Generating data samples
n = 3000 # number of points
dim = 2 # dimensions

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
mask    = (np.sum((X-0.5)**2,axis=1) > 0.37**2)
bdy_pts = np.arange(n)[mask]

X,V,I,K = process_sparse_data(X,W)

print("data samples generated.")

# ## Initialize PDE solver
def get_sol_circle():
    R = 0.4
    Xc = X - 0.5
    h = np.sqrt(np.maximum(0,R**2 - np.sum(Xc**2,axis=1)))
    sol = np.pi * h**2/3.0 * (3*R - h)
    return sol
def get_sol_square():
    sol = 2 * np.minimum(X[:,0], 1 - X[:,0]) * np.minimum(X[:,1], 1 - X[:,1])
    return sol

sol = get_sol_circle()


# ### Running the algorithm
f = np.ones((n))
u = np.zeros((n))
max_it = 500
tol    = 1e-9
t = tqdm.tqdm(range(max_it), position=0)
for i in t:
    if i==0:
        solver = ms.TukeyGraph(X, V, I, K, f, bdy_pts, 'circle')
    error = solver.iterate(u)
    t.set_description_str(f'Error: {error:0.2e}')
    if error < tol:
        break
print(f"Took {t.format_dict['elapsed']:0.2f} seconds to complete.")
print(f"Error: {np.mean((sol-u)**2):0.2e}")


# $$ V = \frac{\pi h^2}{3} (3r -h) $$
# $$ h = R - |x| $$

# ### Plotting the result
fig, ax = plt.subplots(1,1)
im = ax.scatter(X[:,0], X[:,1], c=u)
ax.set_aspect('equal')
ax.axis('off')
plt.show()

