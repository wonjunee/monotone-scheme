#!/usr/bin/env python
# coding: utf-8

# # Motion by Curvature PDE in 3D Cartesian Grids
# 
# In this notebook, we are interested in solving the motion by curvature PDE. Consider a unit square domain $\Omega$ in 3D which is discretized as regular Cartesian grids. Let $u:\Omega\rightarrow \mathbb{R}$ be a function that satisfies the PDE
# $$ |\nabla u(x)| \kappa(x) = f(x) $$
# where $\kappa(x)$ is a curvature function defined as 
# $$ \kappa(x) = - \text{div}\left( \frac{\nabla u(x)}{|\nabla u(x)|} \right)$$
# We will solve this PDE using the monotone scheme proposed in the paper. The below is the outline of this notebook.
# 
# 1. Importing library
# 2. Constructing 3d stencils
# 3. Constructing the right hand side function $f:\Omega\rightarrow \mathbb{R}$
# 4. Solve the PDE using the proposed monotone scheme
# 5. Plot the results

# ## Importing python libraries
import os
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from monotonescheme import Curv3DSolver
import plotly.graph_objects as go


# make folder if not exists
save_fig_path = 'figures'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
save_data_path = 'data'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)


# ## Constructing 3D stencils
# 3D version

# Given a stencil_size, this function will return a numpy array of size [ __, 3 ]
# If stencil_size = 1 then it provides stencils ranging from -1 to 1
# If stencil_size = 2 then it provides stencils ranging from -2 to 2
def create_stencils_3d(stencil_size):
    stencils_tmp = []
    # add cases for i0 == 0
    for i0 in range(-stencil_size, stencil_size+1):
        for i1 in range(-stencil_size, stencil_size+1):
            for i2 in range(-stencil_size, stencil_size+1):
                if i0!=0 or i1!=0 or i2!=0:
                    stencils_tmp.append((i0,i1,i2, np.arctan2(i0,i1), np.arctan2(i0,i2), np.arctan2(i1,i2), i0*i0 + i1*i1 + i2*i2))

    stencils_tmp.sort(key=lambda x: x[6])
    stencils_tmp.sort(key=lambda x: x[5])
    stencils_tmp.sort(key=lambda x: x[4])
    stencils_tmp.sort(key=lambda x: x[3])

    stencils = []
    for i, it in enumerate(stencils_tmp):
        if i == 0:
            stencils.append(it[:3])
        else:
            if it[3] != stencils_tmp[i-1][3] or it[4] != stencils_tmp[i-1][4] or it[5] != stencils_tmp[i-1][5]:
                stencils.append(it[:3])

    del stencils_tmp
    return np.array(stencils)

# Let's get a numpy array that contains the list of stencils given the stencil size in 3D
stencil_size = 2
stencils = create_stencils_3d(stencil_size)

# create an empty f
n = 30
f = np.zeros((n,n,n))
u = np.zeros((n,n,n))
# initializing f
xx = np.linspace(0.5/n,1-0.5/n,n)
X,Y,Z = np.meshgrid(xx,xx,xx)
X2,Y2 = np.meshgrid(xx,xx)

def f_ellipse():
    f = np.zeros((n,n,n))
    theta = -np.pi/3.0
    Xr = 0.5 + np.cos(theta)*(X-0.5) - np.sin(theta)*(Y-0.5)
    Yr = 0.5 + np.sin(theta)*(X-0.5) + np.cos(theta)*(Y-0.5)
    f[4*pow(Xr - 0.5,2) + pow(Yr-0.5,2) + 2*pow(Z-0.5,2) < 0.5 ** 2] = 1
    return f

f = f_ellipse()

# plotting f
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=f.flatten(),
    opacity=0.6,
    isomin=np.min(f),
    isomax=np.max(f),
    surface_count=10,
    caps=dict(x_show=False, y_show=False, z_show=False)
    ))
fig.show()

# ## Solve the PDE
# optimization parameters
max_iter = 50
tolerance= 1e-3

# initialize the HJ solver
solver = Curv3DSolver(f, stencils, stencil_size)

# Running the iterations
start_time = time.time()
error_array = []
offset_arr = [2,5, 10, 15, 25]

pbar = tqdm.tqdm(range(max_iter+1))
for i in pbar:
    error = solver.iterate(u)
    error_array.append(error)
    
    # save figures
    if i % 10  == 0:
        pbar.write(f"i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec")
    # stopping condition
    if error < tolerance or i == max_iter:
        np.save(f"{save_data_path}/data.npy",u)
        break

# Load data
u = np.load(f"data/data.npy")

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=u.flatten(),
    opacity=0.6,
    isomin=np.max(u),
    isomax=np.min(u),
    surface_count=10,
    caps=dict(x_show=False, y_show=False, z_show=False)
    ))
fig.show()





