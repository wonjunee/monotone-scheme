import os
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from Monotone import Tukey2DSolver, Eikonal2DSolver

os.chdir('../')

os.system("bash compile.sh")

# make folder if not exists
save_fig_path = 'figures'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
save_data_path = 'data'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

def saving_fig(u, f, error, start_time, error_array, time_array, i, fig_title='fig'): 
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(131)
    ax.imshow(f, origin='lower')
    ax.set_title("$f$")
    ax.axis('off')
    ax.set_aspect('equal')
    ax = fig.add_subplot(132)
    # ax.imshow(u, origin='lower')
    CS = ax.contour(X,Y,u,10)
    ax.clabel(CS, inline=1, fontsize=8)
    ax.axis('off')
    ax.set_aspect('equal')
    title_str = f"i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec"
    plt.title(title_str)
    ax = fig.add_subplot(133)
    ax.plot(time_array, error_array,'o-')
    plt.savefig(f"{save_fig_path}/{fig_title}-{i}.png")
    
# create an empty f
n = 32
# initializing f
xx = np.linspace(0.5/n,1-0.5/n,n)
X,Y = np.meshgrid(xx,xx)

f = np.zeros((n,n))
# f[(np.abs(X-0.5)<0.4) & (np.abs(Y-0.5)<0.4)] = 1
# f[(np.abs(X-0.5)<0.2) & (Y < 0.6)] = 0
# f[((X-0.5)**2 + (Y-0.5)**2 <0.5**2)] = 1; f[((X-0.6)**2 + (Y-0.5)**2 <0.03**2)] = 0;
f[((X-0.3)**2 + (Y-0.3)**2 <0.22**2) | ((X-0.7)**2 + (Y-0.7)**2 <0.22**2)] = 1
# f[((X-0.25)**2 + (Y-0.25)**2 <0.2**2) | ((X-0.25)**2 + (Y-0.75)**2 <0.2**2) | ((X-0.75)**2 + (Y-0.25)**2 <0.2**2) | ((X-0.75)**2 + (Y-0.75)**2 <0.2**2)] = 1
# returning contiguous arrays in memory
f = np.ascontiguousarray(f, dtype=np.float64)  # the right hand side function


fig,ax =plt.subplots(1,1,figsize=(4,4))
ax.imshow(f, origin='lower')
ax.axis('off')
ax.set_aspect('equal')
plt.savefig("figures/small-perturb.eps")

# optimization parameters
max_iter = 2000
tolerance= 1e-8

# initialize the HJ solver
# for stencil_size in [2, 5, 25, 100]:
for stencil_size in [2]:
    u = np.zeros((n,n))
    u = np.ascontiguousarray(u, dtype=np.float64)  # solution
    print(f"stencil size: {stencil_size}")
    # solver  = Tukey2DSolver(f, stencil_size)
    solver  = Eikonal2DSolver(f, stencil_size)

    # Running the iterations
    start_time = time.time()
    error_array = []
    time_array = []

    error = 1.0
    for i in tqdm.tqdm(range(1,max_iter+1)):
        error = solver.iterate(u)
        error_array.append(error)
        time_array.append(time.time() - start_time)
        # print(f"i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec")
        # if error is less than tolerance stop the iteration
        if error < tolerance:
            print(f"Tolerance met! i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec")
            break
        # save figures
        if i % 50 == 0:
            saving_fig(u, f, error, start_time, error_array, time_array, i//50, fig_title=f'eikonal-stencils-{stencil_size}')
        # save dat
        skip = 50
        if i % skip == 0:
            np.save(f"{save_data_path}/eikonal-stencils-{stencil_size}-i-{i//skip}.npy",u)


    # plotting the result
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    CS = ax.contour(X,Y,u,10)
    ax.clabel(CS, inline=1, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_fig_path}/eikonal-stencils-{stencil_size}.eps")