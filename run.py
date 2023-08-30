import os
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from monotone import Tukey2DSolver, interpolate

os.system("bash compile.sh")

# make folder if not exists
save_fig_path = 'figures'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
save_data_path = 'data'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

def saving_fig(u,error, start_time, error_array, time_array, i, fig_title='fig'): 
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    # ax.contourf(X,Y,u)
    ax.imshow(u, origin='lower')
    ax.axis('off')
    ax.set_aspect('equal')
    title_str = f"i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec"
    plt.title(title_str)
    ax = fig.add_subplot(122)
    ax.plot(time_array, error_array,'o-')
    plt.savefig(f"{save_fig_path}/{fig_title}-{i//10}.png")
    
# create an empty f
n = 256
f = np.zeros((n,n))
u = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        u[i,j] = i*n + j

# initializing f
xx = np.linspace(0.5/n,1-0.5/n,n)
X,Y = np.meshgrid(xx,xx)

f[(np.abs(X-0.5)<0.5) & (np.abs(Y-0.5)<0.5)] = 1
# returning contiguous arrays in memory
u = np.ascontiguousarray(u, dtype=np.float64)  # solution
f = np.ascontiguousarray(f, dtype=np.float64)  # the right hand side function

# optimization parameters
max_iter = 1000
tolerance= 1e-3

# initialize the HJ solver
for stencil_size in [5, 10, 20, 50]:
    solver  = Tukey2DSolver(f, stencil_size)

    # Running the iterations
    start_time = time.time()
    error_array = []
    time_array = []

    error = 1.0
    for i in tqdm.tqdm(range(1,max_iter+1)):
        error = solver.perform_one_iteration(u)
        error_array.append(error)
        time_array.append(time.time() - start_time)
        # print(f"i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec")
        # if error is less than tolerance stop the iteration
        if error < tolerance or i == max_iter:
            print(f"Tolerance met! i: {i} error: {error:0.2e} time: {time.time() - start_time:0.2e} sec")
            break
        # save figures
        if i % 10 == 0:
            saving_fig(u,error, start_time, error_array, time_array, i, fig_title=f'fig-stencils-{stencil_size}')
        # save dat
        if i % 100 == 0:
            np.save(f"{save_data_path}/data-stencils-{stencil_size}-i-{i//100}.npy",u)


    # plotting the result
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    CS = ax.contour(X,Y,u,10)
    ax.clabel(CS, inline=1, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_fig_path}/tukey-square-stencils-{stencil_size}.eps")