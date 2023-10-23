# Monotone schemes for curvature-driven PDEs

## by Jeff Calder (UMN)  and Wonjun Lee (UMN)

- Paper: [arXiv](https://arxiv.org/abs/2310.08450)
- Jeff Calder, School of Mathematics, University of Minnesota: [website](https://www-users.cse.umn.edu/~jwcalder/)
- Wonjun Lee, Institute for Mathematics and Its Applications, Uniersity of Minnesota: [website](https://wonjunee.github.io)

---
# Introduction
This repository contains C++ and python codes for running the monotone algorithm to solve curvature-driven PDEs. Here are list of PDEs that can be solved using this algorithm. Let $\Omega = [0,1]^d$ be a domain in $\mathbb{R}^d$ and $\partial \Omega$ be a boundary of $\Omega$.

### Eikonal equation
$$ |\nabla u(x)| = f(x),\quad  x \in \Omega $$

$$ u(x) = 0,\quad  x \in \partial \Omega $$

### Mean curvature PDE
$$|\nabla u(x)|\kappa(x) = f(x),\quad  x \in \Omega $$

$$ u(x) = 0,\quad  x \in \partial \Omega $$

where $\kappa(x) = - \text{div}\left( \frac{\nabla u}{|\nabla u|} \right)$ is the mean curvature of the level set surface of $u$ passing through $x$.

### Affine flows PDE

$$|\nabla u(x)|\kappa(x)_+^{\alpha} = f(x),\quad  x \in \Omega $$

$$u(x) = 0,\quad  x \in \partial \Omega $$

where $\alpha \in (0,1]$ is a constant depending on the dimension $d$ and $(t)_+ := \max(0,t)$.

### Tukey Depth

$$ |\nabla u(x)| = \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y),\quad  x \in \Omega $$

$$ u(x) = 0,\quad  x \in \partial \Omega $$



---
# Tutorial

## Prerequisites

- `pip`
- `python >= 3.6`

Follow this link to see the instruction for the installation of `pip`: [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/).


## Installing the package

Install the package by running the following command:
```
    pip install monotonescheme
```

## Running the codes

You can find the example python script files and notebook files in ``tests`` folder. The notebook files in the folder solves the following problems:

1. Affine flows in 2D Cartesian grid. 

- tests/affine_PDE_2D.ipynb
- tests/affine_PDE_2D.py

![Alt text](https://github.com/wonjunee/monotone-scheme/blob/v1/figures/affine2d.png)

2. Tukey depth eikonal equation in 2D Cartesian grid.

- tests/tukey_PDE_2D.ipynb
- tests/tukey_PDE_2D.py

![Alt text](https://github.com/wonjunee/monotone-scheme/blob/v1/figures/tukey2d.png)

3. Motion by curvature PDE in 3D Cartesian grid.

- tests/curvature_PDE_3D.ipynb
- tests/curvature_PDE_3D.py

![Alt text](https://github.com/wonjunee/monotone-scheme/blob/v1/figures/square-3d.png)

4. Eikonal equation and Tukey depth eikonal equation in unstructure grids

- tests/Eikonal_PDE_graph.ipynb
- tests/tukey_PDE_graph.ipynb
- tests/Eikonal_PDE_graph.py
- tests/tukey_PDE_graph.py

![Alt text](https://github.com/wonjunee/monotone-scheme/blob/v1/figures/point_cloud.png)