# Monotone schemes for curvature-driven PDEs

## by Jeff Calder (UMN)  and Wonjun Lee (UMN)

- Paper: [arXiv](.)
- Jeff Calder, School of Mathematics, University of Minnesota: [website](https://www-users.cse.umn.edu/~jwcalder/)
- Wonjun Lee, Institute for Mathematics and Its Applications, Uniersity of Minnesota: [website](https://wonjunee.github.io)

---
# Outline
This repository contains c++ and python codes for running the monotone algorithm to solve curvature-driven PDEs. Here are list of PDEs that can be solved using this algorithm. Let $\Omega \subset \mathbb{R}^d$ be an open bounded domain and $\partial \Omega$ be a boundary of $\Omega$.

### Eikonal equation
$$
\begin{align*}
     |\nabla u(x)| &= f(x), && x \in \Omega \\
     x &= 0, && x \in \partial \Omega 
\end{align*}
$$
### Mean curvature PDE
$$
\begin{align*} 
    |\nabla u(x)|\kappa(x) &= f(x), && x \in \Omega \\
    x &= 0, && x \in \partial \Omega 
\end{align*}
$$
where $\kappa(x) = - \text{div}\left( \frac{\nabla u}{|\nabla u|} \right)$ is the mean curvature of the level set surface of $u$ passing through $x$.

### Affine flows PDE
$$
\begin{align*} 
    |\nabla u(x)|\kappa(x)_+^{\alpha} &= f(x), && x \in \Omega \\
    x &= 0, && x \in \partial \Omega 
\end{align*}
$$
where $\alpha \in (0,1]$ is a constant depending on the dimension $d$ and $(t)_+ := \max(0,t)$.

### Tukey Depth
$$ |\nabla u(x)| = \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y), \quad x \in \Omega.$$



---
# Tutorial

## Prerequisites

- `pip`
- `python >= 3.6`

Follow this link to see the instruction for the installation of `pip`: [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/).


## Installing the package

First install the package by running the following command:
```
    pip install MonotoneScheme
```

(TO BE CONTINUED)