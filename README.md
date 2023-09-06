# monotone schemes for solving curvature motion PDEs

## by Jeff Calder (UMN) and Wonjun Lee (UMN)

---
## Outline
This repository contains c++ and python codes for running the monotone algorithm to solve curvature motion PDEs. Here are list of PDEs that can be solved using this algorithm.

### Eikonal equation
$$ |\nabla u(x)| = f(x), \quad x \in \Omega $$
$$ x = 0, \quad x \in \partial \Omega $$

### Mean curvature PDE
$$ |\nabla u(x)|\kappa(x) = f(x), \quad x \in \Omega $$
$$ x = 0, \quad x \in \partial \Omega $$
where $\kappa(x) = - \text{div}\left( \frac{\nabla u}{|\nabla u|} \right)$ is the mean curvature of the level set surface of $u$ passing through $x$.

### Tukey Depth
$$ |\nabla u(x)| = \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y), \quad x \in \Omega.$$



---
## Tutorial

### Prerequisite

- `pip`
- `python >= 3.6`

Follow this link to see the instruction for the installation of `pip`: [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/).


### Installing the package

First install the package by running the following command:
```
    pip install MonotoneScheme
```

(TO BE CONTINUED)