# monotone schemes for solving curvature motion PDEs

## by Jeff Calder (UMN) and Wonjun Lee (UMN)

This repository contains c++ and python codes for running the monotone algorithm to solve curvature motion PDEs. Here are list of PDEs that can be solved using this algorithm.

### Eikonal equation
$$ |\nabla u(x)| = f(x), \quad x \in \Omega $$
$$ x = 0, \quad x \in \partial \Omega $$

### Mean curvature PDE
$$ |\nabla u(x)|\kappa(x) = f(x), \quad x \in \Omega $$
$$ x = 0, \quad x \in \partial \Omega $$
where $\kappa(x) = - \text{div}\left( \frac{\nabla u}{|\nabla u|} \right)$.

### Tukey Depth
$$ |\nabla u(x)| = \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y), \quad x \in \Omega.$$