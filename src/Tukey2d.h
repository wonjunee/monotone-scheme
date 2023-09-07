/**
 * Monotone scheme to solve the Tukey depth eikonal equation in 2D Cartesian grids.
*/

#ifndef TUKEY2D_H
#define TUKEY2D_H

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <future>
#include <cassert>
#include "Monotone2d.h"
#include "Helper.h"

#ifndef  _DEBUG
#define _NDEBUG
#endif

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

namespace py = pybind11;
using namespace std;

class Tukey2DSolver : public Monotone2DSolver {
public:
    
    std::vector<std::vector<double> > rhs_integral_vec_;

    /**
     * initializer
     * @param f_np : numpy array for the right hand side function
     * @param st_size : the size of the stencil. if st_size=1 -> 8 stencils, st_size=2 -> 16 stencils
     */
    Tukey2DSolver(py::array_t<double, py::array::c_style | py::array::forcecast> f_np, int st_size)
    : Monotone2DSolver(f_np, st_size) {
        // resizing stencils and stencils norm
        N_stencils_ = st_size_ * 8;
        dir_vec_.resize(N_stencils_);
        dir_norm_vec_.resize(N_stencils_);

        vector<vector<double> > dir_vec_tmp = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {1,-1}};

        for(size_t k=0,N=dir_vec_tmp.size();k<N;++k){
            double d1x = dir_vec_tmp[k][0];
            double d1y = dir_vec_tmp[k][1];
            double d2x = dir_vec_tmp[(k+1)% N][0];
            double d2y = dir_vec_tmp[(k+1)% N][1];

            for(int count=0;count<st_size_;++count){
                double lambda = 1.0 * count / (1.0*st_size_);
                int ind = k*st_size_ + count;
                double new_dir_x = (1.0-lambda) * d1x + lambda * d2x;
                double new_dir_y = (1.0-lambda) * d1y + lambda * d2y;
                dir_vec_[ind] = {new_dir_x,new_dir_y};
                dir_norm_vec_[ind] = sqrt(new_dir_x*new_dir_x + new_dir_y*new_dir_y)/n_;
            }
        }

        // Initialize the right hand side integral term
        initialize_rhs_integral_vec();

        py::print("Constructor finished. n: ", n_, "stencil size: ", st_size, "number of stencils: ", N_stencils_, "number of threads: ", THREADS_);
    }

    void initialize_rhs_integral_vec(){
        rhs_integral_vec_.resize(n_*n_);
        for(int ind=0;ind<n_*n_;++ind){
            rhs_integral_vec_[ind].resize(N_stencils_);
            int i = ind/n_;
            int j = ind%n_;
            for(int d=0; d<N_stencils_; ++d){
                /**
                * compute int_{(y-x)\cdot p = 0} \rho(y) dS(y) = 0
                * dir = orthogonal to p
                */ 
                vector<double> p = dir_vec_[d];
                vector<double> dir = {-p[1], p[0]};

                double fval = 0;                
                double new_i = i, new_j = j;

                /** forward direction */
                while(check_inside_domain(round(new_i), round(new_j))){
                    fval += f_(round(new_i)*n_+round(new_j));
                    new_j += dir[0];
                    new_i += dir[1];
                }

                new_j = j - dir[0]; new_i = i - dir[1];

                /** reverse direction */
                while(check_inside_domain(round(new_i), round(new_j))){
                    fval += f_(round(new_i)*n_+round(new_j));
                    new_j -= dir[0];
                    new_i -= dir[1];
                }
                fval *= dir_norm_vec_[d];
                rhs_integral_vec_[ind][d] = fval;
            }
        }
    }

    /**
     * calculating the value of the Tukey depth eikonal equation
     * \max_{p \in P^h} |\nabla u(x)| - \int_{p^\perp} \rho(y) dS(y)
     * @param utmp : u^{(k)} values of u at the previous iteration
     * @param f    : the data density \rho
     * @param c    : u(x) computed for u^{(k+1)}. u(x) -> S_h(u^{(k)}, u(x), x)
     * @param ind  : the index for the location x. i = ind/n, j = ind%n
     * @return S_h(u^{(k)}, c, x)
     */
    double calc_u(const DoubleArray2D& utmp, const DoubleArray2D& f, const double c, const int ind){
        double max_val = -1e4;
        int i = ind / n_;
        int j = ind % n_;

        std::vector<double> uval_vec(N_stencils_);
        /** Compute the stencils on 3x3 stencils */
        initialize_uval_dir_vec(uval_vec, utmp, i, j);

        /**  choose each gradient from stencils */
        for(int d=0; d<N_stencils_; ++d){
            if(p_in_subdifferential(uval_vec,d,c)){
            // if(true){
                /** h_dir = norm of p = |p| */ 
                double h_dir = dir_norm_vec_[d];

                /** compute nabla u = u(x) - u(x-p) / |p| */ 
                double uback = uval_vec[d];
                double Du_val = (c - uback) / h_dir;
                double fval = rhs_integral_vec_[ind][d];
                double val = Du_val - fval;
                if(val > max_val){ max_val = val; }
            }
        }
        return max_val;
    }
};
#endif