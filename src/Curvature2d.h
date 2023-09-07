/**
 * Monotone scheme to solve the mean curvature PDEs in 2D Cartesian grids.
*/

#ifndef CURVATURE2D_H
#define CURVATURE2D_H

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


class Curv2DSolver : public Monotone2DSolver {
public:
    /**
     * Initializing member variables
    */

    std::vector< std::vector<int> > stencils_; // vector of stencils ex: {{0,0,1}, {0,1,1}, ... }

    /**
     * initializer
     * @param f_np : numpy array for the right hand side function
     * @param stencils_np : numpy arry for the stencils
     * @param st_size : the size of the stencil. if st_size=1 -> 8 stencils, st_size=2 -> 16 stencils
     */
    Curv2DSolver(py::array_t<double, py::array::c_style | py::array::forcecast> f_np, py::array_t<int, py::array::c_style | py::array::forcecast> stencils_np, int st_size)
    : Monotone2DSolver(f_np, st_size) {
        // initizlie stencils_ from stencil numpy array
        py::buffer_info stencils_buf = stencils_np.request();
        int *stencils                = static_cast<int *>(stencils_buf.ptr);
        N_stencils_ = stencils_buf.shape[0];

        // resizing stencils and stencils norm
        stencils_.resize(N_stencils_);
        dir_norm_vec_.resize(N_stencils_);

        // converting from int* -> vector<vector<int>>
        const int dim = 2;
        for(int it=0;it<N_stencils_;++it){
            stencils_[it].resize(dim); // 2d vector e.g. {0,1}
            double norm_val = 0;
            for(int it1=0;it1<dim;++it1){
                stencils_[it][it1] = stencils[it*dim + it1];
                norm_val += stencils_[it][it1] * stencils_[it][it1];
            }
            dir_norm_vec_[it] = sqrt(norm_val);
        }

        py::print("Constructor finished. n: ", n_, "stencil size: ", st_size, "number of stencils: ", stencils_.size());
    }
                    
    /**
     * Given a function u and a vector q, this function will return the second derivative 
     * with respect to the direction q at the location x=(k,i,j).
     * It will return $-\Delta u = (-u(x-q) + 2u(x) - u(x+q))/h^2$.
    */
    double compute_second_derivative_given_p(const DoubleArray2D& utmp, vector<int>& q, const double c, const int i, const int j) const{
        int im = i-q[1]; int ip = i+q[1];
        int jm = j-q[0]; int jp = j+q[0];
        double umm = 0;
        double upp = 0;
        
        if(check_inside_domain(ip,jp)){
            upp = utmp(ip,jp);
        }
        if(check_inside_domain(im,jm)){
            umm = utmp(im,jm);
        }
        double h2 = dot(q,q)/(n_*n_); // norm of p : |p| * (dx^2)
        return (- umm + 2.0 * c - upp) / h2;
    }

    /**
     * In this function, given a displacement vector v=stencils_[d], it will check if it is 
     * eligible to be in the subdifferential set.
     * For every w=stencils_[it] such that v . w < 0, check if u(x-w) < u(x). If not return false.
     * @param utmp : DoubleArray2D for the solution
     * @param d    : size_t. The index for stencils_
     * @param c    : the value at u(k,i,j)
     * @param i,j: the indices i: y-aixs j: x-axis
     * @return true if stencils_[d] is in subdifferential set.
    */
    bool p_in_subdifferential(const DoubleArray2D& utmp, const size_t& d, const double c, const int i, const int j) const{
        int ip = i - stencils_[d][1];
        int jp = j - stencils_[d][0];
        if(check_inside_domain(ip,jp)){
            if(utmp(ip,jp) > c){
                return false;
            }
        }

        for(int it=0, N_it=stencils_.size(); it<N_it; ++it){ // dir = {x, y}
            int ip = i - stencils_[it][1];
            int jp = j - stencils_[it][0];
            if(check_inside_domain(ip,jp)){
                if(dot(stencils_[it], stencils_[d]) > 0){
                    if(utmp(ip,jp) > c){
                        return false;
                    }
                }
            }
        }
        return true;
    }
    
    /**
     * calculating affine flows
     */
    double calc_u(const DoubleArray2D& utmp, const DoubleArray2D& f, const double c, const int ind){
        double max_val = -1e4;
        int i = ind / n_;
        int j = ind % n_;

        for(size_t d=0, N_d=stencils_.size(); d<N_d; ++d){ // dir = {x, y}
            // choose a eligible vector from stencils
            if(p_in_subdifferential(utmp,d,c,i,j)){
                vector<int> p = stencils_[d];
                vector<int> q = {-p[1], p[0]}; // q is perpendicular to p

                double val = compute_second_derivative_given_p(utmp, q, c, i, j);
                if(val > max_val){ max_val = val; }
            }
        }
        return max_val - f(ind);
    }
};


#endif