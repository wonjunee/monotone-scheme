/**
 * Monotone scheme to solve affine flows in 2D Cartesian grids.
*/

#ifndef AFFINE2D_H
#define AFFINE2D_H

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
#include "Helper.h"

#ifndef  _DEBUG
#define _NDEBUG
#endif

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

namespace py = pybind11;
using namespace std;


class Affine2DSolver{
public:
    /**
     * Initializing member variables
    */
    DoubleArray2D u_; // used for computing the solution of the PDE
    DoubleArray2D f_; // used for computing the solution of the PDE
    std::vector<double> stencils_norm_; // the vector of norms of the stencils
    std::vector< std::vector<int> > stencils_; // vector of stencils ex: {{0,0,1}, {0,1,1}, ... }
    std::vector<double> errors_; // the vector of size n_*n_ that will contain the error at each grid point
    int n_; // size of the grid n_ x n_ x n_
    int st_size_; // stencil size 1,2, or 3
    int st_N_; // the number of elements in stencils
    int THREADS_; // # of threads in CPU
    double max_it_bisection_; // max iteration of bisection method
    
    /**
     * initializer
     * @param n_ : grid size of x-axis
     */
    Affine2DSolver(int n)
    : n_(n), st_size_(0), THREADS_(std::thread::hardware_concurrency()){
    }

    /**
     * initializer
     * @param n_ : grid size of x-axis
     */
    Affine2DSolver(py::array_t<double, py::array::c_style | py::array::forcecast> f_np, py::array_t<int, py::array::c_style | py::array::forcecast> stencils_np, int st_size)
    : st_size_(st_size), THREADS_(std::thread::hardware_concurrency()){
        py::buffer_info f_buf = f_np.request();
        double *f_dbl         = static_cast<double *>(f_buf.ptr);

        n_ = f_buf.shape[0];

        // initialize u_ and f_
        f_.initialize(f_dbl,n_);
        u_.initialize(n_);

        // initizlie stencils_ from stencil numpy array
        py::buffer_info stencils_buf = stencils_np.request();
        int *stencils                = static_cast<int *>(stencils_buf.ptr);
        st_N_ = stencils_buf.shape[0];

        // errors initialize
        errors_.resize(n_*n_);

        // resizing stencils and stencils norm
        stencils_.resize(st_N_);
        stencils_norm_.resize(st_N_);

        // converting from int* -> vector<vector<int>>
        const int dim = 2;
        for(int it=0;it<st_N_;++it){
            stencils_[it].resize(dim); // 2d vector e.g. {0,1}
            double norm_val = 0;
            for(int it1=0;it1<dim;++it1){
                stencils_[it][it1] = stencils[it*dim + it1];
                norm_val += stencils_[it][it1] * stencils_[it][it1];
            }
            stencils_norm_[it] = sqrt(norm_val)/n_;
        }

        py::print("Constructor finished. n: ", n_, "stencil size: ", st_size, "number of stencils: ", stencils_.size());
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
    bool p_in_subdifferential(const DoubleArray2D& utmp, const int& d, const double c, const int i, const int j) const{
        int ip = i - stencils_[d][1];
        int jp = j - stencils_[d][0];
        if(check_inside_domain(ip,jp)){
            if(utmp(ip,jp) > c){
                return false;
            }
        }

        for(int it=d-st_N_/4+1+st_N_, N_it=d+st_N_/4-1+st_N_; it<N_it; ++it){ // dir = {x, y}
            int it0 = it % st_N_;
            int ip = i - stencils_[it0][1];
            int jp = j - stencils_[it0][0];
            if(check_inside_domain(ip,jp)){
                if(utmp(ip,jp) > c){
                    return false;
                }
            }
        }
        return true;
    }

                    
    /**
     * Given a function u and a vector q, this function will return the second derivative 
     * with respect to the direction q at the location x=(k,i,j).
     * It will return $-\Delta u = (-u(x-q) + 2u(x) - u(x+q))/h^2$.
    */
    double compute_second_derivative_given_p(const DoubleArray2D& utmp, vector<int>& q, const double c, const int i, const int j, const int d) const{
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
        // double h2 = dot(q,q)/(n_*n_); // norm of p : |p| * (dx^2)
        double h2 = stencils_norm_[d] * stencils_norm_[d];
        return (- umm + 2.0 * c - upp) / h2;
    }

    /**
     * Given a function u and a vector q, this function will return the first derivative 
     * with respect to the direction q at the location x=(k,i,j).
     * It will return $| \nabla u(x) | = (u(x) - u(x-p))/|p|$.
    */
    double compute_first_derivative_given_p(const DoubleArray2D& utmp, vector<int>& p, const double c, const int i, const int j, const int d) const{
        int im = i-p[1];
        int jm = j-p[0];
        double um = 0;
        if(check_inside_domain(im,jm)){
            um = utmp(im,jm);
        }
        // double h = sqrt(dot(p,p))/n_; // norm of p : |p| * dx
        double h = stencils_norm_[d];
        return (c - um) / h;
    }

    /**
     * calculating affine flows
     * @param utmp : u^{(k)} values of u at the previous iteration
     * @param f    : the right hand side function
     * @param c    : u(x) computed for u^{(k+1)}. u(x) -> S_h(u^{(k)}, u(x), x)
     * @param ind  : the index for the location x. i = ind/n, j = ind%n
     * @return S_h(u^{(k)}, c, x)
     */
    double calc_u(const DoubleArray2D& utmp, const DoubleArray2D& f, const double c, const int ind){
        double max_val = -1e4;
        int i = ind / n_;
        int j = ind % n_;

        for(int d=0, N_d=stencils_.size(); d<N_d; ++d){ // dir = {x, y}
            // choose a eligible vector from stencils
            if(p_in_subdifferential(utmp,d,c,i,j)){
                vector<int> p = stencils_[d];
                vector<int> q = {-p[1], p[0]}; // q is perpendicular to p

                double first_deriv  = compute_first_derivative_given_p(utmp,  p, c, i, j, d);
                double second_deriv = compute_second_derivative_given_p(utmp, q, c, i, j, d);
                double val = first_deriv * first_deriv * fmax(0,second_deriv);
                if(val > max_val){ max_val = val; }
            }
        }
        return max_val - pow(f(ind),3);
    }

    inline bool check_inside_domain(const int i, const int j) const{
        return i>=0 && i<n_ && j>=0 && j<n_;
    }

    /**
     * Given functions and a location at x, it will perform a bisection method at x using
     * the monotone discretization function.
     * @param utmp : DoubleArray2D solution array
     * @param f    : DoubleArray2D the right hand side function
     * @param ind  : the index of the grid. The location x
     * @param val  : It will compute the value of the HJ equation. Being close to 0 is better.
     * @return c : this will be the new updated value for utmp(x).
    */
    double calc_u_bisection_affine(const DoubleArray2D& utmp, const DoubleArray2D& f, const int ind, double& val){
        // py::print("inside the bisection function\n");
        double a = 0.0;
        double b = 1.0;
        double c = (a+b)*0.5;
        val = 100;
        for(size_t i_bi=0; i_bi<max_it_bisection_; ++i_bi){
            double uval = calc_u(utmp, f, c, ind);
            if(uval > 0){ b = c; }
            else        { a = c; }
            c = (a+b)*0.5;
            val = fmin(val, fabs(uval));
        }
        return c;
    }

    /**
     * Given starting and ending indices, it will iterate from the starting index
     * to the ending index and perform bisection method. This function is used for
     * the multithreading purpose. The class member variable u_ will be updated
     * in the end.
     * @param it_start : starting index
     * @param it_end : ending index
     * @return error and u_ will be updated.
    */
    void compute_for_loop_affine(const int it_start, const int it_end){
        double val   = 0;
        for(int ind=it_start;ind<it_end;++ind){
            u_(ind) = calc_u_bisection_affine(u_, f_, ind, val);
            errors_[ind] = fabs(val);
        }
    }

    /**
     * Computing error using errors_ vector
     * error = 1/n \sum_{x \in \mathcal{X}_h} |S_h(u,u(x),x)|
     * @return error of the algorithm
    */
    double compute_error(){
        const double ratio_error = 1.0;
        sort(errors_.begin(), errors_.end());
        double error = 0;
        for(int i=0;i<n_*n_*ratio_error;++i){
            error += errors_[i];
        }
        error /= n_*n_*ratio_error;
        return error;
    }

    /**
     * Performs a single iteration for the monotone discretization scheme for
     * motion by curvature PDE in 3D Cartesian grids. Given a numpy array that you
     * want to be updated ``out_np``, the function will run bisection method at each 
     * x in 3D grids. At the end, it will update ``out_np`` and return the error value
     * which is defined as in the paper.
     * @param out_np : numpy array coming from Python codes.
     * @return error value and out_np will be updated as well.
    */
    double perform_one_iteration(py::array_t<double, py::array::c_style | py::array::forcecast> out_np){

        // keyboard interrupt
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();

        py::buffer_info out_buf      = out_np.request();
        double *out_dbl = static_cast<double *>(out_buf.ptr);
        
        // find the grids size from the numpy array (n x n x n)
        int n = out_buf.shape[0];
 
        // check if the size of the numpy array matches with the class's member variable
        if(n_ != n){
            py::print("ERROR OCCURED: The array size of u does not math with f");
            return -1;
        }

        // set the maximum number of iterations of the bisection method
        double tol_bisection = 1e-5;
        max_it_bisection_    = -log(tol_bisection)/log(2);
        
        // DoubleArray2D out(out_dbl,n);

        for(int ind=0;ind<n_*n_;++ind){
            u_(ind) = out_dbl[ind];
        }
           
        // run the iterations  
        std::vector<std::future<void> > changes(THREADS_);    
        for(int th=0;th<THREADS_;++th){  
            changes[th] = std::async(std::launch::async, &Affine2DSolver::compute_for_loop_affine, this, static_cast<int>(th*n_*n_/THREADS_), static_cast<int>((th+1)*n_*n_/THREADS_));
        }
        for(int th=0;th<THREADS_;++th){
            changes[th].get();
        } 
        for(int ind=0;ind<n_*n_;++ind){
            out_dbl[ind] = u_(ind);
        }
        return compute_error();
    }
};
#endif