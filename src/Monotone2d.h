/**
 * Monotone scheme to solve the eikonal equation in 2D Cartesian grids.
*/

#ifndef MONOTONE2D_H
#define MONOTONE2D_H

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


class Monotone2DSolver{
public:
    /**
     * Initializing member variables
    */
    DoubleArray2D u_; // used for computing the solution of the PDE
    DoubleArray2D f_; // used for computing the solution of the PDE
    std::vector<double> errors_; // the vector of size n_*n_ that will contain the error at each grid point
    std::vector<std::vector<double> > dir_vec_; // direction of stencils starting from (1,0)
    std::vector<double> dir_norm_vec_; // norms of stencils starting from (1,0)
    int n_; // size of the grid n_ x n_ x n_
    int st_size_; // stencil size 1,2, or 3
    int N_stencils_; // the number of elements in stencils
    int THREADS_; // # of threads in CPU
    double max_it_bisection_; // max iteration of bisection method

    std::vector<std::vector<double> > ind_vec_;
    
    /**
     * initializer
     */
    Monotone2DSolver()
    : n_(0), st_size_(0), N_stencils_(0), THREADS_(std::thread::hardware_concurrency()), max_it_bisection_(0) 
    {}

    /**
     * initializer
     * @param f_np : numpy array for the right hand side function
     * @param st_size : the size of the stencil. if st_size=1 -> 8 stencils, st_size=2 -> 16 stencils
     */
    Monotone2DSolver(py::array_t<double, py::array::c_style | py::array::forcecast> f_np, int st_size)
    : st_size_(st_size), THREADS_(std::thread::hardware_concurrency()){
        py::buffer_info f_buf = f_np.request();
        double *f_dbl         = static_cast<double *>(f_buf.ptr);

        n_ = f_buf.shape[0];

        // initialize u_ and f_
        f_.initialize(f_dbl,n_);
        u_.initialize(n_);

        // initialize vectors
        errors_.resize(n_*n_);

        // Construct the index array.
        ind_vec_.resize(n_*n_);
        for(int i=0;i<n_*n_;++i){
            double x = (i%n_ + 0.5)/n_;
            double y = (i/n_ + 0.5)/n_;
            ind_vec_[i] = {static_cast<double>(i), x*x + y*y};
        }
        sort(ind_vec_.begin(), ind_vec_.end(), [](const std::vector<double>& a, const std::vector<double>& b) {return a[1] < b[1]; });
    }

    virtual ~Monotone2DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
    }

    /** compute the number of threads to use for the algorithm */
    inline int get_num_threads(const int n) const{
        int num_threads = std::thread::hardware_concurrency();
        while(n % num_threads != 0){
            --num_threads;
        }
        return num_threads;
    }

    /**
     * In this function, given a displacement vector v=stencils_[d], it will check if it is 
     * eligible to be in the subdifferential set.
     * For every w=stencils_[it] such that v . w < 0, check if u(x-w) < u(x). If not return false.
     * @param uval_vec : vector of linear interpolations of u at the stencils
     * @param dir_vec  : size_t. The index for stencils_
     * @param c    : the value at u(k,i,j)
     * @param i,j: the indices i: y-aixs j: x-axis
     * @return true if stencils_[d] is in subdifferential set.
    */
    bool p_in_subdifferential(const std::vector<double>& uval_vec, const int d, const double c) const{
        for(int it=d-N_stencils_/4+1+N_stencils_, N_it=d+N_stencils_/4-1+N_stencils_; it<=N_it; ++it){ // dir = {x, y}
            int it0 = it % N_stencils_;
            if(uval_vec[it0] > c){
                return false;
            }
        }

        return true;
    }


    /**
     * Given a point x = (j,i), it finds the stencils in 1x1 square and 
     * computes the linear interpolations of the value $u$ along the stencils.
     * It is assuming the Dirichlet boundary condition i.e., u=0 at the boundary of the domain [0,1]^2.
    */
    
    void initialize_uval_dir_vec(vector<double>& uval_vec, const DoubleArray2D& utmp, const int i, const int j){

        vector<vector<int> > dir_vec_tmp = {{1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}, {1,-1}};
        vector<double>       uval_vectmp;

        for(auto dir : dir_vec_tmp){
            int jp = j + dir[0];
            int ip = i + dir[1];
            double uval = 0;
            if(check_inside_domain(ip,jp)){
                uval = utmp(ip,jp);
            }
            uval_vectmp.push_back(uval); 
        }

        for(size_t k=0,N=dir_vec_tmp.size();k<N;++k){
            double u1 = uval_vectmp[k];
            double u2 = uval_vectmp[(k+1) % N];

            for(int count=0;count<st_size_;++count){
                double lambda = 1.0 * count / (1.0*st_size_);
                int    ind    = k*st_size_ + count;
                double new_u  = (1.0-lambda) * u1 + lambda * u2;
                uval_vec[ind] = new_u;
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
    virtual double calc_u(const DoubleArray2D& utmp, const DoubleArray2D& f, const double c, const int ind){
        return 0;
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
    double calc_u_bisection(const DoubleArray2D& utmp, const DoubleArray2D& f, const int ind, double& val){
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
    void compute_for_loop(int it_start, const int it_end){
        double val   = 0;
        for(int ind=it_start;ind<it_end;++ind){
            int ind0 = ind_vec_[ind][0];
            u_(ind0) = calc_u_bisection(u_, f_, ind0, val);
            errors_[ind0] = fabs(val);
        }
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
    void compute_for_loop_with_bdry(double eps, int it_start, const int it_end){
        double val   = 0;
        for(int ind=it_start;ind<it_end;++ind){
            int ind0 = ind_vec_[ind][0];
            double x = (ind0%n_+0.5)/n_;
            double y = (ind0/n_+0.5)/n_;
            // if(pow(x-0.5,2) + pow(y-0.5,2) < pow(0.5-eps,2)){
            if(fabs(x-0.5)<=0.5-eps && fabs(y-0.5)<=0.5-eps){
                u_(ind0) = calc_u_bisection(u_, f_, ind0, val);
                errors_[ind0] = fabs(val);
            }
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
     * motion by curvature PDE in 2D Cartesian grids. Given a numpy array that you
     * want to be updated ``out_np``, the function will run bisection method at each 
     * x in 2D grids. At the end, it will update ``out_np`` and return the error value
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

        for(int ind=0;ind<n_*n_;++ind){
            u_(ind) = out_dbl[ind];
        }

        DoubleArray2D utmp(n_);
        for(int i=0;i<n_*n_;++i){ utmp(i) = u_(i); }
           
        // run the iterations  
        std::vector<std::future<void> > changes(THREADS_);    
        for(int th=0;th<THREADS_;++th){  
            changes[th] = std::async(std::launch::async, &Monotone2DSolver::compute_for_loop, this, static_cast<int>(th*n_*n_/THREADS_), static_cast<int>((th+1)*n_*n_/THREADS_));
        }
        for(int th=0;th<THREADS_;++th){
            changes[th].get();
        } 
        for(int ind=0;ind<n_*n_;++ind){
            out_dbl[ind] = u_(ind);
        }

        // initialize_uval_dir_vec(uval_vec, u_, 5, 5);
        // for(auto i : uval_vec){
        //     py::print(i, ", ");
        // }

        return compute_error();
    }

    /**
     * Performs a single iteration for the monotone discretization scheme for
     * motion by curvature PDE in 2D Cartesian grids. Given a numpy array that you
     * want to be updated ``out_np``, the function will run bisection method at each 
     * x in 2D grids. At the end, it will update ``out_np`` and return the error value
     * which is defined as in the paper.
     * @param out_np : numpy array coming from Python codes.
     * @param eps    : boundary width to be used.
     * @return error value and out_np will be updated as well.
    */
    double perform_one_iteration_with_bdry(py::array_t<double, py::array::c_style | py::array::forcecast> out_np, const double eps){

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

        for(int ind=0;ind<n_*n_;++ind){
            u_(ind) = out_dbl[ind];
        }

        DoubleArray2D utmp(n_);
        for(int i=0;i<n_*n_;++i){ utmp(i) = u_(i); }
           
        // run the iterations  
        std::vector<std::future<void> > changes(THREADS_);    
        for(int th=0;th<THREADS_;++th){  
            changes[th] = std::async(std::launch::async, &Monotone2DSolver::compute_for_loop_with_bdry, this, eps, static_cast<int>(th*n_*n_/THREADS_), static_cast<int>((th+1)*n_*n_/THREADS_));
        }
        for(int th=0;th<THREADS_;++th){
            changes[th].get();
        } 
        for(int ind=0;ind<n_*n_;++ind){
            out_dbl[ind] = u_(ind);
        }

        // initialize_uval_dir_vec(uval_vec, u_, 5, 5);
        // for(auto i : uval_vec){
        //     py::print(i, ", ");
        // }

        return compute_error();
    }
};
#endif