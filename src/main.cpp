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

// #include "Eikonal2d.h"
// #include "Tukey2d.h"
// #include "Affine2d.h"
// #include "Curvature2d.h"
// #include "Curvature3d.h"

namespace py = pybind11;
using namespace std;

#include <vector>

/**
 * Computing dot products given two vectors of any types.
 * @param v : first vector
 * @param w : second vector
 * @return the dot product v.w
*/
template <typename T>
double dot(const std::vector<T>& v, const std::vector<T>& w){
    T s = 0;
    for(int i=0,N=v.size();i<N;++i){
        s += v[i] * w[i];
    }
    return s;
}

/**
 * This is a custom class for taking care of the double array in 3D.
 * Note that this class assume the array has the same size for x,y,z-axes.
 * Here is a simple example of how to use the class
 *     DoubleArray3D a(v,n); // where v is double * and n is the number of points in x-axis.
 *     printf(a(1,2,3)); // this will print out the element at (x=3, y=2, z=1)
 *     a(1,2,3) = 3; // this will assign the value at (x=3, y=2, z=1).
*/
class DoubleArray3D{
public:
    double *data_;
    int     n_;
    DoubleArray3D()                    :data_(nullptr),n_(0) {}
    DoubleArray3D(double* v, int n)    :data_(v)      ,n_(n) {}
    DoubleArray3D(int n)               :data_(nullptr),n_(n) {
        initialize(n);
    }
    void initialize(const int n){
        n_ = n;
        data_ = new double[n_*n_*n_];
        for(int ind=0;ind<n_*n_*n_;++ind){
            data_[ind]=0.0;
        }
    }
    void initialize(double* v, int n){
        if(data_ != nullptr){
            delete [] data_;
        }
        data_ = v;
        n_    = n;
    }
    #ifdef _DEBUG
        inline double&  operator()(const int k, const int i, const int j){
            int ind = k*n_*n_+i*n_+j;
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + k*n_*n_+i*n_+j);
        };
        inline const double&  operator()(const int k, const int i, const int j) const{
            int ind = k*n_*n_+i*n_+j;
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + k*n_*n_+i*n_+j);
        };

        inline double&  operator()(const int ind){
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const{
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + ind);
        };
    #else
        inline double&  operator()(const int k, const int i, const int j){
            return *(data_ + k*n_*n_+i*n_+j);
        };
        inline const double&  operator()(const int k, const int i, const int j) const
        {
        return *(data_ + k*n_*n_+i*n_+j);
        };

        inline double&  operator()(const int ind)
        {
        return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const
        {
        return *(data_ + ind);
        };
    #endif    
};


/**
 * This is a custom class for taking care of the double array in 3D.
 * Note that this class assume the array has the same size for x,y,z-axes.
 * Here is a simple example of how to use the class
 *     DoubleArray2D a(v,n); // where v is double * and n is the number of points in x-axis.
 *     printf(a(1,2,3)); // this will print out the element at (x=3, y=2, z=1)
 *     a(1,2,3) = 3; // this will assign the value at (x=3, y=2, z=1).
*/
class DoubleArray2D{
public:
    double *data_;
    int     n_;
    DoubleArray2D()                    :data_(nullptr),n_(0) {}
    DoubleArray2D(double* v, int n)    :data_(v)      ,n_(n) {}
    DoubleArray2D(int n)               :data_(nullptr),n_(n) {
        initialize(n);
    }
    void initialize(const int n){
        n_ = n;
        data_ = new double[n_*n_];
        for(int ind=0;ind<n_*n_;++ind){
            data_[ind]=0.0;
        }
    }
    void initialize(double* v, int n){
        if(data_ != nullptr){
            delete [] data_;
        }
        data_ = v;
        n_    = n;
    }
    #ifdef _DEBUG
        inline double&  operator()(const int i, const int j){
            int ind = i*n_+j;
            assertm(ind>=0 && ind<n_*n_, "out of bound");
            return *(data_ + i*n_+j);
        };
        inline const double&  operator()(const int i, const int j) const{
            int ind = i*n_+j;
            assertm(ind>=0 && ind<n_*n_, "out of bound");
            return *(data_ + i*n_+j);
        };

        inline double&  operator()(const int ind){
            assertm(ind>=0 && ind<n_*n_, "out of bound");
            return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const{
            assertm(ind>=0 && ind<n_*n_, "out of bound");
            return *(data_ + ind);
        };
    #else
        inline double&  operator()(const int i, const int j){
            return *(data_ + i*n_+j);
        };
        inline const double&  operator()(const int i, const int j) const
        {
        return *(data_ + i*n_+j);
        };

        inline double&  operator()(const int ind)
        {
        return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const
        {
        return *(data_ + ind);
        };
    #endif  
};

/**
 * The base class for solving curvature PDEs in 2D Cartesian grids
*/
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
    Monotone2DSolver(py::array_t<double>& f_np, int st_size)
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
    double perform_one_iteration(py::array_t<double>& out_np){

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
    double perform_one_iteration_with_bdry(py::array_t<double>& out_np, const double eps){

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

        return compute_error();
    }
};

/**
 * Derived class: solving a simple Eikonal equation
 * |\nabla u(x)| = f(x) with homogeneou Dirichlet boundary condition
*/
class Eikonal2DSolver : public Monotone2DSolver {
public:    
    /**
     * initializer
     * @param f_np : numpy array for the right hand side function
     * @param st_size : the size of the stencil. if st_size=1 -> 8 stencils, st_size=2 -> 16 stencils
     */
    Eikonal2DSolver(py::array_t<double>& f_np, int st_size)
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

        py::print("Constructor finished. n: ", n_, "stencil size: ", st_size, "number of stencils: ", N_stencils_, "number of threads: ", THREADS_);
    }

    ~Eikonal2DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
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
                double fval = f(i,j);                
                double val = Du_val - fval;
                if(val > max_val){ max_val = val; }
            }
        }
        return max_val;
    }
};

/**
 * Derived class: solving a Tukey depth Eikonal equation
 * |\nabla u(x)| = \int_{(y-x)\cdot \nabla u(x) = 0} \rho(y) dS(y)
*/
class Tukey2DSolver : public Monotone2DSolver {
public:
    
    std::vector<std::vector<double> > rhs_integral_vec_;

    /**
     * initializer
     * @param f_np : numpy array for the right hand side function
     * @param st_size : the size of the stencil. if st_size=1 -> 8 stencils, st_size=2 -> 16 stencils
     */
    Tukey2DSolver(py::array_t<double>& f_np, int st_size)
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

    virtual ~Tukey2DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
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

/**
 * Derived class: solving a Curvature PDE
 * |\nabla u(x)|\kappa(x) = f(x) with homogeneou Dirichlet boundary condition
*/
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
    Curv2DSolver(py::array_t<double>& f_np, py::array_t<int>& stencils_np, int st_size)
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

    virtual ~Curv2DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
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

/**
 * Derived class: solving a affine flows PDEs
 * |\nabla u(x)|\kappa(x)^{1/3} = f(x) with homogeneou Dirichlet boundary condition
*/
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
    Affine2DSolver(py::array_t<double>& f_np, py::array_t<int>& stencils_np, int st_size)
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

    virtual ~Affine2DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
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
    double perform_one_iteration(py::array_t<double>& out_np){

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

/**
 * Derived class: solving a curvature PDE on 3D Cartesian grids
 * |\nabla u(x)| = f(x) with homogeneou Dirichlet boundary condition
*/
class Curv3DSolver{
public:
    /**
     * Initializing member variables
    */
    DoubleArray3D u_; // used for computing the solution of the PDE
    DoubleArray3D f_; // used for computing the solution of the PDE
    std::vector<double> stencils_norm_; // the vector of norms of the stencils
    std::vector< std::vector<int> > stencils_; // vector of stencils ex: {{0,0,1}, {0,1,1}, ... }

    int n_; // size of the grid n_ x n_ x n_
    int st_size_; // stencil size 1,2, or 3
    int st_N_; // the number of elements in stencils
    int THREADS_; // # of threads in CPU
    double max_it_bisection_; // max iteration of bisection method
    
    /**
     * initializer
     * @param n_ : grid size of x-axis
     */
    Curv3DSolver(int n)
    : n_(n), st_size_(0), THREADS_(std::thread::hardware_concurrency()){
    }

    /**
     * initializer
     * @param n_ : grid size of x-axis
     */
    Curv3DSolver(py::array_t<double>& f_np, py::array_t<int>& stencils_np, int st_size)
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

        // resizing stencils and stencils norm
        stencils_.resize(st_N_);
        stencils_norm_.resize(st_N_);

        // converting from int* -> vector<vector<int>>
        int dim = 3;
        for(int it=0;it<st_N_;++it){
            stencils_[it].resize(dim); // 3d vector e.g. {0,0,1}
            double norm_val = 0;
            for(int it1=0;it1<dim;++it1){
                stencils_[it][it1] = stencils[it*dim + it1];
                norm_val += stencils_[it][it1] * stencils_[it][it1];
            }
            stencils_norm_[it] = sqrt(norm_val) / n_;
        }

        py::print("Constructor finished. n: ", n_, "stencil size: ", st_size, "number of stencils: ", stencils_.size());
    }

    virtual ~Curv3DSolver(){
        if(u_.data_ !=  nullptr) delete [] u_.data_;
    }

    /**
     * In this function, given a displacement vector v=stencils_[d], it will check if it is 
     * eligible to be in the subdifferential set.
     * For every w=stencils_[it] such that v . w < 0, check if u(x-w) < u(x). If not return false.
     * @param utmp : DoubleArray3D for the solution
     * @param d    : size_t. The index for stencils_
     * @param c    : the value at u(k,i,j)
     * @param k,i,j: the indices
     * @return true if stencils_[d] is in subdifferential set.
    */
    bool p_in_subdifferential(const DoubleArray3D& utmp, const size_t& d, const double c, const int k, const int i, const int j) const{
        int kp = k - stencils_[d][2];
        int ip = i - stencils_[d][1];
        int jp = j - stencils_[d][0];
        if(check_inside_domain(kp,ip,jp)){
            if(utmp(kp,ip,jp) > c){
                return false;
            }
        }

        for(int it=0, N_it=stencils_.size(); it<N_it; ++it){ // dir = {x, y}
            int kp = k - stencils_[it][2];
            int ip = i - stencils_[it][1];
            int jp = j - stencils_[it][0];
            if(check_inside_domain(kp,ip,jp)){
                if(dot(stencils_[it], stencils_[d]) > 0){
                    if(utmp(kp,ip,jp) > c){
                        return false;
                    }
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
    double compute_second_derivative_given_p(const DoubleArray3D& utmp, vector<int>& q, const double c, const int k, const int i, const int j) const{
        int km = k-q[2]; int kp = k+q[2];
        int im = i-q[1]; int ip = i+q[1];
        int jm = j-q[0]; int jp = j+q[0];
        double umm = 0;
        double upp = 0;
        
        if(check_inside_domain(kp,ip,jp)){
            upp = utmp(kp,ip,jp);
        }
        if(check_inside_domain(km,im,jm)){
            umm = utmp(km,im,jm);
        }
        double h2 = dot(q,q)/(n_*n_); // norm of p : |p| * (dx^2)
        return (- umm + 2.0 * c - upp) / h2;
    }

    /**
     * calculating affine flows
     */
    double calc_val_affine(const DoubleArray3D& utmp, const DoubleArray3D& f, const double c, const int ind){
        double max_val = -1e4;

        int k = ind / (n_*n_);
        int i = (ind % (n_*n_)) / n_;
        int j = (ind % (n_*n_)) % n_;

        for(size_t d=0, N_d=stencils_.size(); d<N_d; ++d){ // dir = {x, y}
            // choose a eligible vector from stencils
            if(p_in_subdifferential(utmp,d,c,k,i,j)){
                vector<int> p = stencils_[d];
                vector<int> q1 = {-p[1], p[0],0};
                vector<int> q2 = {-p[2],    0,p[0]};
                // q1,q2 = orthogonal to p
                int rand_n = rand();
                if( rand_n % 3 == 0){
                    if(p[0] == 0 && p[1] == 0){
                        q1 = {1,0,0};
                        q2 = {0,1,0};
                    }
                    if(p[0] == 0 && p[2] == 0){
                        q1 = {1,0,0};
                        q2 = {0,0,1};
                    }
                }else if(rand_n % 3 == 1){
                    q1 = {p[2],    0, -p[0]};
                    q2 = {   0, p[2], -p[1]};
                    if(p[0] == 0 && p[2] == 0){
                        q1 = {0,0,1};
                        q2 = {1,0,0};
                    }
                    if(p[1] == 0 && p[2] == 0){
                        q1 = {0,0,1};
                        q2 = {0,1,0};
                    }
                }else{
                    q1 = {p[1], -p[0],    0};
                    q2 = {   0, -p[2], p[1]};
                    if(p[0] == 0 && p[1] == 0){
                        q1 = {0,1,0};
                        q2 = {1,0,0};
                    }
                    if(p[1] == 0 && p[2] == 0){
                        q1 = {0,1,0};
                        q2 = {0,0,1};
                    }
                }
                double val = compute_second_derivative_given_p(utmp, q1, c, k, i, j)
                            +compute_second_derivative_given_p(utmp, q2, c, k, i, j);
                if(val > max_val){ max_val = val; }
            }
        }
        return max_val - f(ind);
    }

    inline bool check_inside_domain(const int k, const int i, const int j) const{
        return k>=0 && k<n_ && i>=0 && i<n_ && j>=0 && j<n_;
    }

    /**
     * Given functions and a location at x, it will perform a bisection method at x using
     * the monotone discretization function.
     * @param utmp : DoubleArray3D solution array
     * @param f    : DoubleArray3D the right hand side function
     * @param ind  : the index of the grid. The location x
     * @param val  : It will compute the value of the HJ equation. Being close to 0 is better.
     * @return c : this will be the new updated value for utmp(x).
    */
    double calc_u_bisection_affine(const DoubleArray3D& utmp, const DoubleArray3D& f, const int ind, double& val){
        // py::print("inside the bisection function\n");
        double a = 0;
        double b = 1;
        double c = (a+b)*0.5;
        val = 0;
        for(size_t i_bi=0; i_bi<max_it_bisection_; ++i_bi){
            val = calc_val_affine(utmp, f, c, ind);
            if(val > 0){ b = c; }
            else       { a = c; }
            c = (a+b)*0.5;
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
    double compute_for_loop_affine(const int it_start, const int it_end){
        double error = 0;
        double val = 0;
        for(int ind=it_start;ind<it_end;++ind){
            u_(ind) = calc_u_bisection_affine(u_, f_, ind, val);
            error += fabs(val);
        }
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
    double perform_one_iteration(py::array_t<double>& out_np){

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
        
        DoubleArray3D out(out_dbl,n);

        for(int ind=0;ind<n_*n_*n_;++ind){
            u_(ind) = out(ind);
        }
           
        // run the iterations  
        // double error = compute_for_loop_affine(utmp,f,0,n_*n_*n_);
        double error = 0;
        std::vector<std::future<double> > changes(THREADS_);    
        for(int th=0;th<THREADS_;++th){  
            changes[th] = std::async(std::launch::async, &Curv3DSolver::compute_for_loop_affine, this, static_cast<int>(th*n_*n_*n_/THREADS_), static_cast<int>((th+1)*n_*n_*n_/THREADS_));
        }
        for(int th=0;th<THREADS_;++th){
            error += changes[th].get();
        } 
        for(int ind=0;ind<n_*n_*n_;++ind){
            out(ind) = u_(ind);
        }
        return error/(n_*n_*n_);
    }
};

void interpolate(py::array_t<double>& u_np, py::array_t<double>& u_small_np){
    py::buffer_info u_buf       = u_np.request();
    py::buffer_info u_small_buf = u_small_np.request();
    double *u_dbl       = static_cast<double *>(u_buf.ptr);
    double *u_small_dbl = static_cast<double *>(u_small_buf.ptr);
    
    // find the grids size from the numpy array (n x n)
    int n       = u_buf.shape[0];
    int n_small = n/2;

    for(int i=0;i<n_small;++i){
        for(int j=0;j<n_small;++j){
            u_dbl[2*i*n+2*j] = u_small_dbl[i*n_small+j];
        }
    }

    for(int i=0;i<n_small;++i){
        for(int j=0;j<n_small;++j){
            double val = 0.5*(u_small_dbl[i*n_small+j] + u_small_dbl[i*n_small+(int) fmin(j+1,n-1)]);
            u_dbl[2*i*n+(2*j+1)] = val;
        }
    }

    for(int i=0;i<n_small;++i){
        for(int j=0;j<n_small;++j){
            double val = 0.5*(u_small_dbl[i*n_small+j] + u_small_dbl[(int) fmin(i+1, n-1)*n_small+j]);
            u_dbl[(2*i+1)*n+2*j] = val;
        }
    }

    for(int i=0;i<n_small;++i){
        for(int j=0;j<n_small;++j){
            int ip = fmin(i+1, n-1);
            int jp = fmin(j+1, n-1);
            double val = 0.25*(u_small_dbl[i *n_small+ j]
                             + u_small_dbl[ip*n_small+ j]
                             + u_small_dbl[i *n_small+jp]
                             + u_small_dbl[ip*n_small+jp]);
            u_dbl[(2*i+1)*n+(2*j+1)] = val;
        }
    }
}

PYBIND11_MODULE(MonotoneScheme, m) {
    // optional module docstring
    m.doc() = "C++ wrapper for monotone discretization code in 2D and 3D Cartesian grids setting";

    m.def("interpolate", &interpolate);
 
    py::class_<Eikonal2DSolver>(m, "Eikonal2DSolver") 
        .def(py::init<py::array_t<double> &, int>())
        .def("perform_one_iteration", &Eikonal2DSolver::perform_one_iteration);

    py::class_<Tukey2DSolver>(m, "Tukey2DSolver") 
        .def(py::init<py::array_t<double> &, int>())
        .def("perform_one_iteration", &Tukey2DSolver::perform_one_iteration)
        .def("perform_one_iteration_with_bdry", &Tukey2DSolver::perform_one_iteration_with_bdry);

    py::class_<Affine2DSolver>(m, "Affine2DSolver") 
        .def(py::init<py::array_t<double> &, py::array_t<int> &, int>())
        .def("perform_one_iteration", &Affine2DSolver::perform_one_iteration);

    py::class_<Curv2DSolver>(m, "Curv2DSolver") 
        .def(py::init<py::array_t<double> &, py::array_t<int> &, int>())
        .def("perform_one_iteration", &Curv2DSolver::perform_one_iteration);

    py::class_<Curv3DSolver>(m, "Curv3DSolver") 
        .def(py::init<py::array_t<double> &, py::array_t<int> &, int>())
        .def("perform_one_iteration", &Curv3DSolver::perform_one_iteration);
}