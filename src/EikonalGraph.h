/**
 * EikonalGraph.h
 * Eikonal equation solver using the subdifferential set.
 * Solving the equation on the graph in any dimensional space d>=2.
 *  Created on: Dec 1, 2022
 *      Author: Wonjun Lee
 */

#ifndef EIKONALGRAPH_H
#define EIKONALGRAPH_H
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <future>


bool comparison(Idx2Value& i, Idx2Value& j);

class EikonalGraph{
public:
    double delta_;      // the small ball excluded from the neighbor set N(x)
    double tol_bis_;    // tolerance of the bisection method
    int max_it_bis_;    // max iterations of the bisection method

    int THREADS_;       // the number of threads

    int DIM_;           // dimension
    int n_;             // the number of vertices
    int m_;             // the number of edges

    XArray X_; // [n_ * DIM_] coordinates array 
    MyArray<double> V_; // [M_]   indicates the distance edge -> norm
    MyArray<int>    I_; // [M_]   indicates of vertices from neighbor points edge -> node I[K[i]] ... I[K[i+1]] = indices of neighbor points from x_i
    MyArray<int>    K_; // [n_+1] indicates the neighbor indices

    std::vector<double> u_; // [n_] solution array
    MyArray<double> f_; // [n_] the right hand side function
    MyArray<int>    bdry_; // indices of boundary points
    std::vector<std::vector<double> > ind_vec_;
    std::vector<std::vector<int> > edge2edge_; // for each index of X, contains a vector of indices such that x \cdot y > 0
    
    /**
     * constructor
     * The python array will be accepted as double* or int* arrays in c++.
     * @param X : coordinates array of points
     * @param V : distance array
     * @param I : indices of vertices from neighbor points
     * @param K : indices of neighbor points
     */
    EikonalGraph(py::array_t<double, py::array::c_style | py::array::forcecast> X, 
                 py::array_t<double, py::array::c_style | py::array::forcecast> V, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> I, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> K,
                 py::array_t<double, py::array::c_style | py::array::forcecast> f_np, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> bdry_np)
    :delta_(0.4), tol_bis_(1e-4), max_it_bis_(40), THREADS_(std::thread::hardware_concurrency())
    {
        py::buffer_info X_buf = X.request();
        py::buffer_info V_buf = V.request();
        py::buffer_info I_buf = I.request();
        py::buffer_info K_buf = K.request();
        py::buffer_info f_buf = f_np.request();
        py::buffer_info bdry_buf = bdry_np.request();
        double *X_dbl      = static_cast<double *>(X_buf.ptr);
        double *V_dbl      = static_cast<double *>(V_buf.ptr);
        int *I_dbl         = static_cast<int *>(I_buf.ptr);
        int *K_dbl         = static_cast<int *>(K_buf.ptr);
        double *f_dbl      = static_cast<double *>(f_buf.ptr);
        int *bdry_dbl      = static_cast<int *>(bdry_buf.ptr);
        n_   = X_buf.shape[0];
        DIM_ = X_buf.shape[1];
        m_   = V_buf.shape[0];
        X_.initialize(X_dbl, n_, DIM_);
        V_.initialize(V_dbl, m_);
        I_.initialize(I_dbl, m_);
        K_.initialize(K_dbl, n_+1);
        f_.initialize(f_dbl, n_);
        bdry_.initialize(bdry_dbl, bdry_buf.shape[0]);

        u_.resize(n_);

        /* find the boundary points */
        std::set<int> bdry_points;
        for(int i=0;i<bdry_.n_;++i){
            bdry_points.insert(bdry_[i]);
        }

        std::vector<int> interior_points;
        for(int x_i=0;x_i<n_;++x_i){
            if(bdry_points.find(x_i)==bdry_points.end()){
                interior_points.push_back(x_i);
            }
        } 

        ind_vec_.resize(interior_points.size());
        for(int i=0, N=interior_points.size();i<N;++i){
            double norm = 0;
            for(int d=0;d<DIM_;++d){
                norm += pow(X_(interior_points[i],d) - 0.5, 2);
            }
            ind_vec_[i] = {static_cast<double>(interior_points[i]), norm};
        }
        sort(ind_vec_.begin(), ind_vec_.end(), [](const std::vector<double>& a, const std::vector<double>& b) {return a[1] < b[1]; });

        // initialize edge2edge vector
        edge2edge_.resize(m_);
        int count = 0;
        for(int x_0=0;x_0<n_;++x_0){
            double delta = 0;
            for(int a=K_(x_0); a<K_(x_0+1); ++a){
                delta = fmax(delta, V_(a));
            }
            delta *= delta_;
            for(int a=K_(x_0); a<K_(x_0+1); ++a){
                int x_a=I_(a);
                for(int b=K_(x_0); b<K_(x_0+1); ++b){
                    if(V_(b) >= delta){
                        int x_b=I_(b);
                        if(dot(x_0, x_a, x_b) > 0){
                            edge2edge_[a].push_back(b);
                            count++;
                        }
                    }
                }
            }
        }

        py::print("interior:", interior_points.size(), "threads: ", THREADS_, "count:", count);


    }

    virtual ~EikonalGraph(){}

    /**
     * find the dot product between (x_j - x_i) . (x_k - x_i)
     * @param x_i : center point
     * @param x_j : neighbor 1
     * @param x_k : nieghbor 2
     * @return the dot product between (x_j - x_i) . (x_k - x_i)
     */
    double dot(const int x_i, const int x_j, const int x_k) const{
        double s = 0;
        for(int d=0; d<DIM_; ++d){
           s += (X_(x_j, d) - X_(x_i, d)) * (X_(x_k, d) - X_(x_i, d));
        }
        return s;
    }

    /**
     * check if (x_j - x_i) is in the subdifferential set
     * @param u : values of u
     * @param ux_i : given value at u[x_i]
     * @param x_i : index of the center point
     * @param x_j : index of the neighbor point
     * @return boolean
     */
    bool is_subdifferential(const std::vector<double>& u, const double ux_i, const int x_i, const int x_j) const{
        double delta = 0;
        for(int k=K_(x_i); k<K_(x_i+1); ++k){
            delta = fmax(delta, V_(k));
        }
        delta *= delta_;

        // for each neighborhood point of x_i
        for(int k=K_(x_i); k<K_(x_i+1); ++k){
            // check if distance > delta_
            if(V_(k) >= delta){
                int x_k = I_(k);
                // check if p \cdot q < 0
                if(dot(x_i, x_j, x_k) > 0){
                    // check if u is smaller at x_k. if small then return false
                    if(ux_i < u[x_k]){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * check if (x_j - x_i) is in the subdifferential set
     * @param u : values of u
     * @param ux_i : given value at u[x_i]
     * @param x_i : index of the center point
     * @param a   : index of the edge
     * @return boolean
     */
    bool is_subdifferential_with_edge2edge(const std::vector<double>& u, const double ux_i, const int x_i, const int a) const{
        // for each neighborhood point of x_i, get x_b such that (x_b - x_i) \cdot (x_a - x_i) > 0
        for(int b=0,N=edge2edge_[a].size();b<N;++b){
            int x_b = I_(edge2edge_[a][b]);
            // check if u is smaller at x_k. if small then return false
            if(ux_i < u[x_b]){
                return false;
            }
        }
        return true;
    }

    /**
     * compute the right hand side function.
     * @param x_i : center position
     * @param x_j : direction to the subdifferential vector
     * @return value of f
    */
    virtual double compute_fval(const MyArray<double>& f, const int x_i, const int x_j) const{
        return f[x_i];
    }

    /**
     * compute the value of the scheme S_h(u, u(x), x)
     * @param u : u
     * @param ux_i : u(x)
     * @param x_i : x
     * @return S_h(u,u(x),x)
    */
    double compute_value(const std::vector<double>& u, const double ux_i, const int x_i) const{
        double max_val = -100;
        for(int a=K_(x_i); a<K_(x_i+1); ++a){
            int x_a = I_(a); // index of a neighborhood point 
            // check if the neighbor point x_j is in subdifferential set
            if(u[x_a] <= ux_i){
                if(is_subdifferential_with_edge2edge(u, ux_i, x_i, a)){
                    // compute |\nabla u|
                    double uval = (ux_i - u[x_a])/V_(a);
                    // compute the right hand side
                    double fval = compute_fval(f_, x_i, x_a);
                    max_val = fmax(max_val,  uval - fval);
                }
            }
        }
        return max_val;
    }
    
    double bisection(double& val, const std::vector<double>& u, const int x_i) {
        double a=0, b=1, 
        c=(a+b)*0.5;

        for(int it=0;it<max_it_bis_;++it){
            val = compute_value(u,c,x_i);
            if(val > 0){ b = c; }
            else       { a = c; }
            c = (a+b)*0.5;
            // if(fabs(val) < tol_bis_) break;
        }
        return c;
    }

    double compute_for_loop(const int start_ind, const int end_ind){
        double error = 0;
        for(int i=start_ind;i<end_ind;++i){
            double val = 0;
            int x_i = ind_vec_[i][0];
            double utmp = u_[x_i];
            u_[x_i] = bisection(val,u_,x_i);
            error += fabs(u_[x_i] - utmp);
        }
        return error;
    }

    /**
     * Gauss-Seidel with sort every iteration
     */
    double iterate(py::array_t<double, py::array::c_style | py::array::forcecast> u_np){
        
        py::buffer_info data_buf = u_np.request();
        double *data = static_cast<double *>(data_buf.ptr);
        for(int i=0;i<n_;++i){
            u_[i] = data[i];
        }

        std::vector<std::future<double> > changes(THREADS_); 
        int N_int = ind_vec_.size();
        for(int th=0;th<THREADS_;++th){
            changes[th] = std::async(std::launch::async, &EikonalGraph::compute_for_loop, this, static_cast<int>(th*N_int/THREADS_), static_cast<int>((th+1)*N_int/THREADS_));
        }
        double error = 0;
        for(int th=0;th<THREADS_;++th){
            error += changes[th].get();
        }
        for(int i=0;i<n_;++i){
            data[i] = u_[i];
        }
        return error / n_;
    }
};
#endif