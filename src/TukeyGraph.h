/**
 * TukeyGraph.h
 * Eikonal equation solver using the subdifferential set.
 * Solving the equation on the graph in any dimensional space d>=2.
 *  Created on: Dec 1, 2022
 *      Author: Wonjun Lee
 */

#ifndef TUKEYGRAPH_H
#define TUKEYGRAPH_H
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
#include "EikonalGraph.h"
#include <string>

class TukeyGraph : public EikonalGraph{
public:
    std::string type_;

    /**
     * constructor
     * The python array will be accepted as double* or int* arrays in c++.
     * @param X : coordinates array of points
     * @param V : distance array
     * @param I : indices of vertices from neighbor points
     * @param K : indices of neighbor points
     */
    TukeyGraph(py::array_t<double, py::array::c_style | py::array::forcecast> X, 
                 py::array_t<double, py::array::c_style | py::array::forcecast> V, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> I, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> K,
                 py::array_t<double, py::array::c_style | py::array::forcecast> f_np, 
                 py::array_t<int, py::array::c_style | py::array::forcecast> bdry_np,
                 std::string type = "square")
    :EikonalGraph(X,V,I,K,f_np,bdry_np)
    {
        type_ = type;
    }

    ~TukeyGraph(){}

    double compute_exact_integral_circle(const MyArray<double>& f, const int x_i, const int x_j) const{
        const double R = 0.4;
        std::vector<double> p(DIM_);
        double p_norm = 0;
        for(int d=0;d<DIM_;++d){
            double val = X_(x_j,d) - X_(x_i,d);
            p[d]       = val;
            p_norm    +=val*val;
        }
        p_norm = sqrt(p_norm);

        double xzp = 0;
        for(int d=0;d<DIM_;++d){
            xzp += (X_(x_i,d) - 0.5) * p[d]/p_norm;
        }
        double r = sqrt(R*R - xzp*xzp);
        if(DIM_ == 2){
            return 2 * r;
        }else if(DIM_ == 3){
            return M_PI * pow(r,2);
        }else if(DIM_ == 4){
            return 4.189 * pow(r,3);
        }else if(DIM_ == 6){
            return 5.264 * pow(r,5);
        }else if(DIM_ == 10){
            return 3.299 * pow(r,9);
        }
        return 0;
    }

    double compute_exact_integral_square_2d(const MyArray<double>& f, const int x_i, const int x_j) const{
        std::vector<double> p(DIM_);
        // initialize the vector from the subdifferential set
        for(int d=0;d<DIM_;++d){
            p[d] = X_(x_j,d) - X_(x_i,d);
        }
        std::swap(p[0],p[1]); p[0] *= -1;

        const double B = 1.0;
        const double A = 0.0;
        
        double x0 = X_(x_i,0);
        double y0 = X_(x_i,1);

        double val = 0;

        if(fabs(x0-0.5)<(B-A)/2 && fabs(y0-0.5)<(B-A)/2){

            if(fabs(p[0]) < 1e-8) return B-A;

            double slope = 1.0*p[1]/p[0];

            double xa,xb,ya,yb;
            xb = B;
            yb = (xb-x0)*slope + y0;
            if(yb>B){
                yb = B;
                xb = (yb-y0)/slope + x0;
            }else if(yb<A){
                yb = A;
                xb = (yb-y0)/slope + x0;
            }
            xa = A;
            ya = (xa-x0)*slope + y0;
            if(ya < A){
                ya = A;
                xa = (ya-y0)/slope + x0;
            }else if(ya>B){
                ya = B;
                xa = (ya-y0)/slope + x0;
            }
            val = sqrt((yb-ya)*(yb-ya)+(xb-xa)*(xb-xa));
        }
        return val;
    }

    /**
     * compute the right hand side function.
     * @param x_i : center position
     * @param x_j : direction to the subdifferential vector
     * @return value of f
    */
    virtual double compute_fval(const MyArray<double>& f, const int x_i, const int x_j) const{
        double fval = 0;
        if(type_ == "circle"){
            fval = compute_exact_integral_circle(f,x_i,x_j);
        }else if(type_ == "square"){
            fval = compute_exact_integral_square_2d(f,x_i,x_j);
        }
        return fval;
    }
};
#endif