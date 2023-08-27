#ifndef HELPER_H
#define HELPER_H

#include <iostream>
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


#endif