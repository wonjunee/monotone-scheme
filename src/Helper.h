#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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



struct Idx2Value{
    int idx_;
    double val_;
    Idx2Value() : idx_(0), val_(0) {}
};

bool comparison(Idx2Value& i, Idx2Value& j){
    return i.val_ < j.val_;
}

template <typename T>
class myarray{
public:
    int n_;
    T* data_;
    bool to_delete_;

    myarray():              n_(0), data_(nullptr),  to_delete_(false) {}
    myarray(int n, T* arr): n_(n), data_(arr),      to_delete_(false) {}
    myarray(py::array_t<T, py::array::c_style | py::array::forcecast>  arr){
        py::buffer_info data_buf = arr.request();
        n_    = data_buf.shape[0];
        data_ = static_cast<T*>(data_buf.ptr);
        to_delete_ = false;
    }
    myarray(int n, py::buffer_info &  data_buf){
        n_ = n;
        data_ = static_cast<T*>(data_buf.ptr);
        to_delete_ = false;
    }
    

    // virtual ~myarray(){
    //     if(to_delete_){
    //         delete [] data_;
    //     }
    // }

    void from_numpy(py::array_t<T, py::array::c_style | py::array::forcecast>  arr, size_t n){
        py::buffer_info data_buf = arr.request();
        data_ = static_cast<T*>(data_buf.ptr);
        n_   = n;
    }
 
    // ~myarray(){
    //     printf("deleteing...");
    //     if(to_delete_){
    //         printf(" deleted ");
    //         delete[] data_;
    //     }
    //     printf("done \n");
    // }

    int size() const{
        return n_;
    }

    T* data() const{
        return data_;
    }

    T operator[](int i) const{ 
#ifdef _DEBUG
        if(i >= n_){
            std::cout << "wrong " << i << " "  << n_ << "\n";
            exit(1);
        }
#endif
        return data_[i]; 
    }

    T & operator [](int i) {
#ifdef _DEBUG
        if(i >= n_){
            std::cout << "wrong " << i << " "  << n_ << "\n";
            exit(1);
        }
#endif
        return data_[i];
    }
};


template <typename T>
class MyArray{
public:
    T *data_;
    int     n_;
    MyArray()               :data_(nullptr),n_(0) {}
    MyArray(T* v, int n)    :data_(v)      ,n_(n) {}
    void initialize(T* v, int n){
        data_ = v;
        n_    = n;
    }
    inline T&  operator()(const int ind)
    {
        if(ind<0 || ind>=n_){
            py::print("error in indices X");
            exit(1);
        }
    return *(data_ + ind);
    };
    inline const T&  operator()(const int ind) const
    {
        if(ind<0 || ind>=n_){
            py::print("error in indices X");
            exit(1);
        }
    return *(data_ + ind);
    };

    T operator[](int ind) const{ 
        return *(data_ + ind); 
    }
};

class XArray{
public:
    double *data_;
    int     n_;
    int     DIM_;
    XArray()                             :data_(nullptr),n_(0), DIM_(0) {}
    XArray(double* v, int n, int DIM)    :data_(v)      ,n_(n), DIM_(DIM) {}
    void initialize(double* v, int n, int DIM){
        if(data_ != nullptr){
            delete [] data_;
        }
        data_ = v;
        n_    = n;
        DIM_  = DIM;
    }
    #ifdef _DEBUG
        inline double&  operator()(const int ind){
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const{
            assertm(ind>=0 && ind<n_*n_*n_, "out of bound");
            return *(data_ + ind);
        };
    #else
        inline double&  operator()(const int i, const int d)
        {
            if(i<0 || i>=n_ || d<0 || d>=DIM_){
                py::print("error in indices X");
                exit(1);
            }
            return *(data_ + i * DIM_ + d);
        };
        inline const double&  operator()(const int i, const int d) const
        {
            if(i<0 || i>=n_ || d<0 || d>=DIM_){
                py::print("error in indices X");
                exit(1);
            }
            return *(data_ + i * DIM_ + d);
        };

        inline double&  operator()(const int ind)
        {
            if(ind<0 || ind>=n_*DIM_){
                py::print("error in indices X");
                exit(1);
            }
            return *(data_ + ind);
        };
        inline const double&  operator()(const int ind) const
        {
            if(ind<0 || ind>=n_*DIM_){
                py::print("error in indices X");
                exit(1);
            }
        return *(data_ + ind);
        };

        double operator[](int ind) const{ 
            if(ind<0 || ind>=n_*DIM_){
                py::print("error in indices X");
                exit(1);
            }
            return *(data_ + ind); 
        }
    #endif    
};


#endif