#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "Eikonal2d.h"
#include "Tukey2d.h"
#include "Affine2d.h"
#include "Curvature2d.h"
#include "Curvature3d.h"

namespace py = pybind11;

void interpolate(py::array_t<double, py::array::c_style | py::array::forcecast> u_np, py::array_t<double, py::array::c_style | py::array::forcecast> u_small_np){
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
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, int>())
        .def("perform_one_iteration", &Eikonal2DSolver::perform_one_iteration);

    py::class_<Tukey2DSolver>(m, "Tukey2DSolver") 
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, int>())
        .def("perform_one_iteration", &Tukey2DSolver::perform_one_iteration)
        .def("perform_one_iteration_with_bdry", &Tukey2DSolver::perform_one_iteration_with_bdry);

    py::class_<Affine2DSolver>(m, "Affine2DSolver") 
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, int>())
        .def("perform_one_iteration", &Affine2DSolver::perform_one_iteration);

    py::class_<Curv2DSolver>(m, "Curv2DSolver") 
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, int>())
        .def("perform_one_iteration", &Curv2DSolver::perform_one_iteration);

    py::class_<Curv3DSolver>(m, "Curv3DSolver") 
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>, py::array_t<int, py::array::c_style | py::array::forcecast>, int>())
        .def("perform_one_iteration", &Curv3DSolver::perform_one_iteration);
} 