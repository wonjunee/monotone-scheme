#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "Affine2d.h"
#include "Curvature2d.h"
#include "Curvature3d.h"

namespace py = pybind11;


PYBIND11_MODULE(monotone, m) {
    // optional module docstring
    m.doc() = "C++ wrapper for monotone discretization code in Cartesian grids setting";


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
