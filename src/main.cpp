#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "affine3d.h"

namespace py = pybind11;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
PYBIND11_MODULE(monotone, m) {
    // optional module docstring
    m.doc() = "C++ wrapper for monotone discretization code in 3D Cartesian grids setting";

    py::class_<Affine3DSolver>(m, "Affine3DSolver") 
        .def(py::init<py::array_t<double> &, py::array_t<int> &, int>())
        .def("perform_one_iteration", &Affine3DSolver::perform_one_iteration);
}
