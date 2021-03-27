#ifndef EIGEN_HPP
#define EIGEN_HPP

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Eigen/Dense>

template <typename T, typename = int>
struct IsEigenVector : std::false_type {};

template <typename T>
struct IsEigenVector<T, decltype(std::declval<typename T::value_type>(), 0)> : std::true_type {};

template<typename T>
std::vector<T> bind_array2cpp(pybind11::array_t<typename T::value_type, pybind11::array::c_style | pybind11::array::forcecast> array) {
    static_assert(IsEigenVector<T>::value, "Not a Eigen vector!");
    using VT = typename T::value_type;
    if (array.ndim() != 2 || array.shape(1) != T::SizeAtCompileTime) {
        throw pybind11::cast_error();
    }
    std::vector<T> eigen_vecs(array.shape(0));
    auto array_unchecked = array.mutable_unchecked();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vecs[i] = Eigen::Map<T>(&array_unchecked(i, 0));
    }
    return eigen_vecs;
}


template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
pybind11::class_<Vector, holder_type> bind_eigen_vector(pybind11::handle scope, std::string const &name, Args&&... args) {
    using Class_ = pybind11::class_<Vector, holder_type>;

    // If the value_type is unregistered (e.g. a converting type) or is itself registered
    // module-local then make the vector binding module-local as well:
    using vtype = typename Vector::value_type;
    auto vtype_info = pybind11::detail::get_type_info(typeid(vtype));
    bool local = !vtype_info || vtype_info->module_local;

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

    cl.def(pybind11::init<>());
    pybind11::detail::vector_if_copy_constructible<Vector, Class_>(cl);
    // pybind11::detail::vector_modifiers<Vector, Class_>(cl);
    pybind11::detail::vector_accessor<Vector, Class_>(cl);

    cl.def("__bool__",
        [](const Vector &v) -> bool {
            return !v.empty();
        },
        "Check whether the list is nonempty"
    );

    cl.def("__len__", &Vector::size);

    cl.def(pybind11::init(std::function(bind_array2cpp<vtype>)));

    cl.def_buffer([](std::vector<vtype> &v) -> pybind11::buffer_info {
        return pybind11::buffer_info(v[0].data(), sizeof(typename vtype::value_type),
                               pybind11::format_descriptor<typename vtype::value_type>::format(), 2,
                               {v.size(), std::size_t(vtype::SizeAtCompileTime)},
                               {sizeof(vtype), sizeof(typename vtype::value_type)});
    });
    return cl;
}


void pybind_eigen(pybind11::module &m) {

    pybind11::bind_vector<std::vector<int32_t>>(m, "VectorXi", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<float>>(m, "VectorXf", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<double>>(m, "VectorXd", pybind11::buffer_protocol());

    bind_eigen_vector<std::vector<Eigen::Vector2i>>(m, "VectorXYi", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector2f>>(m, "VectorXYf", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector2d>>(m, "VectorXYd", pybind11::buffer_protocol());

    bind_eigen_vector<std::vector<Eigen::Vector3i>>(m, "VectorXYZi", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector3f>>(m, "VectorXYZf", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector3d>>(m, "VectorXYZd", pybind11::buffer_protocol());
    
    bind_eigen_vector<std::vector<Eigen::Vector4i>>(m, "VectorXYZWi", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector4f>>(m, "VectorXYZWf", pybind11::buffer_protocol());
    bind_eigen_vector<std::vector<Eigen::Vector4d>>(m, "VectorXYZWd", pybind11::buffer_protocol());
}

#endif