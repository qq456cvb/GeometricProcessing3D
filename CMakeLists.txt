cmake_minimum_required(VERSION 3.14)
project(gp3d)
set(CMAKE_CXX_STANDARD 17)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRC_FILES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
)

file(GLOB EXAMPLES
    ${PROJECT_SOURCE_DIR}/examples/*.cpp
)

file(GLOB_RECURSE PYBIND_SRC_FILES
    ${PROJECT_SOURCE_DIR}/pybind/*.cpp
)

find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )
find_package( pybind11 REQUIRED )
pybind11_add_module(pygeom ${PYBIND_SRC_FILES})
target_include_directories(pygeom PUBLIC ${PYTHON_INCLUDE_DIRS})

FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;-arch=compute_75;-lineinfo;")

cuda_add_library(gp3d SHARED ${SRC_FILES})
target_link_libraries(gp3d ${ARMADILLO_LIBRARIES})
add_executable(main ${EXAMPLES})

target_link_libraries(main gp3d)
target_link_libraries(pygeom PRIVATE gp3d)