cmake_minimum_required(VERSION 2.8.3)
project (hello CXX CUDA)

add_compile_options(-std=c++11 -g)
add_compile_options(-I /usr/include/eigen3/)

find_package(OpenCV REQUIRED)
find_package( CUDA REQUIRED )


set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

cuda_add_library(kernels_lib ${LIB_TYPE} kernels.cu)

add_executable(reconstruct reconstruct.cpp kernels.h)
add_dependencies(reconstruct kernels_lib)

target_include_directories(reconstruct PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries( reconstruct ${OpenCV_LIBS} kernels_lib )
