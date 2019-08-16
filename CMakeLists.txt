cmake_minimum_required(VERSION 2.8.3)
project (hello CXX CUDA)

add_compile_options(-std=c++11 -g)
add_compile_options(-I /usr/include/eigen3/)

find_package(OpenCV REQUIRED)
find_package( CUDA REQUIRED )

#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_75,code=sm_75
#    )

set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

cuda_add_library(kernels_lib ${LIB_TYPE} kernels.cu)
#cuda_add_executable( kernels_lib kernels.cu reconstruct.cpp Eigen2CV.h)

add_executable(reconstruct reconstruct.cpp Eigen2CV.h)
add_dependencies(reconstruct kernels_lib)

target_include_directories(reconstruct PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries( reconstruct ${OpenCV_LIBS} kernels_lib )
