cmake_minimum_required(VERSION 2.8.9)
project (hello)

add_compile_options(-std=c++11 -g)
add_compile_options(-I /usr/include/eigen3/)


find_package( OpenCV REQUIRED )
add_executable(reconstruct reconstruct.cpp Eigen2CV.h)

target_include_directories(reconstruct PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries( reconstruct ${OpenCV_LIBS} )
