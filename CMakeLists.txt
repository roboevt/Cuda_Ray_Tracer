cmake_minimum_required(VERSION 3.21)
project(RayTracer CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)


add_executable(RayTracer main.cu)

set_target_properties(RayTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(RayTracer  GL glfw)
#target_link_libraries(RayTracer)
