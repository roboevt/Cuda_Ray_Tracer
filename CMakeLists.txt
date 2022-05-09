cmake_minimum_required(VERSION 3.8)
project(RayTracer LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)

include_directories(/usr/local/cuda-11.6/include)
add_executable(RayTracer main.cpp Tracer.cu RayTracer.cpp RayTracer.h)

set_property(TARGET RayTracer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(RayTracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(RayTracer GL glfw)