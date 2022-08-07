cmake_minimum_required(VERSION 3.10)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(conv2d_relu_pool2d)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -fexceptions -fasynchronous-unwind-tables -funwind-tables")

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
  message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
  message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
  message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
else()
  message(FATAL_ERROR "Could not found OpenMP!")
  return()
endif()

add_executable(conv2d_relu_pool2d conv2d_relu_pool2d.cc)
target_link_libraries(conv2d_relu_pool2d dl)
