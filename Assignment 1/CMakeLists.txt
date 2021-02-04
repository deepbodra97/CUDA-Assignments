cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(Assignment1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# default CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

# default build type is Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_executable(transpose transpose.cu)
add_executable(deviceQuery deviceQuery.cu)

message(STATUS "build type:  ${CMAKE_BUILD_TYPE}")
