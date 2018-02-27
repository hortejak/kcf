cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

option(OPENCV_CUFFT " Use Nvidia CUFFT implemented in OpenCV. Together with Hostmem from OpenCV." OFF)
option(FFTW "Use FFTW implementation of FFT. If selected together with OPENCV_CUFFT then this option will not be used." OFF)
option(OPENMP "Use OpenMP library. Works with FFTW and OpenCV implementation." OFF)
option(ASYNC "Works only if OPENCV_CUFFT is not ON. Will enable C++ async directive." OFF)
option(DEBUG_MODE "Additional terminal outputs and screens. " OFF)

IF((FFTW AND OPENCV_CUFFT) OR (OPENCV_CUFFT AND ASYNC) OR (ASYNC AND FFTW) OR (ASYNC AND OPENMP))
  MESSAGE(FATAL_ERROR "You can't do that")
ENDIF() #WRONG_COMBINATION

IF(DEBUG_MODE )
  add_definitions(-DDEBUG_MODE )
  MESSAGE(STATUS "Debug mode")
ENDIF() #DEBUG_MODE 

IF(OPENCV_CUFFT)
  add_definitions(-DOPENCV_CUFFT)
  MESSAGE(STATUS "OPENCV+CUFFT version")
ENDIF() #OPENCV_CUFFT

IF(FFTW)
  add_definitions(-DFFTW)
  MESSAGE(STATUS "FFTW")
ENDIF() #FFTW

IF(ASYNC)
  add_definitions(-DASYNC)
  MESSAGE(STATUS "ASYNC")
ENDIF() #ASYNC

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR} ")

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  IF(NOT OPENMP)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  ELSE()
    MESSAGE(STATUS "OpenMP")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ENDIF() #ASYNC
ENDIF ()

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/piotr_fhog ${CMAKE_SOURCE_DIR}/src/cn)

include_directories(${CMAKE_SOURCE_DIR}/sse2neon)

add_subdirectory(src)

IF(OPENCV_CUFFT)
  find_package(CUDA REQUIRED)
  link_directories ( ${CUDA_LIB_DIR} )
  MESSAGE(STATUS "CUDA_LIB_DIR: ${CUDA_LIB_DIR} ")

  include_directories ( ${CUDA_INCLUDE_DIRS} )
  MESSAGE(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER /usr/bin/g++)
  list( APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_62 -std=c++11")

  cuda_add_executable( kcf_vot main_vot.cpp vot.hpp )

  target_link_libraries(kcf_vot ${CUDA_LIBRARIES} ${OpenCV_LIBS}  kcf)
ELSE()
  add_executable(kcf_vot main_vot.cpp vot.hpp)
  target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)
  IF(FFTW)
    target_link_libraries(kcf_vot fftw3f)
    IF(FFTW AND OPENMP)
      target_link_libraries(kcf_vot fftw3_omp)
    ENDIF() #FFTW AND OPENMP
  ENDIF() #FFTW
ENDIF() #OPENCV_CUFFT
