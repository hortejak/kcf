cmake_minimum_required(VERSION 3.0.2)

project(kcf_tracker)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

SET(FFT "OpenCV" CACHE STRING "Select FFT implementation")
SET_PROPERTY(CACHE FFT PROPERTY STRINGS OpenCV OpenCV_cuFFT fftw cuFFTW)
MESSAGE(STATUS "FFT implementation: ${FFT}")

option(OPENMP "Use OpenMP library. Works with FFTW and OpenCV implementation." OFF)
option(ASYNC "Works only if OPENCV_CUFFT is not ON. Will enable C++ async directive." OFF)
option(DEBUG_MODE "Additional terminal outputs and screens. " OFF)
option(PROFILING "Enable profiling using perf_event_open together with libpfm4. " OFF)

IF(DEBUG_MODE )
  add_definitions(-DDEBUG_MODE )
  MESSAGE(STATUS "Debug mode")
ENDIF() #DEBUG_MODE 

IF(PROFILING)
  add_definitions(-DPROFILING )
  MESSAGE(STATUS "Profiling mode")
ENDIF() #DEBUG_MODE 

SET(use_cuda OFF)

IF(FFT STREQUAL "OpenCV")
ELSEIF(FFT STREQUAL "OpenCV_cuFFT")
  add_definitions(-DOPENCV_CUFFT)
  set(use_cuda ON)
ELSEIF(FFT STREQUAL "fftw")
  add_definitions(-DFFTW)
ELSEIF(FFT STREQUAL "cuFFTW")
  add_definitions(-DFFTW -DCUFFTW)
  set(use_cuda ON)
ELSE()
  MESSAGE(FATAL_ERROR "Invalid FFT implementation selected")
ENDIF()

IF(ASYNC AND NOT OPENMP)
  add_definitions(-DASYNC)
  MESSAGE(STATUS "ASYNC")
ENDIF() #ASYNC

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR} ")

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  IF(NOT OPENMP)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wno-unknown-pragmas")
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

IF(NOT use_cuda)
  add_executable(kcf_vot main_vot.cpp vot.hpp)
ELSE()
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
  target_link_libraries(kcf_vot ${CUDA_LIBRARIES})
ENDIF() #OPENCV_CUFFT

target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)

IF(FFT STREQUAL "fftw")
  target_link_libraries(kcf_vot fftw3f)
  IF(OPENMP)
    target_link_libraries(kcf_vot fftw3_omp)
  ENDIF()
ENDIF() #FFTW

IF(FFT STREQUAL "cuFFTW")
  target_link_libraries(kcf_vot cufft cufftw)
ENDIF() #cuFFTW

IF(PROFILING)
  include(ExternalProject)
  find_program(MAKE_EXE NAMES gmake nmake make)
  ExternalProject_Add(project_libpfm4
    URL https://sourceforge.net/projects/perfmon2/files/libpfm4/libpfm-4.9.0.tar.gz/download
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND ${MAKE_EXE}
    INSTALL_COMMAND ${MAKE_EXE} PREFIX=${CMAKE_CURRENT_BINARY_DIR}/external/libpfm4-4.9.0 install
  )

  add_library(libpfm4 STATIC IMPORTED )
  set_property(TARGET libpfm4 PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/libpfm4-4.9.0/lib/libpfm.a)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/libpfm4-4.9.0/include/perfmon)
  target_link_libraries(kcf_vot libpfm4)
ENDIF()
