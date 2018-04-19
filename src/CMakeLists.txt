cmake_minimum_required(VERSION 2.8)

set(KCF_LIB_SRC kcf.cpp kcf.h fft.cpp)

SET(FFT "OpenCV" CACHE STRING "Select FFT implementation")
SET_PROPERTY(CACHE FFT PROPERTY STRINGS OpenCV OpenCV_cuFFT fftw cuFFTW cuFFT)
MESSAGE(STATUS "FFT implementation: ${FFT}")

option(OPENMP "Use OpenMP library. Works with FFTW and OpenCV implementation." OFF)
option(ASYNC "Works only if OPENCV_CUFFT is not ON. Will enable C++ async directive." OFF)
option(CUDA_DEBUG "Enables error cheking for cuda and cufft. " OFF)
option(BIG_BATCH "Enable transforming all features from all scales together." OFF)

IF(PROFILING)
  add_definitions(-DPROFILING )
  MESSAGE(STATUS "Profiling mode")
ENDIF()

IF(BIG_BATCH)
  add_definitions(-DBIG_BATCH )
  MESSAGE(STATUS "Big_batch mode")
ENDIF()

SET(use_cuda OFF)

IF(FFT STREQUAL "OpenCV")
  list(APPEND KCF_LIB_SRC fft_opencv.cpp complexmat.hpp)
ELSEIF(FFT STREQUAL "OpenCV_cuFFT")
  list(APPEND KCF_LIB_SRC fft_opencv_cuda.cpp complexmat.hpp)
  add_definitions(-DOPENCV_CUFFT)
  set(use_cuda ON)
ELSEIF(FFT STREQUAL "fftw")
  list(APPEND KCF_LIB_SRC fft_fftw.cpp complexmat.hpp)
  add_definitions(-DFFTW)
ELSEIF(FFT STREQUAL "cuFFTW")
  list(APPEND KCF_LIB_SRC fft_fftw.cpp complexmat.hpp)
  add_definitions(-DFFTW -DCUFFTW)
  set(use_cuda ON)
ELSEIF(FFT STREQUAL "cuFFT")
    list(APPEND KCF_LIB_SRC fft_cufft.cpp complexmat.cuh)
    add_definitions(-DCUFFT)
    set(use_cuda ON)
    iF(CUDA_DEBUG)
      add_definitions(-DCUDA_DEBUG)
      MESSAGE(STATUS "CUDA debug mode")
    ENDIF()
ELSE()
  MESSAGE(FATAL_ERROR "Invalid FFT implementation selected")
ENDIF()

IF(ASYNC AND NOT OPENMP)
  add_definitions(-DASYNC)
  MESSAGE(STATUS "ASYNC")
ENDIF() #ASYNC

FIND_PACKAGE( OpenCV REQUIRED )

IF(use_cuda)
  find_package(CUDA REQUIRED)

  if (NOT ${OpenCV_USE_CUFFT})
    message(SEND_ERROR "OpenCV was not built with cuFFT")
  endif()

  include_directories ( ${CUDA_INCLUDE_DIRS} )
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_HOST_COMPILER /usr/bin/g++)
  list( APPEND CUDA_NVCC_FLAGS -O3 --gpu-architecture sm_62 -std=c++11 -default-stream per-thread)
  find_cuda_helper_libs(cufftw)
  IF(FFT STREQUAL "cuFFT")
  add_subdirectory(cuda)
  cuda_add_library(complexmat complexmat.cu)
  ENDIF()
  
ENDIF()

add_subdirectory(piotr_fhog)
add_subdirectory(cn)

add_library(kcf STATIC ${KCF_LIB_SRC})
target_link_libraries(kcf fhog cndata  ${OpenCV_LIBS})
set_target_properties(kcf PROPERTIES VERSION 1.0.0 SOVERSION 1)

IF(FFT STREQUAL "fftw")
  target_link_libraries(kcf fftw3f)
  IF(OPENMP)
    target_link_libraries(kcf fftw3_omp)
  ELSEIF(ASYNC)
    target_link_libraries(kcf fftw3_threads)
  ENDIF()
ENDIF() #FFTW

IF(FFT STREQUAL "cuFFTW")
  target_link_libraries(kcf ${CUDA_cufft_LIBRARY} ${CUDA_cufftw_LIBRARY})
ENDIF() #cuFFTW

IF(FFT STREQUAL "cuFFT")
    target_link_libraries(kcf ${CUDA_cufft_LIBRARY} ${CUDA_LIBRARIES} complexmat)
ENDIF()

IF(PROFILING)
  target_link_libraries(kcf pfm)
ENDIF()
