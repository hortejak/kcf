cmake_minimum_required(VERSION 3.0.2)

project(kcf_tracker)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

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
  target_link_libraries(kcf_vot ${CUDA_cufft_LIBRARY} ${CUDA_cufftw_LIBRARY})
ENDIF() #cuFFTW

IF(PROFILING)
  target_link_libraries(kcf_vot pfm)
ENDIF()
