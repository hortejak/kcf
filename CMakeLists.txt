cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

option(OPENCV_CUFFT "If OFF CPU implementation using OpenCV implementation of fft will be used. If ON Nvidia CUFFT implemented in OpenCV will be used. Together with Hostmem from OpenCV." OFF)
option(VISULIZE_RESULT "Check if you want to visulize the result." OFF)
option(ASYNC "Default setting is on. Works only if OPENCV_CUFFT is not ON. Will enable C++ async directive." ON)
if(VISULIZE_RESULT)
  add_definitions(-DVISULIZE_RESULT)
  MESSAGE(STATUS "Visulize result")
endif()
if(OPENCV_CUFFT)
  add_definitions(-DOPENCV_CUFFT)
  MESSAGE(STATUS "OPENCV+CUFFT version")
elseif(ASYNC)
  add_definitions(-DASYNC)
  MESSAGE(STATUS "CPU+ASYNC version")
else()
  MESSAGE(STATUS "CPU+SERIAL version")
endif() #OPENCV_CUFFT

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR} ")

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF ()

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/piotr_fhog ${CMAKE_SOURCE_DIR}/src/cn)

include_directories(${CMAKE_SOURCE_DIR}/sse2neon)

add_subdirectory(src)

if(OPENCV_CUFFT)
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
else()
add_executable(kcf_vot main_vot.cpp vot.hpp)
target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)
endif() #OPENCV_CUFFT
