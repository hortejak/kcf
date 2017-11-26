cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")
set(OpenCV_DIR /home/nvidia/Libraries/opencv-3.2-release/share/OpenCV)
SET( CUFFT_LIB /usr/local/cuda/lib64/libcufft.so.8.0 )
SET( CUDA_LIB /usr/local/cuda-8.0/targets/aarch64-linux/lib/libcudart.so)

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

add_subdirectory(src)

# add_executable(kcf_vot main_vot.cpp vot.hpp)

# Try to find TraX header and library ...
# FIND_PACKAGE(trax REQUIRED COMPONENTS core opencv)
# IF (trax_FOUND)
#    MESSAGE(STATUS "Trax protocol found: building kcf_trax")
#    INCLUDE_DIRECTORIES(${TRAX_INCLUDE_DIRS})
#    LINK_DIRECTORIES(${TRAX_LIBRARY_DIRS})
# 
#    add_executable(kcf_trax main_trax.cpp)
#    target_link_libraries(kcf_trax ${OpenCV_LIBS} kcf ${TRAX_LIBRARIES})

find_package(CUDA REQUIRED)
# set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)
list( APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_62 -std=c++11")

cuda_add_executable( kcf_vot main_vot.cpp vot.hpp )

target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf ${CUFFT_LIB} ${CUDA_LIB})
#target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)
#ENDIF()

