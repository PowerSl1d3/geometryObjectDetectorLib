cmake_minimum_required(VERSION 3.20)
project(lineGrouper)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(lineGrouper SHARED
        lineGrouper.cpp
        lineGrouper.h)

#target_compile_options(lineGrouper PRIVATE -O3)

target_include_directories(lineGrouper PRIVATE ../lineProcess)

target_link_libraries(lineGrouper
        lineProcess
        ${OpenCV_LIBS}
        pthread)