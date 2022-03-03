cmake_minimum_required(VERSION 3.20)
project(lineVisualizer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(lineVisualizer SHARED
        lineVisualizer.cpp
        lineVisualizer.h)

target_include_directories(lineVisualizer PRIVATE
        ../lineProcess
        ../lineGrouper
        ../lineFinder)

#target_compile_options(lineVisualizer PRIVATE -O3)

target_link_libraries(lineVisualizer
        ${OpenCV_LIBS}
        pthread
        lineFinder
        lineGrouper)