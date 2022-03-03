cmake_minimum_required(VERSION 3.20)
project(lineFinder)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(lineFinder SHARED
        lineFinder.cpp
        lineFinder.h)

#target_compile_options(lineFinder PRIVATE -O3)

target_link_libraries(lineFinder
        ${OpenCV_LIBS}
        pthread)