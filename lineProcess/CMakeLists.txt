cmake_minimum_required(VERSION 3.20)
project(lineProcess)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

add_library(lineProcess SHARED
        lineProcess.cpp
        lineProcess.h)

#target_compile_options(lineProcess PRIVATE -O3)

target_link_libraries(lineProcess
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        pthread)