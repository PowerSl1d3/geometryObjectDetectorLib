cmake_minimum_required(VERSION 3.20)
project(geometryObjectDetectorLib)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lineFinder)
add_subdirectory(lineGrouper)
add_subdirectory(lineProcess)
add_subdirectory(lineVisualizer)

add_executable(lineProc main.cpp)

#target_compile_options(lineProc PRIVATE -O3)
target_include_directories(lineProc PRIVATE
        lineFinder
        lineGrouper
        lineProcess
        lineVisualizer)

target_link_libraries(lineProc
        ${OpenCV_LIBS}
        lineFinder
        lineGrouper
        lineProcess
        lineVisualizer)