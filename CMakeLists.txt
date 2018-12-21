cmake_minimum_required(VERSION 3.5)
project(green_screen_matting)
add_definitions(-std=c++11)

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)

find_package(OpenCV REQUIRED)

find_package (Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    /usr/include/eigen3
)

add_library(color_matting src/color_matting.cpp)
target_link_libraries(color_matting ${OpenCV_LIBS})

add_library(bayesian_matting src/bayesian_matting.cpp)
target_link_libraries(bayesian_matting ${OpenCV_LIBS})

add_library(trimap_generator src/trimap_generator.cpp)
target_link_libraries(trimap_generator ${OpenCV_LIBS})

add_executable(baysian_demo src/baysian_demo.cpp)
target_link_libraries(baysian_demo 
    ${OpenCV_LIBS}
    color_matting
    bayesian_matting
    trimap_generator
)

add_executable(color_demo src/color_demo.cpp)
target_link_libraries(color_demo 
    ${OpenCV_LIBS}
    color_matting
    bayesian_matting
    trimap_generator
)