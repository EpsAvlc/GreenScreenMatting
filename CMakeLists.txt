cmake_minimum_required(VERSION 3.5)
project(green_screen_matting)
add_definitions(-std=c++11)
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

add_executable(green_screen_matting src/main.cpp)
target_link_libraries(green_screen_matting 
    ${OpenCV_LIBS}
    color_matting
    bayesian_matting
)
