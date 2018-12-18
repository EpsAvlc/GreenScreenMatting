cmake_minimum_required(VERSION 2.6)
project(green_screen_matting)

add_executable(green_screen_matting main.cpp)

install(TARGETS green_screen_matting RUNTIME DESTINATION bin)
