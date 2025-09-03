cmake_minimum_required(VERSION 3.15)
project(Neuro-Test CXX)

file(GLOB SOURCES "*.cpp")
add_executable(Neuro-Test ${SOURCES})
target_include_directories(Neuro-Test PRIVATE ../../include/)
target_link_libraries(Neuro-Test PRIVATE Neuro)
