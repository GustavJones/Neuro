cmake_minimum_required(VERSION 3.15)
project(Neuro CXX)

file(GLOB SOURCES "*.cpp")
add_library(Neuro ${SOURCES})
target_include_directories(Neuro PRIVATE ../../include/)
