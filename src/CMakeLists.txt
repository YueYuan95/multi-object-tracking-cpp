cmake_minimum_required(VERSION 3.5)
project(mot)

#debug
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0  -Wall -g -pg")

#set(CMAKE_CXX_STANDARD 11)
include_directories("./include/")
set(LOCAL_LIBRARY main)
message(${CUDA_NVCC_FLAGS})
#CUDA_ADD_EXECUTABLE(mot main.cpp)
add_executable(mot main.cpp)
target_link_libraries(mot  ${LOCAL_LIBRARY})