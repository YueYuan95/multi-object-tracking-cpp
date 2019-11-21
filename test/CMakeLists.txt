cmake_minimum_required(VERSION 3.5)
project(test_algorithm)

#debug
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0  -Wall -g -pg")

#set(CMAKE_CXX_STANDARD 11)
include_directories("./include/")
include_directories("./test")
set(LOCAL_LIBRARY main)
message(${CUDA_NVCC_FLAGS})
#CUDA_ADD_EXECUTABLE(test_algorithm main.cpp)
AUX_SOURCE_DIRECTORY(. DIR_TEST1_SRCS)
add_executable(test_algorithm main.cpp ${DIR_TEST1_SRCS})
target_link_libraries(test_algorithm  ${LOCAL_LIBRARY})