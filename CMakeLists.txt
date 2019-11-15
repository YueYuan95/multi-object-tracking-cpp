
cmake_minimum_required(VERSION 3.5)

project(main)
#MESSAGE(FATAL_ERROR "${CMAKE_BUILD_TYPE}")

if(CMAKE_COMPILER_IS_GNUCC)
    message("COMPILER IS GNUCC")
    ADD_DEFINITIONS ( -std=c++11 )
endif(CMAKE_COMPILER_IS_GNUCC)

set(CUDA_DIR "/usr/local/cuda")
find_package(CUDA)

SET(CUDA_NVCC_FLAGS --disable-warnings;-std=c++11;-O3;-gencode arch=compute_75,code=sm_75)

message(${CUDA_NVCC_FLAGS})

#debug
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0  -Wall -g -pg")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0  -Wall -g")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#opencv
#set(OpenCV_DIR "/data/wuh/open_source/opencv-3.4.6/build")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories("/data/wuh/open_source/opencv-3.4.6/install/lib")
#include_directories("/data/wuh/open_source/opencv-3.4.6/install/include")
link_directories("/usr/local/lib/")
include_directories("/usr/local/include/opencv4")

#glog
LINK_LIBRARIES(glog)

#message(${SOURCE_FILES})

#tensorRT
include_directories(/usr/local/TensorRT-5.1.2.2/include)
link_directories(/usr/local/TensorRT-5.1.2.2/lib)

# 5. 头文件
# 5.1. 定义函数，用于递归添加头文件
function(include_sub_directories_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})               # 当前路径是一个目录吗，是的话就加入到包含目录
        message("include dir: " ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*) # 获得当前目录下的所有文件，让如ALL_SUB列表中
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            include_sub_directories_recursively(${root_dir}/${sub}) # 对子目录递归调用，包含
        endif()
    endforeach()
endfunction()

# 5.2. 添加头文件
include_sub_directories_recursively(${CMAKE_SOURCE_DIR}/include) # 对子目录递归调用，包含
include_directories(${CMAKE_SOURCE_DIR}/test)
MESSAGE(STATUS "CMAK_SOURCE_DIR" ${CMAKE_SOURCE_DIR})

# 6. 添加源文件
FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cu  ${CMAKE_SOURCE_DIR}/src/*.cpp)
#FILE(GLOB_RECURSE SOURCE_FILES )

LINK_LIBRARIES(opencv_ml opencv_objdetect opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_shape opencv_videoio opencv_video)
LINK_LIBRARIES(nvcaffe_parser nvinfer nvinfer_plugin nvparsers)


CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES})
#CUDA_ADD_EXECUTABLE(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudart.so)
target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudnn.so)
#target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/3rdParty/libbyavs_api.so)
#target_link_libraries(/home/panhm/MyPorjects/algorithm_module/3rdParty/libbyavs_api.so)
target_link_libraries(${PROJECT_NAME} /usr/local/lib/libopencv_dnn.so.4.1)
add_subdirectory(test)
add_subdirectory(src)