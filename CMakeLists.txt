cmake_minimum_required(VERSION 3.5)
                           
project(main)         
#MESSAGE(FATAL_ERROR "${CMAKE_BUILD_TYPE}")

if(CMAKE_COMPILER_IS_GNUCC)
    message("COMPILER IS GNUCC")    
    ADD_DEFINITIONS ( -std=c++11 )  
endif(CMAKE_COMPILER_IS_GNUCC)

# CUDA
set(CUDA_DIR "/usr/local/cuda")
find_package(CUDA)
SET(CUDA_NVCC_FLAGS --disable-warnings;-std=c++11;-O3;-gencode arch=compute_75,code=sm_75)
#CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED)
message(${CUDA_NVCC_FLAGS})

# opencv
link_directories("/home/yuany/opencv-346/lib")
include_directories("/home/yuany/opencv-346/include")

# 4. tensorRT
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
MESSAGE(STATUS "CMAK_SOURCE_DIR" ${CMAKE_SOURCE_DIR})

# 6. 添加源文件
FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cu  ${CMAKE_SOURCE_DIR}/src/*.cpp)


# 7. 添加链接库
LINK_LIBRARIES(opencv_ml opencv_objdetect opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_shape opencv_videoio opencv_video)
#LINK_LIBRARIES(nvcaffe_parser nvinfer nvinfer_plugin nvparsers)
#LINK_LIBRARIES(avcodec avformat swscale avutil)
#LINK_LIBRARIES(cuda nvcuvid)
#LINK_LIBRARIES(glog)

# 8.source directory，源文件目录
#AUX_SOURCE_DIRECTORY(test DIR_SRCS)

# 9. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，
#但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
#SET(ALL_SRCS ${DIR_SRCS} ${PROTO_SRCS} ${SOURCE_FILES} ${M_INCLUDE_FILES})
SET(ALL_SRCS ${PROTO_SRCS} ${SOURCE_FILES} ${M_INCLUDE_FILES})

# 10.add executable file，添加要编译的可执行文件
# ADD_EXECUTABLE(${PROJECT_NAME} ${ALL_SRCS})
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${ALL_SRCS})

# 11. 链接目标文件与库文件，添加可执行文件所需要的库，比如我们用到了libm.so（命名规则：lib+name+.so），就添加该库的名称
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${PROTOBUF_LIBRARIES})
target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudart.so)
#target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudnn.so)
