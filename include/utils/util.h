/***************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/common
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The following functions are tools, in every chapter
                   *preorderTraversal:Pretraversal
                   *backTraversal:Post traversal
                   *TreeToGraph:Transfers a tree to a graph
                   *visualize:Visualizes the tracking results
                   *listDir:Opens a dir and gets information
                   *VexSort:compare the vex score between A and B 
                            and return the bigger
                   *VexSortUp:compare the vex score between A and B and 
                    return the smaller
                   *get_ov_n1:Calculate the overlap base on rect1
                   *get_ov_n2:Calculate the overlap base on rect2
                   *writeResult:Write result in files
****************************************************************************/
#ifndef __UTIL_H_
#define __UTIL_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>

#include "mht_tree.h"
#include "mht_graph.h"
#include "mht_tracker.h"
#include "byavs.h"
#include "NetOperator.h"
#include "extractor.h"
//#include "glog/logging.h"

#define debug std::cout<<"[Time: "<<__TIMESTAMP__<<", File: "<<__FILE__<<" "<<"Line: "<<__LINE__<<" ] "
#define debugend std::endl;

int TreeToGraph(std::vector<Tree>, Graph&);
int GraphToTree(Graph, std::vector<std::vector<int>>&);
int visualize(int, cv::Mat, byavs::TrackeObjectCPUs results);//in test.cpp
int visualize(int, cv::Mat,std::vector<cv::Rect_<float>> det_result);
int visualize (int, cv::Mat, byavs::TrackeObjectGPUs, std::string);///
int visualize(int , cv::Mat , byavs::TrackeObjectCPUs , char );
void listDir(const char *name, std::vector<std::string> &fileNames, bool lastSlash);
bool VexSort(VexNode, VexNode);
bool VexSortUp(VexNode, VexNode);
double get_iou(cv::Rect_<float> , cv::Rect_<float>);
double get_ov_n1(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
double get_ov_n2(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
int writeResult(int, byavs::TrackeObjectCPUs);//in test.cpp
int writeResult(int, byavs::TrackeObjectCPUs, std::string, std::string);
int writeResult(int, byavs::TrackeObjectGPUs, std::string, std::string);

//
int convert_to_tracking_input(std::string img_pth, std::vector<cv::Rect_<float>>, std::vector<float>, byavs::TrackeInputGPU&);

//
double get_feature_distance_cosine(std::vector<float>, std::vector<float>);
double get_feature_distance_euclidean(std::vector<float>, std::vector<float>);
int get_feature_cost_matrix(std::vector<std::vector<double>>&, std::vector<std::vector<float>>, std::vector<std::vector<float>>);
double get_pow_sum(std::vector<float>);
double get_vector_time(std::vector<float>, std::vector<float>);

//GPU Image Operate
bool convert_image_from_cpu_to_gpu(std::vector<byavs::GpuMat>& gpu_mats, std::vector<cv::Mat> img_mats);
bool convert_image_from_cpu_to_gpu(std::vector<bdavs::AVSGPUMat>& gpu_mats, std::vector<cv::Mat> img_mats);
bool convert_image_from_cpu_to_gpu(bdavs::AVSGPUMat & gpu_mat, cv::Mat img_mat);
bool convert_image_from_cpu_to_gpu(byavs::GpuMat & gpu_mat, cv::Mat img_mat);

int crop_avs_mat(byavs::GpuMat, cv::Rect_<float>, bdavs::AVSGPUMat&);
int crop_gpu_mat(byavs::GpuMat, std::vector<cv::Rect_<float>>, std::vector<bdavs::AVSGPUMat>&);

int crop_avs_mat(bdavs::AVSGPUMat, cv::Rect_<float>, bdavs::AVSGPUMat&);
int crop_gpu_mat(bdavs::AVSGPUMat, std::vector<cv::Rect_<float>>, std::vector<bdavs::AVSGPUMat>&);

int release_avs_gpu_mat(std::vector<bdavs::AVSGPUMat> &);
int normalization(std::vector<std::vector<double>>&);
std::vector<float> norm(std::vector<float>);

#endif
