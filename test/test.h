/***************************************************************************
   Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/common
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The following functions are to test functions in 
                    every chapter
****************************************************************************/

#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>

#include "mht_tree.h"
#include "mht_tracker.h"
#include "mht_graph.h"
#include "detector.h"
#include "util.h"
#include "byavs.h"
#include "km.h"
#include "NetOperator.h"
#include "extractor.h"
#include "feature_extract.h"
#include "kalman_tracker_v2.h"


//  test_mht functions
int test_flow();
int test_graph();
int test_tree();
int test_treeTograph();
int test_gating();
int test_read_txt();
int test_detector_inference();
int test_NMS();
int test_writeResult();
int test_all();
int test_mwis();
int test_km();

// test_reid functions
int test_extract();
int test_feature_extractor();


// test gpu mat operate
std::vector<cv::Mat> get_cpu_mats();
cv::Mat get_cpu_mat();
int get_gpu_mats(std::vector<bdavs::AVSGPUMat>&);
int get_gpu_mat(bdavs::AVSGPUMat&);
int get_gpu_mat(byavs::GpuMat&);
int get_bounding_boxes(std::vector<cv::Rect_<float>>&);
int test_crop_gpu_mat();

// test kalman filter
int get_kalman_test_boxes(std::vector<cv::Rect_<float>>& object_boxes);
int test_kf_initiate();
int test_kalman_batch_test();
#endif
