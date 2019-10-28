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

#include "mht_tree.h"
#include "mht_graph.h"
#include "mht_tracker.h"
#include "byavs.h"
//#include "glog/logging.h"

#define debug std::cout<<"[Time: "<<__TIMESTAMP__<<", File: "<<__FILE__<<" "<<"Line: "<<__LINE__<<" ] "
#define debugend std::endl;

int TreeToGraph(std::vector<Tree>, Graph&);
int GraphToTree(Graph, std::vector<std::vector<int>>&);
int visualize(int, cv::Mat, byavs::TrackeObjectCPUs results);//in test.cpp
int visualize(int, cv::Mat,std::vector<cv::Rect_<float>> det_result);
int visualize (int, cv::Mat, byavs::TrackeObjectCPUs, std::string);///
int visualize(int , cv::Mat , byavs::TrackeObjectCPUs , char );
void listDir(const char *name, std::vector<std::string> &fileNames, bool lastSlash);
bool VexSort(VexNode, VexNode);
bool VexSortUp(VexNode, VexNode);
double get_iou(cv::Rect_<float> , cv::Rect_<float>);
double get_ov_n1(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
double get_ov_n2(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
int writeResult(int, byavs::TrackeObjectCPUs);//in test.cpp
int writeResult(int, byavs::TrackeObjectCPUs, std::string, std::string);
#endif
