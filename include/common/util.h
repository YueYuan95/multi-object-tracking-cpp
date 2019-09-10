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


int TreeToGraph(std::vector<Tree>, Graph&);
int GraphToTree(Graph, std::vector<std::vector<int>>&);
int visualize(int, cv::Mat, byavs::TrackeObjectCPUs results);
int visualize(int, cv::Mat,std::vector<cv::Rect_<float>> det_result);
int visualize(int , cv::Mat , byavs::TrackeObjectCPUs , char );
void listDir(const char *name, std::vector<std::string> &fileNames, bool lastSlash);
bool VexSort(VexNode, VexNode);
bool VexSortUp(VexNode, VexNode);
double get_iou(cv::Rect_<float> , cv::Rect_<float>);
double get_ov_n1(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
double get_ov_n2(cv::Rect_<float> rec1, cv::Rect_<float> rec2);
int writeResult(int, byavs::TrackeObjectCPUs);
#endif
