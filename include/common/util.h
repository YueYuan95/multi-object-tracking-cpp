#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "mht_tree.h"
#include "mht_graph.h"
#include "mht_tracker.h"
#include "byavs.h"


int TreeToGraph(std::vector<Tree>, Graph&);
int GraphToTree(Graph, std::vector<std::vector<int>>&);
int visualize(bool visual, byavs::TrackeObjectCPUs results);
void listDir(const char *name, std::vector<std::string> &fileNames, bool lastSlash);