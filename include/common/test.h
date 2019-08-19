#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>

#include "mht_tree.h"
#include "mht_tracker.h"
#include "mht_graph.h"
#include "detector.h"
#include "util.h"
#include "byavs.h"

int test_flow();
int test_graph();
int test_tree();
int test_treeTograph();
int test_gating();
int test_read_txt();
int test_detector_inference();
//int test_visualize();
int test_all();

#endif
