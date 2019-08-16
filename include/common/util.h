#include <iostream>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <math.h>

#include "mht_tree.h"
#include "mht_graph.h"
#include "mht_tracker.h"

int TreeToGraph(std::vector<Tree>, Graph&);
int GraphToTree(Graph, std::vector<std::vector<int>>&);

/*Unit Test Funtion*/
int test_graph();
int test_tree();
int test_treeTograph();
int test_gating();