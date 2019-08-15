#include <iostream>

#include "mht_tree.h"
#include "mht_graph.h"

int TreeToGraph(std::vector<Tree>, Graph);
int GraphToTree(Graph, std::vector<std::vector<int>>&);

/*Unit Test Funtion*/
int test_graph();
int test_tree();
