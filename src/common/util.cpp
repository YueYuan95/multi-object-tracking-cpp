#include "util.h"

int TreeToGraph(std::vector<Tree> tree_list, Graph graph){

    for(auto tree : tree_list){

        VexNode temp_node;
        temp_node.path.clear();
        //tmep_node.path.push_back();
    }
}

int test_graph(){
    std::vector<VexNode> vex_node_list;
    VexNode temp_node = {1.0, {1,3,4}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {1,3,2}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {1,2,1}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {1,2,3}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {2,2,1}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {2,2,3}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {2,1,0}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, {0,0,5}};
    vex_node_list.push_back(temp_node);
    
    Graph A(vex_node_list);
    A.printGraph();
    A.mwis();
}


