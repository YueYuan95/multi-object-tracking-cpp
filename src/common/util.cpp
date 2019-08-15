#include "util.h"
#include "mht_tree.h"


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

int test_tree(){

    treeNode root = {{10,9,8,7},6,2,1,NULL};
    std::shared_ptr<treeNode> root_ptr(new treeNode(root));

    treeNode root_a = {{10,9,8,7},6,2,1,NULL};
    treeNode root_b = {{10,9,8,7},6,2,2,NULL};
    treeNode root_c = {{10,9,8,7},6,2,3,NULL};
    treeNode root_d = {{10,9,8,7},6,2,4,NULL};

    std::shared_ptr<treeNode> root_a_ptr(new treeNode(root_a));
    std::shared_ptr<treeNode> root_b_ptr(new treeNode(root_b));
    std::shared_ptr<treeNode> root_c_ptr(new treeNode(root_c));
    std::shared_ptr<treeNode> root_d_ptr(new treeNode(root_d));

    std::vector<std::shared_ptr<treeNode>> node_list;

    node_list.push_back(root_a_ptr);
    node_list.push_back(root_b_ptr);
    node_list.push_back(root_c_ptr);
    node_list.push_back(root_d_ptr);

    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict;
    dict[1] = node_list;
    //map<int, vector<std::shared_ptr<treeNode>>>:: iterator it;
    //dict.insert(pair<int, std::vector<std::shared_ptr<treeNode>>> 1,testTree);
    
    Tree test_tree(root_ptr,1,3);
    //test_tree(root1,1,3);
    //printf("")
    // std::cout<<"id:"<<test_tree.getId()<<std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    // std::cout<<"leaf_node:"<<&(test_tree.getLeafNode()[0])<<std::endl;
    // std::cout<<"head_node:"<<test_tree.getHead()<<std::endl;
    // std::cout<<"box:"<<test_tree.getRoot()->box<<std::endl;
    
    
    test_tree.addNode(dict);
    test_tree.printTree(root_ptr);

}
