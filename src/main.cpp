#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "mht_tree.h"
#include "mht_graph.h"

int test_graph(){
    std::vector<VexNode> vexnode_list;
    VexNode temp_node = {1.0, {1,3,4}};
}

/*std::vector<treeNode> test_tree(){
    std::vector<treeNode> root_list;
    treeNode root = {{10,9,8,7},6,2,1};
    root_list.push_back(root);
    return root_list;
}*/

int main(){
   //test_graph();
    
   
    treeNode root = {{10,9,8,7},6,2,1,NULL};
    std::cout<<root.box<<std::endl;

    std::shared_ptr<treeNode> root1(new treeNode(root));
    std::vector< std::shared_ptr<treeNode>> testTree;
    testTree.push_back(root1);
    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict;
    //map<int, vector<std::shared_ptr<treeNode>>>:: iterator it;
    //dict.insert(pair<int, std::vector<std::shared_ptr<treeNode>>> 1,testTree);
    dict[1] = testTree;

    Tree test_tree(root1,1,3);
    //test_tree(root1,1,3);
    //printf("")
    std::cout<<"id:"<<test_tree.getId()<<std::endl;
    std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    std::cout<<"leaf_node:"<<&(test_tree.getLeafNode()[0])<<std::endl;
    std::cout<<"head_node:"<<test_tree.getHead()<<std::endl;
    std::cout<<"box:"<<test_tree.getRoot()->box<<std::endl;
    
    test_tree.addNode(dict);
}
