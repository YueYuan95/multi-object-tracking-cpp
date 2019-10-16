/**************************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/include
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The struct, treenode, is a node of a tree
                   *The class, Tree, is to form tracking tree family
****************************************************************************/

#include "mht_tree.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <vector>
#include <deque>

int Tree::tree_id = 1;

/*
Forms tracking tree family
Input: tree root node, label and N
*/
Tree::Tree (std::shared_ptr<treeNode> root, int i, int n) {  


    id = tree_id;
    tree_id++;
    label = i;
    root_node = root;
    leaf_node.push_back(root);
    head_node = root;
    N = n;
    miss_times = 0;
    //hit_times = 0;  //DEPRECATED
}

/*
Adds leaf to a tree
The inputs are the node's index(label) and the node itself
*/
int Tree::addNode (int node_index, std::shared_ptr<treeNode> tree_Node) {


    tree_Node->parent = leaf_node[node_index];
    tree_Node->children.clear();
    leaf_node[node_index]->children.push_back(tree_Node);   
}

/*  
Updates the current leaf_node 
*/
int Tree::changeLeaf () { 

    if(leaf_node.size() == 0) return 1;
    if(leaf_node.size() == 1 && leaf_node[0]->children.size() == 0) return 1;
    std::vector<std::shared_ptr<treeNode>> candidate;
    candidate.clear();
    for (int i=0; i < leaf_node.size(); i++) {
        for (int j=0; j < leaf_node[i]->children.size();j++) {
            candidate.push_back(leaf_node[i]->children[j]);
        }
    }
    leaf_node = candidate;    
}

/*
search the indexes' nodes in leaf_node
then add the nodes' children
the inputs is the dictionary of a route
*/
int Tree::addNode (std::map<int, std::vector<std::shared_ptr<treeNode>>> dict) {


    std::map<int, std::vector<std::shared_ptr<treeNode>>>::iterator it;
    it = dict.begin();
    std::vector<std::shared_ptr<treeNode>> sub_children_node;
    while (it != dict.end()) {
        leaf_node[it->first]->children = it->second;
        for (int i = 0; i<it->second.size(); i++) {
            it->second[i]->parent = leaf_node[it->first]; 
        }
        it++;
    }
    sub_children_node.clear();
    for (int i = 0; i < leaf_node.size(); i++) {
        for (int j=0; j< leaf_node[i]->children.size(); j++) {
            sub_children_node.push_back(leaf_node[i]->children[j]);
        }
    }
    leaf_node = sub_children_node; 
}


/*
pruns the branches of a headnode, which is the first index of a route
the input is the route
*/   
int Tree::pruning(std::vector<int> route) {

    if ((leaf_node[0]->level - head_node->level) < (N-1)) {       
        return 1;
    }
    for (int i = 0; i < head_node->children.size(); i++) {
        if (head_node->children[i]->index != route[1]) {
            head_node->children.erase(head_node->children.begin()+i);
            i--;
        } 
    }

    if (head_node->children.size() == 0) {
        return 1;
    }
    if (head_node->children.size() == 1) {
        head_node = head_node->children[0];
        generateLeafNode();
        return 1;
    } 
    if (head_node->children.size() > 1) {
        std::cout<< "Pruning Wrong:" << " head_node index:" 
                    << head_node->index
                    << " head_node children size:"
                    << head_node->children.size()
                    << std::endl;
    }
}

int Tree::preTravel(std::shared_ptr<treeNode> node) {
    if (node->children.size() == 0){
        leaf_node.push_back(node);
        return 1;
    }
    for (int i=0; i < node->children.size(); i++) {
        preTravel(node->children[i]);
    }
}


int Tree::generateLeafNode() {
    leaf_node.clear();
    if (head_node->children.size() == 0){
        leaf_node.push_back(head_node);
    }
    for (int i=0; i < head_node->children.size();i++) {
       preTravel(head_node->children[i]); 
    }
}

/* 
inference funtion of sending tracking results
Input: the tracking result
*/
int Tree::sentResult (std::vector<int> route, cv::Rect_<float>& result) {

    if (route[0] == 0) {
        return 0;
    }
    if (route[1] == head_node->index) {
        result = head_node->box;
        return 1;
    } else {
        return 0;
    } 
}


/*
inference function of sending tracking results
Input: the tracking result
*/
int Tree::sentResult (cv::Rect_<float>& result) {

    if (leaf_node[0]->level >= N) {
        result = head_node->box;
        return 1;
    } else { 
        return 0;
    }
}

/*
print the tree node level by level
*/
void Tree::printTree (std::shared_ptr<treeNode> root) {

    if (root == NULL) {
        return;
    }

    int i;
    int current_level = 1;
    std::cout << "[";

    std::deque<std::shared_ptr<treeNode>> queue_tree_node;
    queue_tree_node.push_back(root);
   
    while (queue_tree_node.size()) {
        std::shared_ptr<treeNode> temp_node = queue_tree_node.front();
        queue_tree_node.pop_front();

        if (temp_node->level>current_level) {
            std::cout<<std::endl;
            std::cout<<"[";
            current_level = temp_node->level;
        }
        //print tree
        std::cout<<temp_node->index<<" ";
        
        if (queue_tree_node.size()==0) {
            std::cout << "]";
        }

        if (queue_tree_node.size() !=0 && 
            temp_node->parent != queue_tree_node[0]->parent) {
            std::cout<<"][";
        }

        for (i = 0; i < temp_node->children.size(); i++) {
            if (temp_node->children[i]) {
                queue_tree_node.push_back(temp_node->children[i]);
            }
        }
    }
    std::cout<<std::endl;
}

/* 
An ICH node appears when the tree is not confirmed
index:-1, box: head_node's box, score:0.0001, children: the head_node's children
the ICH node will become the head_node of a tree after pruning
*/
int Tree::createICH() {

    //exclude the first N frames
    if (leaf_node[0]->level - head_node->level == (N-1)) {  
        int i, j;
        std::shared_ptr<treeNode> ICH_ptr(new treeNode);
        
        for (i = 0; i < head_node->children.size(); i++) {
            for (j = 0; j < head_node->children[i]->children.size(); j++) {
                ICH_ptr->children.push_back(head_node->children[i]->children[j]);
            }
        }
        ICH_ptr->parent = head_node;
        ICH_ptr->level = head_node->level+1;
        ICH_ptr->score = 0.01;
        ICH_ptr->index = -1;
        ICH_ptr->box = head_node->box;
        // if Kalman theory is used:
        //ICH_ptr->kalman_tracker = head_node->kalman_tracker; 
        head_node = ICH_ptr;

        //delete the same child of ICH node and establish the parent-child relationship
        for (i = 0; i < ICH_ptr->children.size(); i++) {
            if (i == 0) {
                ICH_ptr->children[i]->parent = ICH_ptr;
            } else {
                for (j = 0; j < i; j++) {
                    if (ICH_ptr->children[i]->index == ICH_ptr->children[j]->index) {
                        ICH_ptr->children.erase(ICH_ptr->children.begin()+i);
                        i--;
                    }
                    else {
                        ICH_ptr->children[i]->parent = ICH_ptr;
                    }
                }
            } 
        }
    }
    else {
        return 1;
    }
}

// get functions:

int Tree::getId() {
    return id;
}

int Tree::getN() {
    return N;
}

int Tree::getLabel() {
    return label;
}

std::shared_ptr<treeNode> Tree::getRoot() {
    return root_node;
}

std::shared_ptr<treeNode> Tree::getHead() {
    return head_node;
}

std::vector<std::shared_ptr<treeNode>> Tree::getLeafNode() {
    return leaf_node;
}
