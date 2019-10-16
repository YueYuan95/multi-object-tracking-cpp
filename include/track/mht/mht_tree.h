/**************************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/include
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The struct, treenode, is a node of a tree
                   *The class, Tree, is to form tracking tree family
                   *addNode:Adds leaf to a tree
                   *changeLeaf:Updates the current leaf_node
                   *pruning:pruns the branches of a headnode, which is the first index of a route
                   *preTravel:pre traversal
                   *generateLeafNode:pre traversal without input
                   *sentResult:inference funtion of sending tracking results
                   *printTree:print the tree level by level
                   *createICH: create ICH head node for inconfirmed tree
                   *get functions: return the values of a tree
*****************************************************************************************/

#ifndef __TREE_NODE_H_
#define __TREE_NODE_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <deque>
#include "kalman_tracker.h"

typedef struct treeNode {
    cv::Rect_<float> box;
    float score;
    int level;
    int index;
    /* One node can only have one parent, if 
    different node gating this same box or node to be child,
    this same box should be different children nodes in different parents node
    */
    std::shared_ptr<treeNode> parent;
    std::vector<std::shared_ptr<treeNode>> children;
    KalmanTracker kalman_tracker;
} treeNode;

class Tree {
 private:
  static int tree_id;
  int id, N, label;
  std::shared_ptr<treeNode> root_node;
  std::shared_ptr<treeNode> head_node;
  std::vector<std::shared_ptr<treeNode>> leaf_node;
 public:
  Tree(std::shared_ptr<treeNode> root, int label, int N);
    
  //public number
  // miss_times records the miss_object 
  // by counting how many times the tree isn't gated or confirmed
  int miss_times;
  // hit_times records the miss_object 
  // by counting how many times the tree is confirmed
  // currently DEPRECATED
  //int hit_times;

  int addNode(int, std::shared_ptr<treeNode>);
  int changeLeaf();
  int addNode(std::map<int, std::vector<std::shared_ptr<treeNode>>> dict);
  int pruning(std::vector<int> route);
  int generateLeafNode();
  int preTravel(std::shared_ptr<treeNode>);

  int sentResult(std::vector<int>, cv::Rect_<float>&);
  int sentResult(cv::Rect_<float>&);
  void printTree(std::shared_ptr<treeNode> root);

  int createICH();//create inconfirmed head_node
  //get function
  int getId();
  int getLabel();
  int getN();
  std::shared_ptr<treeNode> getRoot();
  std::shared_ptr<treeNode> getHead();
  std::vector<std::shared_ptr<treeNode>> getLeafNode();

  
};

#endif
