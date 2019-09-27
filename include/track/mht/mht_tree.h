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

  //set function
  int setHead();

  //get function
  int getId();
  int getLabel();
  int getN();
  std::shared_ptr<treeNode> getRoot();
  std::shared_ptr<treeNode> getHead();
  std::vector<std::shared_ptr<treeNode>> getLeafNode();

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
};

#endif
