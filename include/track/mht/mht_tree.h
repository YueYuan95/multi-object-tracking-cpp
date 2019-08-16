#ifndef __TREE_NODE_H_/*capitalize class name*/
#define __TREE_NODE_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <deque>

typedef struct treeNode{
    cv::Rect_<float> box;/*for drawing boxs*/
    float score;
    int level;
    int index;
    /* One node can only have one parent, if 
    different node gating this same box or node to be chirld,
    this same box should be different chirlden nodes in different parents node
    */
    std::shared_ptr<treeNode> parent;/*shared_ptr is an AI point*/
    std::vector<std::shared_ptr<treeNode>> children;

} treeNode;

class Tree{
    private:
        static int tree_id;
        int id, N, label;
        std::shared_ptr<treeNode> root_node;
        std::shared_ptr<treeNode> head_node;
        std::vector<std::shared_ptr<treeNode>> leaf_node;

    public:

        Tree(std::shared_ptr<treeNode> root, int label, int N);

        /*set function*/
        int setHead();
        
        /*get function*/
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

        int sentResult(std::vector<int>, cv::Rect_<float>&);
        int sentResult(cv::Rect_<float>&);
        void printTree(std::shared_ptr<treeNode> root);

};

#endif
