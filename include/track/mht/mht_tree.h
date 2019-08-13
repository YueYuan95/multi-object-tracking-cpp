#ifndef __TREE_NODE_H_
#define __TREE_NODE_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>

typedef struct treeNode{
    cv::Rect_<float> box;
    float score;
    int level;
    int index;
    /* One node can only have one parent, if 
    different node gating this same box or node to be chirld,
    this same box should be different chirlden nodes in different parents node
    */
    std::shared_ptr<treeNode> parent;
    std::vector<std::shared_ptr<treeNode>> chirlden;

} treeNode;

class Tree{
    private:
        int id;
        std::shared_ptr<treeNode> root_node;
        std::shared_ptr<treeNode> head_node;
        std::vector<std::shared_ptr<treeNode>> leaf_node;

    public:

        Tree(std::shared_ptr<treeNode> root, int N);

        /*set function*/
        int setHead();

        /*get function*/
        std::shared_ptr<treeNode> getHead();
        std::vector<std::shared_ptr<treeNode>> getLeafNode();

        int addNode(int, treeNode);
        int pruning();
        int print();

};

#endif
