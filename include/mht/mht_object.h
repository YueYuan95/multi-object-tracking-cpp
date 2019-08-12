#include <iostream>

typedef treeNode{
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

typdef edgeNode{
    int adjvex;
    edgeNode* nextEdge;
} edgeNode;

typedef vexNode{
    float score;
    vector<int> path;
    edgeNode* firstEdge;
} vexNode;

class Tree{
    private:
        int id;
        std::shared_ptr<Node> root_node;
        std::shared_prt<Node> head_node;
        std::vector<std::shared_ptr<Node>> leaf_node;

    public:

        Tree(std::shared_prt<Node> root);

        /*set function*/
        int setHead();

        /*get function*/
        std::shared_ptr<Node> getHead();
        std::vector<std::shared_ptr<Node>> getLeafNode();

        int addNode();
        int print();

};

class Graph{
    private:
        int vexnum;
        int edgenum;
        vector<VexNode> adjlist;
    public:
        int DFS();
        std::vector<std::vector<int>> 
        int print();

}

int TreeToGraph();
int GraphToTree();