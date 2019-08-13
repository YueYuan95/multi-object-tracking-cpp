#include <iostream>
#include <vector>

typedef struct VexNode{
    float score;
    std::vector<int> path;
} VexNode;

class Graph{
    private:
        int vexnum;
        int edgenum;
        std::vector<VexNode> node_list;
        std::vector<std::vector<int>> adj_mat;

        std::vector<std::vector<int>> dej_mat;
        std::vector<int> dp;
        std::vector<std::vector<int>> stk;
        int mx, ns;
        float score;

        std::vector<int> vetexlist;
        std::vector<int> max_clique_list;

    public:
        Graph(std::vector<VexNode>);
        int DFS(int, int, int);
        int mwis();
        int print();
};
