#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <vector>
#include <map>

typedef struct VexNode{
    float score;
    int id;
    std::vector<int> path;
} VexNode;

class Graph{
    private:
        int m_node_num;
        std::vector<VexNode> m_node_list;
        std::vector<std::vector<int>> m_adj_mat;
        std::vector<std::vector<int>> m_dej_mat;
        
        std::vector<int> m_max_clique;
        std::vector<std::vector<int>> m_stk;
        int max, ns;
        float m_score;
        
        std::vector<int> m_dep;
        std::vector<int> m_vetex_list;
        std::vector<float> m_score_list;
        std::vector<std::vector<int>> m_max_clique_list;

    public:
        Graph();
        Graph(std::vector<VexNode>);
        int DFS(int, int, int, float);
        int mwis(std::map<int,std::vector<int>>&);

        int deal_candiate(int, int, std::vector<int>&, std::vector<int>, std::vector<int>&);
        int mwis_greed(std::map<int, std::vector<int>>& routes);
        int printGraph();
};

#endif
