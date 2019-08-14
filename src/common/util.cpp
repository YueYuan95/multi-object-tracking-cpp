#include "util.h"

int preorderTraversal(treeNode tree_node, std::vector<int>& path, 
                std::vecteor<std::vector<int>>& path_list){
    
    path.push_back(tree_node.index);
    if(tree_node.chirlden.size() == 0){
        path_list.push(path);
    }else{
        for(int i=0; i < tree_node.chirlden.size(); i++){
            preordeTraversal(tree_node[i], path, path_list);i
            path.pop_back();
        }
    }

}

int backTraversal(treeNode tree_node, treeNode head_node, std::vector<int>& path, 
                std::vector<std::vector<int>>& path_list, int N){
    path.push_back(tree_node.index);
    /*When the depth of the tree is not big than N*/
    if(tree_node.parent == head_node){
        path.push_back(tree_node.parent.index, path, path_list);
        if(path.size() < N){
            for(int i=N-path.size();i>0;i--){
                path.push_back(0);
            }
        }
        path_list.push_back(path);
        return;
    }
    /*When this node is a root node*/
    if(tree_node.parent == null &&  path.size() < N){
        for(int i=N-path.size(); i > 0; i--){
            path.push_back(0);
        }
        path_list.push_back(path);
        return;

    }
    backTraversal(tree_node.parent, head_node, path, path_list);
}

int TreeToGraph(std::vector<Tree> tree_list, Graph graph){

    std::vector<int> path; 
    std::vector<std::vector<int>> path_list;

    for(auto tree : tree_list){
        
        //temp_node.path.clear();
        //tmep_node.path.push_back();
        
        //preorderTraversal(tree.getHead(),path, path_list);
        for(auto leaf : tree.leaf_node){
             path.clear();
             backTraversal(leaf, path, path_list, tree.getN());
        }
        
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


