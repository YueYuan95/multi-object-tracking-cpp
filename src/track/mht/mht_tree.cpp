#include "mht_tree.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <vector>

Tree::Tree(std::shared_ptr<treeNode> root, int id_, int N)
{
        
    id = id_;
    root_node = root;
    leaf_node.push_back(root);//
    head_node = root;
       
}

int Tree::addNode(std::map<int, std::vector<std::shared_ptr<treeNode>>> dict)
{
    //search the indexes' nodes in leaf_node
    //then add the nodes' children
    std::map< int, std::vector< std::shared_ptr<treeNode> > > ::iterator it;
    it = dict.begin();
    int i, j;
    std::vector<std::shared_ptr<treeNode>> search_node;
    while(it!=dict.end())
    {
        
        //traverse in leaf_node
        for(i=0; i<leaf_node.size(); i++)
        {
            if(leaf_node[i]->index==it->first)
            {
                search_node.push_back(leaf_node[i]);
            }
        }
        for(j=0; j<search_node.size(); j++)
        {
            search_node[j]->children = it->second;
            leaf_node = search_node[j]->children;
        }
        it++;
        search_node.clear();  
    }
     


}
        
int Tree::pruning()
{

}

int Tree::getId()
{
    return id;
}

std::shared_ptr<treeNode> Tree::getRoot()
{
    return root_node;
}

std::shared_ptr<treeNode> Tree::getHead()
{
    return head_node;
}

std::vector<std::shared_ptr<treeNode>> Tree::getLeafNode()
{
    return leaf_node;
}
