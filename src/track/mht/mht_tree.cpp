#include "mht_tree.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <vector>
#include <deque>

int Tree::tree_id = 1;

Tree::Tree(std::shared_ptr<treeNode> root, int i, int n)
{
       
    id = tree_id;
    tree_id++;
    label = i;
    root_node = root;
    leaf_node.push_back(root);
    head_node = root;
    N = n;
    miss_times = 0;///
    hit_times = 0;///

}

int Tree::addNode(int node_index, std::shared_ptr<treeNode> tree_Node){

    tree_Node->parent = leaf_node[node_index];
    tree_Node->children.clear();
    leaf_node[node_index]->children.push_back(tree_Node);
    
}

int Tree::changeLeaf(){
   
    if(leaf_node.size() == 0) return 1;
    if(leaf_node.size() == 1 && leaf_node[0]->children.size() == 0) return 1;
    std::vector<std::shared_ptr<treeNode>> candidate;
    candidate.clear();
    for(int i=0; i < leaf_node.size(); i++){
        for(int j=0; j < leaf_node[i]->children.size();j++){
            candidate.push_back(leaf_node[i]->children[j]);
        }
    }
    leaf_node = candidate;
            
}

int Tree::addNode(std::map<int, std::vector<std::shared_ptr<treeNode>>> dict)
{
    //search the indexes' nodes in leaf_node
    //then add the nodes' children
    std::map<int, std::vector<std::shared_ptr<treeNode>>>::iterator it;
    it = dict.begin();
    std::vector<std::shared_ptr<treeNode>> sub_children_node;
    while(it!=dict.end())
    {
        leaf_node[it->first]->children = it->second;
        for(int i=0;i<it->second.size();i++){
            it->second[i]->parent = leaf_node[it->first]; 
        }
        it++;
        // leaf_node.clear();
        // 
        // for(i=0;i<leaf_node.size(); i++)
        // {
        //     leaf_node.push_back(search_node[i]);
        //     leaf_node[i]->level++;
        // }
         /*for(j=0; j<search_node.size(); j++)
         {
             search_node[j]->children = it->second;
             leaf_node = search_node[j]->children;
         }
         it++;*/
    }
    sub_children_node.clear();
    for(int i=0;i<leaf_node.size();i++){
        for(int j=0;j<leaf_node[i]->children.size();j++){
            sub_children_node.push_back(leaf_node[i]->children[j]);
        }
    }
    leaf_node = sub_children_node; 


}
        
int Tree::pruning(std::vector<int> route)
{
    //root is static
    //int variable is the index of head_node
    //vector saves the indexes of the nodes on the route
    //int i ,j , count, is_valid;
   // std::vector<int> route_list;
    
    //search for the head_node(if head_node is known, this step is unnesessary)
   /* for(i=0; i<leaf_node.size();i++)
    {
        if(leaf_node[i]->index == route->first){
            head_node = leaf_node[i];
        }
    }*/

    //std::shared_ptr<treeNode> head_node_temp;
    
    //delete
    //for(auto i : route){
    //    std::cout<<i<<std::endl;
    //}

    /*if(id == 15){
        for(auto i : route){
            std::cout<<i<<" "<<std::endl;
        }
    }*/

    // if(route[0] != head_node->index && route[0] != 0){
    //     std::cout<<"Head Index is not the frist of the path"<<std::endl;
    // }
    // if(route[0] == 0){
    //     return 1;
    // }

    // if(route[0] != head_node->index){
    //     std::cout<<"Head Index is not the frist of the path"<<std::endl;
    // }

    if((leaf_node[0]->level - head_node->level) < (N-1)){       
        //result = head_node->box;
        //std::cout<<"head_node->level:"<<head_node->level<<" leaf_node[0]->level:"<<leaf_node[0]->level<<" head_node->box:"<<result<<std::endl;///
        return 1;
    }

    std::cout<<id<<" before pruning CTree,head_node children size:"<<head_node->children.size()<<std::endl;///
    for(int i=0;i<head_node->children.size();i++)
    {
        if(head_node->children[i]->index !=route[1])
        {
            head_node->children.erase(head_node->children.begin()+i);
            i--;
        }
        
    }
    /*std::vector<std::shared_ptr<treeNode>>::iterator iter;
    
      for(iter = head_node->children.begin(); iter != head_node->children.end()+1;){

        if( (*iter)->index != route[1]){
           iter= head_node->children.erase(iter);
           if(id==31){
               std::cout<<(*iter)->index<<std::endl;///
           }
        }
        if(iter != head_node->children.end()) iter++;
    }*/

    if(head_node->children.size() == 0){
        return 1;
    }
    if(head_node->children.size() == 1){
        head_node = head_node->children[0];
        generateLeafNode();
        return 1;
    } 
    if(head_node->children.size() > 1){
        std::cout<<"Pruning Wrong:"<<" head_node index:"<<head_node->index<<" head_node children size:"<<head_node->children.size()<<std::endl;
    }
    //std::cout<<head_node->index<<std::endl;
    //m_kalman_tracker.update(head_node.box);
    //head_node.box = m_kalman_tracker.getBbox();
    
/*

    if(head_node->children.size()>1)
    {
        count++;
        for(i=0;i<head_node->children.size();i++)
        {
            route_list.push_back(i);//is that necessary?
        }

        for(i=0;i<head_node->children.size();i++)
        {
           head_node_temp = head_node->children[i];
           for(j=0;j<head_node_temp->children.size();j++)
           {
                if(head_node_temp->children[j]->index == route->second[count])
                {
                    is_valid++;
                }
           }
           
        }
            
            head_node_temp = route_list[i];
            for(j=0; j<head_node_temp->children.size(); j++)
            {
                if()
            }
            while(head_node_temp->children)
        }
    }*/

}

int Tree::preTravel(std::shared_ptr<treeNode> node){
    
    if(node->children.size() == 0){
        leaf_node.push_back(node);
        return 1;
    }
    for(int i=0; i < node->children.size(); i++){
        preTravel(node->children[i]);
    }
}


int Tree::generateLeafNode(){
    leaf_node.clear();
    if(head_node->children.size() == 0){
        leaf_node.push_back(head_node);
    }
    for(int i=0;i < head_node->children.size();i++){
       preTravel(head_node->children[i]); 
    }
}

int Tree::sentResult(std::vector<int> route, cv::Rect_<float>& result){

    if(route[0] == 0){
        return 0;
    }
    if(route[1] == head_node->index){
        result = head_node->box;
        
        return 1;
    }
    else{

        return 0;
    }
    
    
}

int Tree::sentResult(cv::Rect_<float>& result){
    
    
    if( leaf_node[0]->level >= N){///((leaf_node[0]->level - head_node->level) == (N-2))//(head_node->index!=-1)

        result = head_node->box;
        //std::cout<<"head_node->level:"<<head_node->level<<" leaf_node[0]->level:"<<leaf_node[0]->level<<" head_node->box:"<<result<<std::endl;///
        //std::cout<<" head_node->box:"<<result<<std::endl;///
        return 1;
        
    }
    else{
        //std::cout<<"head_node->level:"<<head_node->level<<" leaf_node[0]->level:"<<leaf_node[0]->level<<" head_node->box:"<<result<<std::endl;///
        //result = { };
        return 0;
    }
    //return 1;
    
}

void Tree::printTree(std::shared_ptr<treeNode> root)
{
    if(root == NULL)
    {
        return;
    }

    int i;
    int current_level = 1;
    std::cout<<"[";

    std::deque<std::shared_ptr<treeNode>> queue_tree_node;
    queue_tree_node.push_back(root);
   
    while(queue_tree_node.size())
    {
        
        std::shared_ptr<treeNode> temp_node = queue_tree_node.front();
        queue_tree_node.pop_front();

        if(temp_node->level>current_level){
            std::cout<<std::endl;
            std::cout<<"[";
            current_level = temp_node->level;
        }
        //print tree
        std::cout<<temp_node->index<<" ";
        
        if(queue_tree_node.size()==0)
        {
            std::cout<<"]";
        }

        if(queue_tree_node.size()!=0 && temp_node->parent!=queue_tree_node[0]->parent)
        {
            std::cout<<"][";
        }

        for(i=0;i<temp_node->children.size();i++){
            if(temp_node->children[i]){
                queue_tree_node.push_back(temp_node->children[i]);
            }
        }
    }
    std::cout<<std::endl;
}

int Tree::createICH(){

    if (leaf_node[0]->level - head_node->level == (N-1))//exclude the first N frames
    {
        int i, j;
        std::shared_ptr<treeNode> ICH_ptr(new treeNode);
        
        for(i=0; i<head_node->children.size(); i++)
        {
            for(j=0; j<head_node->children[i]->children.size(); j++)
            {
                ICH_ptr->children.push_back(head_node->children[i]->children[j]);
            }
        }
        //std::cout<<"ICH_ptr->children.size:"<<ICH_ptr->children.size();
        ICH_ptr->parent = head_node;
        ICH_ptr->level = head_node->level+1;
        ICH_ptr->score = 0.01;
        ICH_ptr->index = -1;
        ICH_ptr->box = head_node->box;
        //ICH_ptr->kalman_tracker = head_node->kalman_tracker; 
        head_node = ICH_ptr;

        /*delete the same child of ICH node
          establish the parent-child relationship*/
        for(i=0; i<ICH_ptr->children.size(); i++)
        {
            if(i==0)
            {
                ICH_ptr->children[i]->parent = ICH_ptr;
            }
            else
            {
                for(j=0; j<i; j++)
                {
                    if(ICH_ptr->children[i]->index==ICH_ptr->children[j]->index)
                    {
                        ICH_ptr->children.erase(ICH_ptr->children.begin()+i);
                        i--;
                    }
                    else
                    {
                        ICH_ptr->children[i]->parent = ICH_ptr;
                    }
                }
            }
            
        }

        
        //std::cout<<" head_node children size:"<<head_node->children.size()<<" head_node->level:"<<head_node->level<<" leaf_node->level:"<<leaf_node[0]->level<<" head_node->box:"<<head_node->box<<std::endl;
    }
    else
    {
        return 1;
    }

}

int Tree::getId()
{
    return id;
}
int Tree::getN(){
    return N;
}

int Tree::getLabel()
{
    return label;
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
