#include "mht_tracker.h"
#include "mht_tree.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <math.h>

void MHT_tracker::gating(std::vector<cv::Rect_<float>> det_result, std::vector<Tree> tree_list)
{
    int i, j;
    float x1, y1, x2, y2, distance;
    float threshold = 20;//threshold of the distance,changeable
    float xx1, yy1, xx2, yy2, w, h, IOU;//IOU is the score
    float zero = 0;
    bool success_flag = false; 

    //push the leaf_node of the trees into a vector
    std::vector<std::shared_ptr<treeNode>> leaf_node_list;
    for(i=0; i<tree_list.size(); i++)
    {
        for(j=0; j<tree_list[i].getLeafNode().size(); j++)
        {
            if(tree_list[i].getLeafNode()[j]->index!=0)
            {
                leaf_node_list.push_back(tree_list[i].getLeafNode()[j]);//
            }
            else //add a zero_node to the leaf_node's zero_node
            {
                treeNode zero_node;//,creat a zero_node which has no box
                std::shared_ptr<treeNode> zero_node_ptr(new treeNode(zero_node));
                zero_node.index = 0;
                zero_node.score = -1;
                zero_node.level = tree_list[i].getLeafNode()[j]->level+1;
                zero_node.parent = tree_list[i].getLeafNode()[j];
                tree_list[i].getLeafNode()[j]->children.push_back(zero_node_ptr);
            }  
        }
    }
    
    //match the box to leaf_nodes in the leaf_node_list, the leaf_node_list has no zero_nodes
    for(i=0; i<det_result.size(); i++)
    {
        //create a node for each det_result
        treeNode det_node;
        std::shared_ptr<treeNode> det_node_ptr(new treeNode(det_node));
        det_node.box = det_result[i];
        det_node.index = i+1;
        det_node.level = 1;//initialize the level of each tree/node 1

        //caculate the central coordinate of the det_result;
        x1 = det_node.box.x + det_node.box.width/2;
        y1 = det_node.box.y + det_node.box.height/2;

        for(j=0; j<leaf_node_list.size(); j++)
        {
            //caculate the distance,could be a function 
            //here we caculate the Euclidean distance
            x2 = leaf_node_list[j]->box.x + leaf_node_list[j]->box.width/2;
            y2 = leaf_node_list[j]->box.y + leaf_node_list[j]->box.height/2;
            distance = sqrt(pow(x1-x2,2)+pow(y1-y2,2));
            //caculate the score, which is IOU here
            xx1 = std::max(det_node.box.x, leaf_node_list[j]->box.x);
            yy1 = std::max(det_node.box.y, leaf_node_list[j]->box.y);
            xx2 = std::min(det_node.box.x + det_node.box.width,leaf_node_list[j]->box.x + leaf_node_list[j]->box.width);
            yy2 = std::min(det_node.box.y+det_node.box.height,leaf_node_list[j]->box.y + leaf_node_list[j]->box.height);
            w = std::max(zero, xx2-xx1);
            h = std::max(zero, yy2-yy1);
            IOU = w*h/(det_node.box.width*det_node.box.height+leaf_node_list[j]->box.width*leaf_node_list[j]->box.height-w*h);
            
            if(distance<threshold)
            {
                //addNode??
                leaf_node_list[j]->children.push_back(det_node_ptr);
                det_node_ptr->score = IOU;
                det_node_ptr->level = leaf_node_list[j]->level+1;
                det_node_ptr->parent = leaf_node_list[j];
                success_flag = true;
            }
        }
        //for those boxes which do not match any existing trees:create a new tree for them
        if(success_flag == false)
        {
            Tree gate(det_node_ptr,3,3);//label=3,N=3
        }
        
    }

    //for those leaf_nodes which are not matched box:add zero_node to them
        for(i=0; i<leaf_node_list.size(); i++)
        {
            if(leaf_node_list[i]->children.size() == 0)
            {
                
                treeNode zero_node;//,creat a zero_node which has no box
                std::shared_ptr<treeNode> zero_node_ptr(new treeNode(zero_node));
                zero_node.index = 0;
                zero_node.score = -1;
                zero_node.level = leaf_node_list[i]->level+1;
                zero_node.parent = leaf_node_list[i];
                leaf_node_list[i]->children.push_back(zero_node_ptr);
            }
        }
}

/*std::vector<std::shared_ptr<treeNode>> MHT_tracker::find_leaf_node(std::shared_ptr<treeNode> root)
{
    int i;
    while(root->children.size())
        {
            for(i=0; i<root->children.size(); i++)
            {
                temp = root->children[i];
                find_leaf_node(root)
            }
        }
}*/