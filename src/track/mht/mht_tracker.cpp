#include "mht.tracker.h"
#include "mht_tree.h"
#include <iostream>
#include <opencv2/core/core.hpp>

void MHT_tracker::gating(std::vector<cv::Rect_<float>> det_result, std::vector<std::shared_ptr<treeNode>> tree_list)
{
    int i, j;
    float x1, y1, x2, y2, distance;
    float threshold = 20;//threshold of the distance,changeable
    float score_temp;//score_temp is random now, but will be caculated in another function based on the paper

    
    for(i=0: i<det_result.size(); i++)
    {
        //firstly create a node for each det_result
        treeNode det_node;
        std::shared_ptr<treeNode> det_node_ptr(new treeNode(det_node));
        det_node.box = det_result[i];
        //det_node.index = ;

        //caculate the central coordinate of the det_result;
        x1 = det_node.box.x + det_node.box.width/2;
        y1 = det_node.box.y + det_node.box.height/2;

        for(j=0; j<tree_list.size(); j++)
        {
            //caculate the distance,could be a function 
            //here we caculate the Euclidean distance
            x2 = tree_list[j]->box.x + tree_list[j]->box.width/2;
            y2 = tree_list[j]->box.y + tree_list[j]->box.height/2;
            distance = sqrt(square(x1-x2)+square(y1-y2));
            
            if(distance<threshold)
            {
                
                //addNode??
                tree_list[j]->children.push_back(det_node_ptr);
                det_node_ptr.score = score_temp;
                det_node_ptr.level = tree_list[j]->level+1;
                det_node_ptr.parent = tree_list[j];
            }
        }
    }
}