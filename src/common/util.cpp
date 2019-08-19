#include "util.h"


/*
 *===================================
 *  Tree convert to Graph
 *
 *===================================
*/

int preorderTraversal(treeNode tree_node, std::vector<int>& path, 
                std::vector<std::vector<int>>& path_list){
    
    path.push_back(tree_node.index);
    if(tree_node.children.size() == 0){
        path_list.push_back(path);
    }else{
        for(int i=0; i < tree_node.children.size(); i++){
            preorderTraversal(*(tree_node.children[i]), path, path_list);
            path.pop_back();
        }
    }

}

int backTraversal(treeNode tree_node, std::shared_ptr<treeNode> head_node,
         std::vector<int>& path, std::vector<std::vector<int>>& path_list, int N){
    path.push_back(tree_node.index);
    
    /*When the depth of the tree is not big than N*/
    if(tree_node.parent == head_node){
        path.push_back(tree_node.parent->index);
        if(path.size() < N){
            for(int i=N-path.size()+1;i>0;i--){
                path.push_back(0);
            }
        }
        path_list.push_back(path);
        return 1;
    }
   /*When this node is a root node*/
    if(tree_node.parent == NULL &&  path.size() < N){
        
        for(int i=N-path.size()+1; i > 0; i--){
            path.push_back(0);
        }
        path_list.push_back(path);
        return 1;

    }
    if(tree_node.parent != NULL){

        backTraversal(*(tree_node.parent), head_node, path, path_list, N);
    }
}

int TreeToGraph(std::vector<Tree> tree_list, Graph& graph){

    std::vector<int> path; 
    std::vector<std::vector<int>> path_list;
    std::vector<VexNode> graph_node_list;
    
    for(auto tree : tree_list){
        
        std::cout<<"Tree No."<<tree.getId()<<std::endl;
        //preorderTraversal(tree.getHead(),path, path_list);
        for(auto leaf : tree.getLeafNode()){
            path.clear();
            backTraversal(*(leaf), tree.getHead(), path, path_list, tree.getN());
        }
        for(auto path : path_list){
            VexNode graph_node;
            graph_node.path.clear();
            for(int i = path.size()-1; i >=0; i--){
                std::cout<<path[i]<<" ";
                graph_node.id = tree.getId();
                graph_node.path.push_back(path[i]);
            }
            std::cout<<std::endl;
            graph_node_list.push_back(graph_node);
         }
         path_list.clear();
    }
    
    graph = Graph(graph_node_list);
}

int visualize(bool visual, byavs::TrackeObjectCPUs results)
{
    
    //byavs::TrackeObjectCPUs results;
    MHT_tracker tracker;
    tracker.sentResult(results);
    int frame_count;

    while (!results.empty())
    {

        dect_rects = results.front();
        curr_img = ;//files[frame_count-1];
        imgBGR = cv::imread(curr_img);

        VectorToDetectObject(dect_rects,detection);
        tracker.inference(imgBGR, detection, keyObjects);
        TrackeObjectToVector(keyObjects, tracker_result);

        if(visual)
        {
            Mat imgShow = imgBGR;
                        // for(int i=0; i < dect_rects.size(); i++){
                        //      rectangle(imgBGR,dect_rects[i],Scalar(255,0,0),3,1,0);
                        // }
            for(int j=0; j < result.size(); j++)
            {
                std::string id = result[j].identity_code;
                std::Point left_top = result[j].c1;
                std::Point right_bottom = result[j].c2;
                Scalar color = result[j].color;
                cv::putText(imgShow, id, left_top, FONT_HERSHEY_SIMPLEX, 1 ,color,3,8);
                cv::rectangle(imgShow, left_top, right_bottom, color, 3, 1, 0);
            }
                        // resize(imgShow,imgShow,Size(imgShow.cols/2,imgShow.rows/2),0,00,INTER_LINEAR);
                        // imshow(seq,imgShow);
            imwrite("result/"+to_string(frame_count)+".jpg", imgShow);
                //      waitKey(1);
        }

        frame_count++;
        results.pop();
    }
}

