#include "mht_tracker.h"

int MHT_tracker::inference(std::vector<cv::Rect_<float>> det_result,byavs::TrackeObjectCPUs& results){

    Graph graph;
    std::map<int, std::vector<int>> path;
    std::cout<<"Gating......."<<std::endl;
    gating(det_result);
    std::cout<<"Tree to Graph"<<std::endl;
    TreeToGraph(graph);
    std::cout<<"Sovle MWIS..."<<std::endl;
    sovle_mwis(graph, path);
    std::cout<<"Pruning......"<<std::endl;
    pruning(path);
    std::cout<<"Send Result..."<<std::endl;
    sentResult(results);
    std::cout<<"End one tracking"<<std::endl;
}


int MHT_tracker::gating(std::vector<cv::Rect_<float>> det_result)
{
    int i, j;
    float x1, y1, x2, y2, distance;
    float threshold = 40;//threshold of the distance,changeable
    double iou; 
    float zero = 0;
    bool success_flag; 
    std::vector<Tree> new_tree_list;

    //push the leaf_node of the trees into a vector
    std::vector<std::shared_ptr<treeNode>> leaf_node_list;
    for(i=0; i<tree_list.size(); i++)
    {
        for(j=0; j<tree_list[i].getLeafNode().size(); j++)
        {
            leaf_node_list.push_back(tree_list[i].getLeafNode()[j]);//
        }
    }
    
    //match the box to leaf_nodes in the leaf_node_list, the leaf_node_list has no zero_nodes
    for(i=0; i<det_result.size(); i++)
    {
        //create a node for each det_result
        success_flag = false;

        //caculate the central coordinate of the det_result;
        x1 = det_result[i].x + det_result[i].width/2;
        y1 = det_result[i].y + det_result[i].height/2;
        //std::cout<<"x1:"<<x1<<"y1:"<<y1<<std::endl;

        for(j=0; j<leaf_node_list.size(); j++)
        {
            //caculate the distance,could be a function 
            //here we caculate the Euclidean distance
            KalmanTracker temp_kalman = leaf_node_list[j]->kalman_tracker;
            temp_kalman.predict();
            cv::Rect_<float> predict_box = temp_kalman.getBbox();
            
            x2 = predict_box.x + predict_box.width/2;
            y2 = predict_box.y + predict_box.height/2;
            distance = sqrt(pow(x1-x2,2)+pow(y1-y2,2));

            iou = get_iou(det_result[i], predict_box); 

            if(distance<threshold)
            {
                //addNode??
                std::shared_ptr<treeNode> det_node_ptr(new treeNode);
                det_node_ptr->index = i+1;
                det_node_ptr->score = iou;
                det_node_ptr->level = leaf_node_list[j]->level+1;
                det_node_ptr->parent = leaf_node_list[j];
                det_node_ptr->kalman_tracker = leaf_node_list[j]->kalman_tracker;
                det_node_ptr->kalman_tracker.update(det_result[i]);
                det_node_ptr->box = det_node_ptr->kalman_tracker.getBbox();

                leaf_node_list[j]->children.push_back(det_node_ptr);
                success_flag = true;
                //std::cout<<"Detect index :"<< i+1 << " Leaf Node Index : "<< leaf_node_list[j]->index <<"  distance:"<<distance<<std::endl;
            }
        }
        //for those boxes which do not match any existing trees:create a new tree for them
        if(success_flag == false)
        {
            std::shared_ptr<treeNode> det_node_ptr(new treeNode);
            det_node_ptr->box = det_result[i];
            det_node_ptr->index = i+1;
            det_node_ptr->score = 0.01;
            det_node_ptr->level = 1;        //initialize the level of each tree/node 1
            det_node_ptr->kalman_tracker = KalmanTracker(det_result[i], 3);
            Tree gate(det_node_ptr,3,N);    //label=3,N=3
            new_tree_list.push_back(gate);
        }
        
    }
    
    for(i=0; i<new_tree_list.size(); i++)
    {
        tree_list.push_back(new_tree_list[i]);
    }
    
    //for those leaf_nodes which are not matched box:add zero_node to them
    for(i=0; i<leaf_node_list.size(); i++)
    {
            if(leaf_node_list[i]->children.size() == 0)
            {
                std::shared_ptr<treeNode> zero_node_ptr(new treeNode);
                zero_node_ptr->kalman_tracker = leaf_node_list[i]->kalman_tracker; 
                zero_node_ptr->index = 0;
                zero_node_ptr->score = 0;
                zero_node_ptr->kalman_tracker.predict();
                zero_node_ptr->box = zero_node_ptr->kalman_tracker.getBbox();
                zero_node_ptr->level = leaf_node_list[i]->level+1;
                zero_node_ptr->parent = leaf_node_list[i];
                leaf_node_list[i]->children.push_back(zero_node_ptr);
            }
    }

    for(i=0; i < tree_list.size(); i++){
        tree_list[i].changeLeaf();
    }

    return 1;
}

int MHT_tracker::sovle_mwis(Graph graph, std::map<int, std::vector<int>>& path){

    graph.mwis(path);
}

int MHT_tracker::pruning(std::map<int, std::vector<int>> path){
    
    for(int i=0; i < tree_list.size();i++){
        
        if(path.count(tree_list[i].getId())){
            tree_list[i].pruning(path[tree_list[i].getId()]);
            tree_list[i].miss_times = 0;
            tree_list[i].hit_times += 1;
        }else{
            tree_list[i].miss_times += 1;
            tree_list[i].hit_times = 0;
        }

    }
}

int MHT_tracker::sentResult(byavs::TrackeObjectCPUs& results){

    
    for(int i=0; i < tree_list.size();){
        if(tree_list[i].miss_times < miss_time_thrd){
            if(tree_list[i].hit_times > hit_time_thrd){
                cv::Rect_<float> bbox;
                byavs::BboxInfo box;
                byavs::TrackeObjectCPU trk_obj_cpu;
                tree_list[i].sentResult(bbox);
                trk_obj_cpu.label = tree_list[i].getLabel();
                trk_obj_cpu.id = tree_list[i].getId();
                trk_obj_cpu.box = {(int)bbox.x, (int)bbox.y, (int)bbox.width, (int)bbox.height};
                results.push_back(trk_obj_cpu);
            }
            i++; 
        }else{
            tree_list.erase(tree_list.begin()+i);
        }
        
    }

}

std::vector<Tree> MHT_tracker::get_tree_list()
{
    return tree_list;
}

int MHT_tracker::backTraversal(treeNode tree_node, std::shared_ptr<treeNode> head_node, 
                std::vector<int>& path, std::vector<float>& path_score, std::vector<std::vector<int>>& path_list, 
                std::vector<std::vector<float>>& path_score_list,int N){

    path.push_back(tree_node.index);
    path_score.push_back(tree_node.score); 
    /*When the depth of the tree is not big than N*/
    if(tree_node.parent == head_node){
        path.push_back(tree_node.parent->index);
        path_score.push_back(tree_node.parent->score);
        if(path.size() <= N){
            for(int i=N-path.size()+1;i>0;i--){
                path.push_back(0);
                path_score.push_back(0.0);
            }
        }
        path_list.push_back(path);
        path_score_list.push_back(path_score);
        return 1;
    }

    /*When this node is a root node*/
    if(tree_node.parent == NULL &&  path.size() <= N){
        
        for(int i=N-path.size()+1; i > 0; i--){
            path.push_back(0);
            path_score.push_back(0.0);
        }
        path_list.push_back(path);
        path_score_list.push_back(path_score);
        return 1;

    }
    if(tree_node.parent != NULL){

        backTraversal(*(tree_node.parent), head_node, path, path_score, path_list, path_score_list, N);
    }
}

int MHT_tracker::TreeToGraph(Graph& graph){

    std::vector<int> path;
    std::vector<float> path_score;
    std::vector<std::vector<int>> path_list;
    std::vector<std::vector<float>> path_score_list;
    std::vector<VexNode> graph_node_list;
    
    for(auto tree : tree_list){
        
        std::cout<<"Tree No."<<tree.getId()<<std::endl;
        //preorderTraversal(tree.getHead(),path, path_list);
        for(auto leaf : tree.getLeafNode()){
            path.clear();
            path_score.clear();
            backTraversal(*(leaf), tree.getHead(), path, path_score, path_list, path_score_list,tree.getN());
        }

        if(path_list.size() > 100){
            std::vector<VexNode> temp_node_list;
            for(int i=0; i < path_list.size(); i++){
                VexNode graph_node;
                graph_node.path.clear();
                graph_node.score = 0;
                for(int j = path_list[i].size()-1; j >=0; j--){
                    std::cout<<path_list[i][j]<<" ("<<path_score_list[i][j]<<") "<<" ";
                    graph_node.id = tree.getId();
                    graph_node.score += path_score_list[i][j];
                    graph_node.path.push_back(path_list[i][j]);
                }
                std::cout<<graph_node.score;
                std::cout<<std::endl;
                temp_node_list.push_back(graph_node);
            }
            std::sort(temp_node_list.begin(), temp_node_list.end(), VexSort);
            for(int i=0; i<100; i++){
                graph_node_list.push_back(temp_node_list[i]);
            }
        }else{
            for(int i=0; i < path_list.size(); i++){
                 VexNode graph_node;
                 graph_node.path.clear();
                 graph_node.score = 0;
                 for(int j = path_list[i].size()-1; j >=0; j--){
                     std::cout<<path_list[i][j]<<" ("<<path_score_list[i][j]<<") "<<" ";
                     graph_node.id = tree.getId();
                     graph_node.score += path_score_list[i][j];
                     graph_node.path.push_back(path_list[i][j]);
                 }
                 std::cout<<graph_node.score;
                 std::cout<<std::endl;
                 graph_node_list.push_back(graph_node);
             }
        }

        path_list.clear();
        path_score_list.clear();
    }
    
    graph = Graph(graph_node_list);
}


