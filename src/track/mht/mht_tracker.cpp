#include "mht_tracker.h"

int MHT_tracker::inference (std::vector<cv::Rect_<float>> det_result, 
                            std::vector<float> det_result_score, 
                            byavs::TrackeObjectCPUs& results) {

    Graph graph;
    std::map<int, std::vector<int>> path;
    
    //NMS
    if (det_result.size() != 0) {
        std::cout << "NMS..." << std::endl;
        det_result = NMS( det_result, det_result_score);
        std::cout << "after NMS, det_result size:" << det_result.size() 
                    << std::endl;
    }
    
    //Constructs track tree, update, gate and score
    std::cout<<"gating..."<<std::endl;
    gating(det_result);
    
    //Changes tree to graph
    std::cout<<"TreeToGraph..."<<std::endl;
    TreeToGraph(graph);

    //Globle hypothesis formation
    std::cout<<"sovle_mwis..."<<std::endl;
    sovle_mwis(graph, path);

    //Tracking result 
    std::cout<<"sentResult..."<<std::endl;
    sentResult(results);

    //Track tree prunning
    std::cout<<"pruning..."<<std::endl;
    pruning(path);
}

std::vector<cv::Rect_<float>> MHT_tracker::NMS (
    std::vector<cv::Rect_<float>> det_result,
    std::vector<float> det_result_score) {
    //if a pair of boxes's iou is bigger than ov_thre,  
    //the det_box with lower score will be deleted
    float score_diff = 2;
    //delete intersection
    double ov_n1, ov_n2;
    for (int i = 0; i < det_result.size(); i++) {
        for (int j = i+1; j < det_result.size(); j++) {
                ov_n1 = get_ov_n1(det_result[i],det_result[j]);
                ov_n2 = get_ov_n2(det_result[i],det_result[j]);
                if (ov_n1 ==1) {
                    det_result.erase(det_result.begin()+j);
                    j--;
                }
                if (ov_n2 ==1) {
                    det_result.erase(det_result.begin()+i);
                    i--;
                    break;
                }   
        }
    }
    
    //Hungarian Algorithm
    HungarianAlgorithm HungAlgo;
    std::vector<std::vector<double>> cost_matrix 
                                    = computeDistance(det_result, det_result);
    std::vector<int> assign;
    assign.clear();
    HungAlgo.Solve(cost_matrix, assign);
    
    for (int i = 0; i < assign.size(); i++) {
        if (cost_matrix[i][assign[i]] == 1) {
            assign[i] = -1;
        } 
    }
    
    //push back detection result
    std::vector<cv::Rect_<float>> detection;
    for (int i = 0; i < assign.size(); i++) {
        if (assign[i] == -1) {
            detection.push_back(det_result[i]);
        } else {
            if (det_result_score[i]-det_result_score[assign[i]] >= score_diff) {
                detection.push_back(det_result[i]);
            } else if (det_result_score[i] - det_result_score[assign[i]] 
                       <= -score_diff) {
                continue;
            } else {
                detection.push_back(det_result[i]);
            }
        } 
    }
    return  detection;
}

int MHT_tracker::gating(std::vector<cv::Rect_<float>> det_result) {
    
    int i, j;
    float x1, y1, x2, y2, distance ;
    float distance_thre = 40;//threshold of the distance,changeable
    float iou_thre = 0.4; //threshold of IOU score
    //float thre = iou_thre * exp(-distance_thre); //DEPRECATED
    float maxScaleDiff = 1.4;
    float xx1, yy1, xx2, yy2, w, h, IOU;//IOU is the score
    float zero = 0;
    bool success_flag; 
    std::vector<Tree> new_tree_list;
    new_tree_list.clear();
    
    //pushes the leaf_node of the trees into a vector
    std::vector<std::shared_ptr<treeNode>> leaf_node_list;
    leaf_node_list.clear();
    for (i = 0; i < tree_list.size(); i++) {
        for (j = 0; j < tree_list[i].getLeafNode().size(); j++) {
            leaf_node_list.push_back(tree_list[i].getLeafNode()[j]);
        }
    }
    
    //matches each detected box to leaf_nodes in the leaf_node_list
    for (i = 0; i < det_result.size(); i++) {
            success_flag = false;

            //caculates the central coordinate of the detected box;
            x1 = det_result[i].x + det_result[i].width/2;
            y1 = det_result[i].y + det_result[i].height/2;
            
            for (j = 0; j < leaf_node_list.size(); j++) {
                //caculates the central coordinate of the leaf_node box
                x2 = leaf_node_list[j]->box.x + leaf_node_list[j]->box.width/2;
                y2 = leaf_node_list[j]->box.y + leaf_node_list[j]->box.height/2;
                //KalmanTracker temp_kalman = leaf_node_list[j]->kalman_tracker;///
                //temp_kalman.predict();///
                //cv::Rect_<float> predict_box = temp_kalman.getBbox();///
                //x2 = predict_box.x + predict_box.width/2;
                //y2 = predict_box.y + predict_box.height/2;
                //cv::Mat P = temp_kalman.getKalmanFilter().errorCovPre;
                //double cov = P.at<float>(0,0);
                //std::cout<<"erroCovPre: "<<P<<std::endl;
                //std::cout<<"cov: "<<cov<<std::endl;//P(cv::Range(0,1),cv::Range(0,1))
                distance = sqrt(pow(x1-x2,2)+pow(y1-y2,2));//caculates the Euclidean distance
                //distance = sqrt((pow(x1-x2,2)+pow(y1-y2,2))/cov);
                IOU = get_iou(det_result[i], leaf_node_list[j]->box);
                //IOU = get_iou(det_result[i], predict_box);
                
                if (IOU/(1+distance) > iou_thre/(1+distance_thre)) {
                    if (std::max(det_result[i].height/leaf_node_list[j]->box.height, 
                        leaf_node_list[j]->box.height/det_result[i].height) 
                        <= maxScaleDiff) {
                        std::shared_ptr<treeNode> det_node_ptr(new treeNode);
                        det_node_ptr->box = det_result[i];
                        det_node_ptr->index = i+1;
                        det_node_ptr->score = IOU;
                        //det_node_ptr->score = IOU/(1+distance); //DEPRECATED
                        det_node_ptr->level = leaf_node_list[j]->level+1;
                        det_node_ptr->parent = leaf_node_list[j];
                        //det_node_ptr->kalman_tracker = leaf_node_list[j]->kalman_tracker;
                        //det_node_ptr->kalman_tracker.update(det_result[i]);
                        // det_node_ptr->box = det_node_ptr->kalman_tracker.getBbox();
                        ///std::cout<<"update box:"<<i+1<<" "<<det_node_ptr->box<<std::endl;
                    
                        leaf_node_list[j]->children.push_back(det_node_ptr);
                        success_flag = true;
                    }
                    success_flag = true;
                }
                
            }
            
            //creates a new tree, for those boxes which do not match any existing trees
            if (success_flag == false) {
                std::shared_ptr<treeNode> det_node_ptr(new treeNode);
                det_node_ptr->box = det_result[i];
                det_node_ptr->index = i+1;
                det_node_ptr->score = 0.01;//new tree's head_node's score
                det_node_ptr->level = 1;//initialize the level of each tree/node 1
                //det_node_ptr->kalman_tracker = KalmanTracker(det_result[i], 3);

                Tree gate(det_node_ptr,3,N);//label=3,N=3
                new_tree_list.push_back(gate);
            }
    }

    //forms a tree list
    for (i = 0; i < new_tree_list.size(); i++) {
        tree_list.push_back(new_tree_list[i]);
    }
        
    //adds zero_node to them for leaf_nodes which are not matched any boxes
    for (i = 0; i < leaf_node_list.size(); i++) {
        if (leaf_node_list[i]->children.size() == 0) {
            std::shared_ptr<treeNode> zero_node_ptr(new treeNode);
            //zero_node_ptr->kalman_tracker = leaf_node_list[i]->kalman_tracker; 
            zero_node_ptr->index = 0;
            zero_node_ptr->score = 0;
            //zero_node_ptr->score = tree_list[i].getLeafNode()[j]->score;
            zero_node_ptr->box = leaf_node_list[i]->box;
            // zero_node_ptr->kalman_tracker.predict();
             // zero_node_ptr->box = zero_node_ptr->kalman_tracker.getBbox();
            zero_node_ptr->level = leaf_node_list[i]->level+1;
            zero_node_ptr->parent = leaf_node_list[i];
            leaf_node_list[i]->children.push_back(zero_node_ptr);
        }
    }

    //update the leaf_nodes
    for (i = 0; i < tree_list.size(); i++) {
        tree_list[i].changeLeaf();
    }

    return 1;
}

int MHT_tracker::sovle_mwis (Graph graph, std::map<int, std::vector<int>>& path) {
// choses an MWIS method
// the first one is MWIS by greedy algorithm
// the seconde one is traditional MWIS

    graph.mwis_greed(path);
    //graph.mwis(path);
}

/*int MHT_tracker::pruning(std::map<int, std::vector<int>> path){
    
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
}*/

int MHT_tracker::pruning(std::map<int, std::vector<int>> path){
// pruns the unselected routes
    for (int i = 0; i < tree_list.size();i++) {
        if (path.count(tree_list[i].getId())) {  //if confirmed
            tree_list[i].pruning(path[tree_list[i].getId()]);

            int count_path_zero=0;
            for (int j = 1; j < path[tree_list[i].getId()].size(); j++) {
                if(path[tree_list[i].getId()][j] == 0) {
                    count_path_zero++;
                }
            }
            
            if((count_path_zero == N-1) && 
               (path[tree_list[i].getId()][0] != 0)) {
                tree_list[i].miss_times = tree_list[i].miss_times+ N-1;
            } else if ((count_path_zero == N-1) && 
                       (path[tree_list[i].getId()][0] == 0)) {
                tree_list[i].miss_times += 1;
            } else {  //confirmes and gates
                tree_list[i].miss_times = 0;
            }
        }
        else { //inconfirmed
            tree_list[i].createICH();
            tree_list[i].miss_times += 1;
        }
    std::cout << "tree" << tree_list[i].getId() << " miss time:"
                << tree_list[i].miss_times << std::endl;
    }
}

// DEPRECATED
/*int MHT_tracker::sentResult(byavs::TrackeObjectCPUs& results){

    
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

}*/

int MHT_tracker::sentResult (byavs::TrackeObjectCPUs& results) {
// confirms tracking results
    for (int i = 0; i < tree_list.size();) {
        //if the id is not missed
        if (tree_list[i].miss_times < miss_time_thrd) {
            cv::Rect_<float> bbox;
            byavs::BboxInfo box;
            byavs::TrackeObjectCPU trk_obj_cpu;
            
            tree_list[i].sentResult(bbox);
            trk_obj_cpu.label = tree_list[i].getLabel();
            trk_obj_cpu.id = tree_list[i].getId();
            trk_obj_cpu.box = {(int)bbox.x, (int)bbox.y, 
                               (int)bbox.width, (int)bbox.height};
            results.push_back(trk_obj_cpu);
            i++; 
        } else {
            tree_list.erase(tree_list.begin()+i);
        }
    }
    for (int i = 0; i < results.size(); i++) {
        if(results[i].box.width == 0 && results[i].box.height == 0) {
            results.erase(results.begin()+i);
            i--;
        }
    }
}

int MHT_tracker::backTraversal (treeNode tree_node, 
                        std::shared_ptr<treeNode> head_node, 
                        std::vector<int>& path, 
                        std::vector<float>& path_score, 
                        std::vector<std::vector<int>>& path_list, 
                        std::vector<std::vector<float>>& path_score_list,int N) {
    path.push_back(tree_node.index);
    path_score.push_back(tree_node.score); 
    // When the depth of the tree is not big than N
    if (tree_node.parent == head_node) {
        path.push_back(tree_node.parent->index);
        path_score.push_back(tree_node.parent->score);
        if (path.size() < N) {
            for (int i = N-path.size(); i > 0; i--) {
                path.push_back(0);
                path_score.push_back(0.0);
            }
        }
        path_list.push_back(path);
        path_score_list.push_back(path_score);
        return 1;
    }
    //When this node is a root node
    if (tree_node.parent == NULL &&  path.size() < N) {
        for (int i = N-path.size(); i > 0; i--) {
            path.push_back(0);
            path_score.push_back(0.0);
        }
        path_list.push_back(path);
        path_score_list.push_back(path_score);
        return 1;
    }
    if (tree_node.parent != NULL) {
        backTraversal(*(tree_node.parent), head_node, path, path_score, path_list, 
                        path_score_list, N);
    }
}

int MHT_tracker::TreeToGraph (Graph& graph) {
// transters tree to graph
    std::vector<int> path;
    std::vector<float> path_score;
    std::vector<std::vector<int>> path_list;
    std::vector<std::vector<float>> path_score_list;
    std::vector<VexNode> graph_node_list;
    
    for (auto tree : tree_list) {
        //preorderTraversal(tree.getHead(),path, path_list);
        for (auto leaf : tree.getLeafNode()){
            path.clear();
            path_score.clear();
            backTraversal(*(leaf), tree.getHead(), path, path_score, path_list, 
                            path_score_list,tree.getN());
        }

        if ( path_list.size() > 100) { //selects the top 100 routes
            std::vector<VexNode> temp_node_list;
            for (int i = 0; i < path_list.size(); i++) {
                VexNode graph_node;
                graph_node.path.clear();
                graph_node.score = 0;
                for (int j = path_list[i].size()-1; j >=0; j--) {
                    graph_node.id = tree.getId();
                    graph_node.score += path_score_list[i][j];
                    graph_node.path.push_back(path_list[i][j]);
                }
                temp_node_list.push_back(graph_node);
            }
            std::sort(temp_node_list.begin(), temp_node_list.end(), VexSort);
            for (int i = 0; i < 100; i++) {
                graph_node_list.push_back(temp_node_list[i]);
            }
        } else {
            for(int i = 0; i < path_list.size(); i++) {
                 VexNode graph_node;
                 graph_node.path.clear();
                 graph_node.score = 0;
                 for (int j = path_list[i].size()-1; j >= 0; j--) {
                     graph_node.id = tree.getId();
                     graph_node.score += path_score_list[i][j];
                     graph_node.path.push_back(path_list[i][j]);
                 }
                 graph_node_list.push_back(graph_node);
             }
        }
         path_list.clear();
         path_score_list.clear();
    }
    
    //std::sort(graph_node_list.begin(), graph_node_list.end(), VexSortUp);

    graph = Graph(graph_node_list);

    for (int i=0; i<graph_node_list.size(); i++) {
        std::cout << "index:" << i << " Tree Id:" 
                  << graph_node_list[i].id << " ";
        for (int j = 0; j < graph_node_list[i].path.size(); j++) {
            std::cout << graph_node_list[i].path[j] << " ";
        }
        std::cout<<graph_node_list[i].score;
        std::cout<<std::endl;
    }
}

std::vector<std::vector<double>> MHT_tracker::computeDistance (
                                        std::vector<cv::Rect_<float>> det_result0, 
                                        std::vector<cv::Rect_<float>> det_result) {
//computes the cost_matrix of Hunguarian Algorithm
    int det_num0 = det_result0.size();
    int det_num = det_result.size();
    //IOU treshold in NMS, the pair with iou>ov_thre will be deleted
    float ov_thre = 0.35;  
    std::vector<std::vector<double>> cost_matrix;
    cost_matrix.clear();
    cost_matrix.resize(det_num0, std::vector<double>(det_num, 1));

    for (int i = 0; i < det_num0; i++) {
        for (int j = 0; j < det_num; j++) {
            double iou = get_iou(det_result0[i], det_result[j]);
            if (iou > ov_thre){
                //exclude self matching and intersection
                if (i == j) {  
                    cost_matrix[i][j] = 1;
                } else {
                    cost_matrix[i][j] = 1-iou;
                }
            } else {
                cost_matrix[i][j] = 1;
            }
        }
    }

    return cost_matrix;
}
 
std::vector<Tree> MHT_tracker::get_tree_list() {
    return tree_list;
}