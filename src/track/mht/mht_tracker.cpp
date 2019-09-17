#include "mht_tracker.h"

int MHT_tracker::inference(std::vector<cv::Rect_<float>> det_result, std::vector<float> det_result_score,byavs::TrackeObjectCPUs& results, byavs::TrackeObjectCPUs& predicting_results){

    Graph graph;
    std::map<int, std::vector<int>> path;

    if(det_result.size()!=0)
    {
        std::cout<<"NMS..."<<std::endl;
        det_result = NMS( det_result, det_result_score);
        std::cout<<"after NMS, det_result size:"<<det_result.size()<<std::endl;
        // for(int i=0; i<det_result.size(); i++)
        // {
        //     std::cout<<i+1<<" "<<det_result[i]<<std::endl;
        // }
    }

    //std::cout<<"gating..."<<std::endl;
    gating(det_result, predicting_results);
    //std::cout<<"after gating, tree_list size:"<<tree_list.size()<<std::endl;
    // for(int i=0; i<tree_list.size(); i++)
    // {
    //     std::cout<<"tree "<<i+1<<" head node box: "<<tree_list[i].getHead()->box<<std::endl;
    // }

    //std::cout<<"TreeToGraph..."<<std::endl;
    TreeToGraph(graph);
    //std::cout<<"after TreeToGraph, tree_list size:"<<tree_list.size()<<std::endl;

    //std::cout<<"sovle_mwis..."<<std::endl;
    sovle_mwis(graph, path);
    //std::cout<<"after sovle_mwis, tree_list size:"<<tree_list.size()<<std::endl;

    //std::cout<<"sentResult..."<<std::endl;
    sentResult(results);
    //std::cout<<"after sentResult, results size:"<<results.size()<<std::endl;

    //std::cout<<"pruning..."<<std::endl;
    pruning(path);
    // for(int i=0;i<tree_list.size();i++)
    // {
    //     std::cout<<"tree No."<<tree_list[i].getId()<<" hit_times:"<<tree_list[i].hit_times<<" miss_times:"<<tree_list[i].miss_times<<std::endl;
    // }///

    

    
}

std::vector<cv::Rect_<float>> MHT_tracker::NMS(std::vector<cv::Rect_<float>> det_result, std::vector<float> det_result_score)
{
    float score_diff = 2;

    /*delete intersection*/
    double ov_n1, ov_n2;
    for(int i=0; i<det_result.size(); i++)
    {
        for (int j=i+1; j<det_result.size(); j++)
        {
                ov_n1 = get_ov_n1(det_result[i],det_result[j]);
                ov_n2 = get_ov_n2(det_result[i],det_result[j]);
                if(ov_n1 ==1)
                {
                    det_result.erase(det_result.begin()+j);
                    j--;
                }
                if(ov_n2 ==1)
                {
                    det_result.erase(det_result.begin()+i);
                    i--;
                    break;
                }
                
        }
    }
    
    HungarianAlgorithm HungAlgo;
    std::vector<std::vector<double>> cost_matrix = computeDistance(det_result, det_result);
    //std::cout<<"size:"<<cost_matrix.size()<<std::endl;
    std::vector<int> assign;
    assign.clear();
    HungAlgo.Solve(cost_matrix, assign);
    
    for(int i=0; i<assign.size(); i++)
    {
        //std::cout<<"match "<<i<<" "<<assign[i]<<std::endl;
        if(cost_matrix[i][assign[i]]==1)//self maching, and iou<0.4
        {
            assign[i]=-1;
        }
        
    }
    
    /*set overlap box with lower score to zero box*/
    std::vector<cv::Rect_<float>> detection;
    //cv::Rect_<float> zero_det_box = cv::Rect(0,0,0,0);
    for(int i=0; i<assign.size(); i ++)
    {
        if(assign[i]==-1)//
        {
            detection.push_back(det_result[i]);
        }
        else
        {
            //std::cout<<i<<"("<<det_result_score[i]<<")"<<assign[i]<<"("<<det_result_score[assign[i]]<<")"<<std::endl;
            if(det_result_score[i]-det_result_score[assign[i]]>=score_diff)
            {
                detection.push_back(det_result[i]);
            }
            else if(det_result_score[i]-det_result_score[assign[i]]<=-score_diff)
            {
                //detection.push_back(zero_det_box);
                continue;
            }
            else
            {
                detection.push_back(det_result[i]);
            }
        }
        
    }
    return  detection;
}

int MHT_tracker::gating(std::vector<cv::Rect_<float>> det_result, byavs::TrackeObjectCPUs& predicting_results)
{
    
    int i, j;
    float x1, y1, x2, y2, distance ;
    float threshold = 40;//threshold of the distance,changeable
    float iou_thre = 0.4*exp(-40); //threshold of IOU score
    float maxScaleDiff = 1.4;
    float xx1, yy1, xx2, yy2, w, h, IOU;//IOU is the score
    //double iou; 
    float zero = 0;
    bool success_flag; 
    std::vector<Tree> new_tree_list;
    new_tree_list.clear();
    //cv::Rect_<float> zero_det_box = cv::Rect(0,0,0,0);
    byavs::TrackeObjectCPU track_predict;///
    byavs::BboxInfo track_predict_box;
    predicting_results.clear();///
    
    //push the leaf_node of the trees into a vector
    std::vector<std::shared_ptr<treeNode>> leaf_node_list;
    std::vector<cv::Rect_<float>> leaf_node_predict_list;
    leaf_node_list.clear();
    leaf_node_predict_list.clear();
    for(i=0; i<tree_list.size(); i++)
    {
        for(j=0; j<tree_list[i].getLeafNode().size(); j++)
        {
            leaf_node_list.push_back(tree_list[i].getLeafNode()[j]);///
            //std::cout<<"Tree "<<tree_list[i].getId()<<" leaf node list size:"<<leaf_node_list.size()<<std::endl;
        }
    }
    // for(i=0; i<leaf_node_list.size(); i++)
    // {
    //     KalmanTracker temp_kalman = leaf_node_list[i]->kalman_tracker;///
    //     temp_kalman.predict();///
    //     cv::Rect_<float> predict_box = temp_kalman.getBbox();///
    //     leaf_node_predict_list.push_back(predict_box);
    //     track_predict_box.topLeftX = predict_box.x;///
    //     track_predict_box.topLeftY = predict_box.y;///
    //     track_predict_box.width = predict_box.width;///
    //     track_predict_box.height = predict_box.height;///
    //     track_predict.box = track_predict_box;///
    //     track_predict.id = 0;///
    //     predicting_results.push_back(track_predict);///
    // }

    /*HungarianAlgorithm*/
    /*if(leaf_node_list.size()>0){
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix = computeDistance(leaf_node_list, det_result);
        std::vector<int> assign;
        assign.clear();
        HungAlgo.Solve(cost_matrix, assign);
    }*/
    
    /*match the box to leaf_nodes in the leaf_node_list, the leaf_node_list has no zero_nodes*/
    for(i=0; i<det_result.size(); i++)
    {
        
        ///if(det_result[i]!=zero_det_box)
        ///{
            success_flag = false;
            ///std::cout<<i+1<<std::endl;///
            //caculate the central coordinate of the det_result;
            x1 = det_result[i].x + det_result[i].width/2;
            y1 = det_result[i].y + det_result[i].height/2;
            //std::cout<<"x1:"<<x1<<"y1:"<<y1<<std::endl;
            
            for(j=0; j<leaf_node_list.size(); j++)
            {
                /*caculate the Euclidean distance*/
                x2 = leaf_node_list[j]->box.x + leaf_node_list[j]->box.width/2;
                y2 = leaf_node_list[j]->box.y + leaf_node_list[j]->box.height/2;
                // cv::KalmanFilter k_filter(1, 1, 0, cv::CV_32F );;
                // cv::CV_PROP_RW cov = k_filter.errorCovPre;
                // cv::CV_PROP_RW cov_inverse;
                // cv::invert(cov, cov_inverse, cv::DECOMP_LU);
                
                //KalmanTracker temp_kalman = leaf_node_list[j]->kalman_tracker;///
                //std::cout<<leaf_node_list[j]->box<<std::endl;
                //temp_kalman.predict();///
                //cv::Mat P = temp_kalman.getKalmanFilter().errorCovPre;
                //cv::Rect_<float> predict_box = temp_kalman.getBbox();///
                //x2 = predict_box.x + predict_box.width/2;
                //y2 = predict_box.y + predict_box.height/2;
                //double cov = P.at<float>(0,0);
                //std::cout<<"cov: "<<cov<<std::endl;//P(cv::Range(0,1),cv::Range(0,1))
                //std::cout<<"erroCovPre: "<<P<<std::endl;

                //x2 = leaf_node_predict_list[j].x + leaf_node_predict_list[j].width/2;
                //y2 = leaf_node_predict_list[j].y + leaf_node_predict_list[j].height/2;
                
                //std::cout<<"index : " << j+1 << " box :" << leaf_node_predict_list[j] <<std::endl;
                
                distance = sqrt(pow(x1-x2,2)+pow(y1-y2,2));
                //std::cout<<"distance0:"<<distance<<std::endl;
                //distance = sqrt((pow(x1-x2,2)+pow(y1-y2,2))/cov);
                

               
                //std::cout<<"x2:"<<x2<<"y2:"<<y2<<std::endl;
                //std::cout<<"Detect index :"<< i+1 << " Leaf Node Index : "<< leaf_node_list[j]->index <<"  distance:"<<distance<<std::endl;

                //caculate the score, which is IOU here
                /*xx1 = std::max(det_result[i].x, leaf_node_list[j]->box.x);
                yy1 = std::max(det_result[i].y, leaf_node_list[j]->box.y);
                xx2 = std::min(det_result[i].x + det_result[i].width,leaf_node_list[j]->box.x + leaf_node_list[j]->box.width);
                yy2 = std::min(det_result[i].y+det_result[i].height,leaf_node_list[j]->box.y + leaf_node_list[j]->box.height);
                w = std::max(zero, xx2-xx1);
                h = std::max(zero, yy2-yy1);
                IOU = w*h/(det_result[i].width*det_result[i].height+leaf_node_list[j]->box.width*leaf_node_list[j]->box.height-w*h);*/

                IOU = get_iou(det_result[i], leaf_node_list[j]->box);
                //IOU = get_iou(det_result[i], predict_box);
                
                //cv::Rect_<float> predict_box_post = cv::Rect(x2-det_result[i].width/2, y2-det_result[i].height/2, det_result[i].width, det_result[i].height);
                //IOU = get_iou(det_result[i], predict_box_post);
                
                //std::cout<<"IOU:"<<IOU<<std::endl;
                /*if(i==42)
                {
                    std::cout<<"Detect index :"<< i+1 << " Leaf Node Index : "<< leaf_node_list[j]->index <<"  distance:"<<distance<<" IOU:"<<IOU<<std::endl;
                }*/

                //std::cout<<"Detect index :"<< i+1 << " Leaf_node "<< j <<"  distance:"<<distance<<" IOU:"<<IOU<<std::endl;
 
                //if(IOU*exp(-distance)> 0.6*exp(-2*sqrt(3)))//&& distance < threshold
                if(IOU*exp(-distance) > iou_thre )
                {
                    //                          std::cout<<"Detect index :"<< i+1 <<" det_result height "<<det_result[i].height<<" predict height "<<predict_box.height<<std::endl;
                    ////if(std::max(det_result[i].height/leaf_node_predict_list[j].height, leaf_node_predict_list[j].height/det_result[i].height) <= maxScaleDiff)
                    if(std::max(det_result[i].height/leaf_node_list[j]->box.height, leaf_node_list[j]->box.height/det_result[i].height) <= maxScaleDiff)
                    {
                        //std::cout<<"selected predict_box:"<<leaf_node_list[j]->index<<" "<<std::endl;
                        //std::cout<<"Detect index :"<< i+1 << " Leaf_node Index : "<< leaf_node_list[j]->index <<"  distance:"<<distance<<" IOU:"<<IOU<<std::endl;
                        
                        /*inorder to visulize the predict result*/
                        std::cout<<"distance:"<<distance<<std::endl;

                        std::shared_ptr<treeNode> det_node_ptr(new treeNode);
                        det_node_ptr->box = det_result[i];
                        //std::cout<<"det_result[i]:"<<det_result[i]<<std::endl;
                        det_node_ptr->index = i+1;
                        //det_node_ptr->score = IOU*exp(-distance);
                        det_node_ptr->score = IOU;
                        det_node_ptr->level = leaf_node_list[j]->level+1;
                        det_node_ptr->parent = leaf_node_list[j];

                        //det_node_ptr->kalman_tracker = leaf_node_list[j]->kalman_tracker;
                        //det_node_ptr->kalman_tracker.update(det_result[i]);
                        // det_node_ptr->box = det_node_ptr->kalman_tracker.getBbox();
                        ///std::cout<<"update box:"<<i+1<<" "<<det_node_ptr->box<<std::endl;
                        //cv::Rect_<float> update_box = det_node_ptr->kalman_tracker.getBbox();
                        //det_node_ptr->box = cv::Rect(update_box.x+(update_box.width-det_result[i].width)/2, update_box.y+(update_box.height-det_result[i].height)/2,det_result[i].width,det_result[i].height);
                    
                        leaf_node_list[j]->children.push_back(det_node_ptr);
                        success_flag = true;
                    }
                    success_flag = true;
                }
                
            }
            ///std::cout<<std::endl;
            /*for those boxes which do not match any existing trees:create a new tree for them*/
            if(success_flag == false)
            {
                std::shared_ptr<treeNode> det_node_ptr(new treeNode);
                det_node_ptr->box = det_result[i];
                det_node_ptr->index = i+1;
                det_node_ptr->score = 0.01;
                det_node_ptr->level = 1;//initialize the level of each tree/node 1
               //det_node_ptr->kalman_tracker = KalmanTracker(det_result[i], 3);

                Tree gate(det_node_ptr,3,N);//label=3,N=3
                new_tree_list.push_back(gate);
            }
            
        ///}
        
    }

    for(i=0; i<new_tree_list.size(); i++)
    {
        tree_list.push_back(new_tree_list[i]);
    }
        
    /*for those leaf_nodes which are not matched box:add zero_node to them*/
    for(i=0; i<leaf_node_list.size(); i++)
    {
        if(leaf_node_list[i]->children.size() == 0)
        {
            std::shared_ptr<treeNode> zero_node_ptr(new treeNode);
            //zero_node_ptr->kalman_tracker = leaf_node_list[i]->kalman_tracker; 
            zero_node_ptr->index = 0;
            zero_node_ptr->score = 0;
            //zero_node_ptr->score = tree_list[i].getLeafNode()[j]->score;
            zero_node_ptr->box = leaf_node_list[i]->box;
            // zero_node_ptr->kalman_tracker.predict();
            // zero_node_ptr->box = zero_node_ptr->kalman_tracker.getBbox();//the zero node's box is a predict box of the leaf-node
            
            //cv::Rect_<float> zero_node_box = zero_node_ptr->kalman_tracker.getBbox();
            //zero_node_ptr->box = cv::Rect(zero_node_box.x+(zero_node_box.width-leaf_node_list[i]->box.width)/2,zero_node_box.y+(zero_node_box.height-leaf_node_list[i]->box.height)/2,leaf_node_list[i]->box.width,leaf_node_list[i]->box.height);
            ///zero_node_ptr->box = leaf_node_list[i]->box;
            zero_node_ptr->level = leaf_node_list[i]->level+1;
            zero_node_ptr->parent = leaf_node_list[i];
            leaf_node_list[i]->children.push_back(zero_node_ptr);
        }
    }

    for(i=0; i < tree_list.size(); i++)
    {
        tree_list[i].changeLeaf();
        ///tree_list[i].printTree(tree_list[i].getRoot());
        ///std::cout<<std::endl;////
        //std::cout<<"previous miss_times:"<<tree_list[i].miss_times<<" previous hit_times:"<<tree_list[i].hit_times<<std::endl;
        //std::cout<<"Tree "<<tree_list[i].getId()<<" leaf_node quantity:"<<tree_list[i].getLeafNode().size()<<std::endl;
    }

    return 1;
        

    // //match the box to leaf_nodes in the leaf_node_list, the leaf_node_list has no zero_nodes
    // for(i=0; i<det_result.size(); i++)
    // {
    //     //create a node for each det_result
    //     success_flag = false;

    //     //caculate the central coordinate of the det_result;
    //     x1 = det_result[i].x + det_result[i].width/2;
    //     y1 = det_result[i].y + det_result[i].height/2;
    //     //std::cout<<"x1:"<<x1<<"y1:"<<y1<<std::endl;

    //     for(j=0; j<leaf_node_list.size(); j++)
    //     {
    //         //caculate the distance,could be a function 
    //         //here we caculate the Euclidean distance
    //         KalmanTracker temp_kalman = leaf_node_list[j]->kalman_tracker;
    //         temp_kalman.predict();
    //         cv::Rect_<float> predict_box = temp_kalman.getBbox();
            
    //         x2 = predict_box.x + predict_box.width/2;
    //         y2 = predict_box.y + predict_box.height/2;
    //         distance = sqrt(pow(x1-x2,2)+pow(y1-y2,2));

    //         iou = get_iou(det_result[i], predict_box); 

    //         if(IOU>0.5 || distance < threshold)
    //         {
    //             //addNode??
    //             std::shared_ptr<treeNode> det_node_ptr(new treeNode);
    //             det_node_ptr->index = i+1;
    //             det_node_ptr->score = iou;
    //             det_node_ptr->level = leaf_node_list[j]->level+1;
    //             det_node_ptr->parent = leaf_node_list[j];
    //             det_node_ptr->kalman_tracker = leaf_node_list[j]->kalman_tracker;
    //             det_node_ptr->kalman_tracker.update(det_result[i]);
    //             det_node_ptr->box = det_node_ptr->kalman_tracker.getBbox();

    //             leaf_node_list[j]->children.push_back(det_node_ptr);
    //             success_flag = true;
    //             //std::cout<<"Detect index :"<< i+1 << " Leaf Node Index : "<< leaf_node_list[j]->index <<"  distance:"<<distance<<std::endl;
    //         }
    //     }
    //     //for those boxes which do not match any existing trees:create a new tree for them
    //     if(success_flag == false)
    //     {
    //         std::shared_ptr<treeNode> det_node_ptr(new treeNode);
    //         det_node_ptr->box = det_result[i];
    //         det_node_ptr->index = i+1;
    //         det_node_ptr->score = 0.01;
    //         det_node_ptr->level = 1;        //initialize the level of each tree/node 1
    //         det_node_ptr->kalman_tracker = KalmanTracker(det_result[i], 3);
    //         Tree gate(det_node_ptr,3,N);    //label=3,N=3
    //         new_tree_list.push_back(gate);
    //     }
        
    // }
    
    // for(i=0; i<new_tree_list.size(); i++)
    // {
    //     tree_list.push_back(new_tree_list[i]);
    // }
    
    // //for those leaf_nodes which are not matched box:add zero_node to them
    // for(i=0; i<leaf_node_list.size(); i++)
    // {
    //         if(leaf_node_list[i]->children.size() == 0)
    //         {
    //             std::shared_ptr<treeNode> zero_node_ptr(new treeNode);
    //             zero_node_ptr->kalman_tracker = leaf_node_list[i]->kalman_tracker; 
    //             zero_node_ptr->index = 0;
    //             zero_node_ptr->score = 0;
    //             zero_node_ptr->kalman_tracker.predict();
    //             zero_node_ptr->box = zero_node_ptr->kalman_tracker.getBbox();
    //             zero_node_ptr->level = leaf_node_list[i]->level+1;
    //             zero_node_ptr->parent = leaf_node_list[i];
    //             leaf_node_list[i]->children.push_back(zero_node_ptr);
    //         }
    // }

    // for(i=0; i < tree_list.size(); i++){
    //     tree_list[i].changeLeaf();
    //     ///tree_list[i].printTree(tree_list[i].getRoot());
    //     ///std::cout<<std::endl;////
    //     //std::cout<<"previous miss_times:"<<tree_list[i].miss_times<<" previous hit_times:"<<tree_list[i].hit_times<<std::endl;
    //     //std::cout<<"Tree "<<tree_list[i].getId()<<" leaf_node quantity:"<<tree_list[i].getLeafNode().size()<<std::endl;
    // }

    // return 1;
}

int MHT_tracker::sovle_mwis(Graph graph, std::map<int, std::vector<int>>& path){

    graph.mwis(path);
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
    
    
    for(int i=0; i < tree_list.size();i++)
    {
        
        if(path.count(tree_list[i].getId()))//if confirmed
        {
            //std::cout<<tree_list[i].getId()<<" before pruning CTree,head_node children size:"<<tree_list[i].getHead()->children.size()<<std::endl;
            tree_list[i].pruning(path[tree_list[i].getId()]);

            int count_path_zero=0;
            for(int j=1; j <path[tree_list[i].getId()].size(); j++)////
            {
                if(path[tree_list[i].getId()][j]==0)
                {
                    count_path_zero++;
                }
            }
            // for(int j=0; j <path[tree_list[i].getId()].size(); j++)////
            // {
            //     if(path[tree_list[i].getId()][j]==0)
            //     {
            //         count_path_zero++;
            //     }
            // }
            
            if((count_path_zero==N-1) && (path[tree_list[i].getId()][0]!=0))///17 0 0 0 0 0 0 0 0 0 0:indicate it has not been matched for N-1 times
            {
                tree_list[i].miss_times=tree_list[i].miss_times+ N-1;
            }
            
            // if(count_path_zero==N)
            // {
            //     tree_list[i].miss_times=tree_list[i].miss_times+1;
            // }

            else if((count_path_zero==N-1)&& (path[tree_list[i].getId()][0]==0))///0 0 0 0 0 0 0 0 0 0 0 
            {
                tree_list[i].miss_times += 1;
            }

            else ///confirmed and gated
            {
                //tree_list[i].pruning(path[tree_list[i].getId()]);
                tree_list[i].miss_times = 0;
            }
            //tree_list[i].pruning(path[tree_list[i].getId()]);
            //tree_list[i].miss_times = 0;
            ///tree_list[i].hit_times += 1;
        }
        else//inconfirmed
        {
            tree_list[i].createICH();
            tree_list[i].miss_times += 1;
            ///tree_list[i].hit_times = 0;
            ///std::cout<<"after createing ICH, tree "<<tree_list[i].getId()<<": head_node index "<<tree_list[i].getHead()->index<<" head_node children size "<<tree_list[i].getHead()->children.size()<<std::endl;
        }
    //std::cout<<"tree"<<tree_list[i].getId()<<" miss time:"<<tree_list[i].miss_times<<std::endl;
    }
}

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

int MHT_tracker::sentResult(byavs::TrackeObjectCPUs& results){

    
    for(int i=0; i < tree_list.size();){
        if(tree_list[i].miss_times < miss_time_thrd){
            /*if(tree_list[i].hit_times > hit_time_thrd){*/
                cv::Rect_<float> bbox;
                byavs::BboxInfo box;
                byavs::TrackeObjectCPU trk_obj_cpu;
                //std::cout<<"Tree "<<tree_list[i].getId()<<": ";
                tree_list[i].sentResult(bbox);
                //std::cout<<i+1<<" "<<bbox<<std::endl;
                trk_obj_cpu.label = tree_list[i].getLabel();
                trk_obj_cpu.id = tree_list[i].getId();
                trk_obj_cpu.box = {(int)bbox.x, (int)bbox.y, (int)bbox.width, (int)bbox.height};
                results.push_back(trk_obj_cpu);
                // std::cout<<"The result of Object ID "<<trk_obj_cpu.id <<" is sent" << " ";
                // std::cout<<"head_node->level:"<<tree_list[i].getHead()->level<<" leaf_node->level:"<<tree_list[i].getLeafNode()[0]->level<<" ";///
                // std::cout<<"Box is "<<bbox<< std::endl;
                // std::cout<<std::endl;
            ///}
            i++; 
        }else{
            tree_list.erase(tree_list.begin()+i);
        }
        
    }
    for(int i=0; i<results.size(); i++)
    {
        if(results[i].box.width==0 && results[i].box.height==0)
        {
            results.erase(results.begin()+i);
            i--;
        }
    }

    // for(int i=0; i<results.size();i++)
    // {
    //     std::cout<<results[i].box.topLeftX<<" "<<results[i].box.topLeftY<<" "<<results[i].box.width<<" "<<results[i].box.height<<std::endl;
    // }

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
        if(path.size() < N){
            for(int i=N-path.size();i>0;i--){
                path.push_back(0);
                path_score.push_back(0.0);
            }
        }
        path_list.push_back(path);
        path_score_list.push_back(path_score);
        return 1;
    }
   /*When this node is a root node*/
    if(tree_node.parent == NULL &&  path.size() < N){
        
        for(int i=N-path.size(); i > 0; i--){
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
        
        //preorderTraversal(tree.getHead(),path, path_list);
        for(auto leaf : tree.getLeafNode()){
            path.clear();
            path_score.clear();
            backTraversal(*(leaf), tree.getHead(), path, path_score, path_list, path_score_list,tree.getN());
        }
        ///std::cout<<"Tree No."<<tree.getId()<<" Path.size :"<<path_list.size()<<std::endl;
        if(path_list.size() > 100){
            std::vector<VexNode> temp_node_list;
            for(int i=0; i < path_list.size(); i++){
                VexNode graph_node;
                graph_node.path.clear();
                graph_node.score = 0;
                for(int j = path_list[i].size()-1; j >=0; j--){
                    graph_node.id = tree.getId();
                    graph_node.score += path_score_list[i][j];
                    graph_node.path.push_back(path_list[i][j]);
                }
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
                     graph_node.id = tree.getId();
                     graph_node.score += path_score_list[i][j];
                     graph_node.path.push_back(path_list[i][j]);
                 }
                 graph_node_list.push_back(graph_node);
             }
        }
        // for(int i=0; i < path_list.size(); i++){
        //     VexNode graph_node;
        //     graph_node.path.clear();
        //     graph_node.score = 0;
        //     for(int j = path_list[i].size()-1; j >=0; j--){
        //         std::cout<<path_list[i][j]<<" ("<<path_score_list[i][j]<<") "<<" ";
        //         graph_node.id = tree.getId();
        //         graph_node.score += path_score_list[i][j];
        //         graph_node.path.push_back(path_list[i][j]);
        //     }
        //     std::cout<<graph_node.score;
        //     std::cout<<std::endl;
        //     graph_node_list.push_back(graph_node);
        //  }
         path_list.clear();
         path_score_list.clear();
    }
    
    //std::sort(graph_node_list.begin(), graph_node_list.end(), VexSortUp);

    graph = Graph(graph_node_list);

    for(int i=0; i<graph_node_list.size(); i++)
    {
        std::cout<<"index:"<<i << " Tree Id:"<<graph_node_list[i].id<<" ";
        for(int j = 0; j < graph_node_list[i].path.size(); j++){
            std::cout<<graph_node_list[i].path[j]<<" ";
        }
        std::cout<<graph_node_list[i].score;
        std::cout<<std::endl;
    }
}

std::vector<std::vector<double>> MHT_tracker::computeDistance(std::vector<cv::Rect_<float>> det_result0, std::vector<cv::Rect_<float>> det_result){
            
    //int trk_num = leaf_node_list.size();
    int det_num0 = det_result0.size();
    int det_num = det_result.size();
    float ov_thre = 0.35;
    std::vector<std::vector<double>> cost_matrix;
    cost_matrix.clear();
    cost_matrix.resize(det_num0, std::vector<double>(det_num, 1));

    for(int i=0; i < det_num0; i++){
        for(int j=0; j < det_num; j++){
            double iou = get_iou(det_result0[i], det_result[j]);
            if(iou > ov_thre){
                if(i==j)//exclude self matching and intersection
                {
                    cost_matrix[i][j] = 1;
                }
                else
                {
                    cost_matrix[i][j] = 1-iou;
                }
                //std::cout<<i+1<<" "<<j+1<<" iou "<<iou<<std::endl;
            }else{
                cost_matrix[i][j] = 1;
            }
        }
    }

    return cost_matrix;

}
 
