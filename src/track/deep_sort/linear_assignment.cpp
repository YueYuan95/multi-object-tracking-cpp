#include "linear_assignment.h"

int LinearAssignment::min_cost_matching(DistanceMetric dm, KalmanTrackerV2 kf, std::vector<cv::Rect_<float>> dt_boxes,
    std::map<int, int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detections,
    std::vector<std::vector<float>> detection_feature, std::vector<int> detection_indexs,
    std::vector<BaseTracker> tracker_list, std::vector<int> tracker_indexs){
    
    // if(tracker_indexs.size() == 0){
    //     for(int i=0; i<tracker_list.size(); i++) tracker_indexs.push_back(i);
    // }
    // if(detection_indexs.size() == 0){
    //     for(int i=0; i<detection_feature.size(); i++) detection_indexs.push_back(i);
    // }

    std::vector<std::vector<double>> cost_matrix;
    cost_matrix.clear();

    std::vector<std::vector<float>> temp_features_list;
    std::vector<int> temp_tracker_ids;

    for(int i=0; i < detection_indexs.size(); i++){
        temp_features_list.push_back(detection_feature[detection_indexs[i]]);
    }
    for(int i=0; i < tracker_indexs.size(); i++){
        temp_tracker_ids.push_back(tracker_list[tracker_indexs[i]].get_id());
    }

    double start, end;
    start = clock();
    dm.distance(cost_matrix, temp_features_list, temp_tracker_ids);
    end = clock();
    debug<<"compute feature distance cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
    debug<<"cost matrix"<<debugend;
    for(int i=0; i < cost_matrix.size(); i++){
        for(int j=0; j < cost_matrix[i].size(); j++){
            std::cout<<cost_matrix[i][j]<<" "; 
        }
        std::cout<<std::endl;
    }

    start = clock();
    gate_cost_matrix(kf, cost_matrix, tracker_list, tracker_indexs, dt_boxes, detection_indexs);
    end = clock();
    debug<<"compute gate_cost_matrix cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
    
    debug<<"after gate cost matrix"<<debugend;
    for(int i=0; i < cost_matrix.size(); i++){
        for(int j=0; j < cost_matrix[i].size(); j++){
            std::cout<<cost_matrix[i][j]<<" "; 
        }
        std::cout<<std::endl;
    }

    
    //Hungari Algorithm
    std::vector<int> assign;
    assign.clear();
    start = clock();
    hung_algo.Solve(cost_matrix, assign);
    end = clock();
    debug<<"hungarian cost time : "<<(double)(end -start)/CLOCKS_PER_SEC * 1000<<" ms" <<debugend;

    //TODO: use set to get unmatched detections
    for(int i=0; i < detection_indexs.size(); i++){
        if(std::find(assign.begin(), assign.end(), i) == assign.end()){
            um_detections.push_back(detection_indexs[i]);
        }
    }

    debug<<"assign :";
    for(auto i : assign) std::cout<<i<<" ";
    std::cout<<std::endl;

    for(int i=0; i < assign.size(); i++){
        if(assign[i] != -1){
            debug<<"cost :"<<cost_matrix[i][assign[i]]<<debugend;
            if(cost_matrix[i][assign[i]] > max_feature_distance){
                um_trackers.push_back(tracker_indexs[i]);
                um_detections.push_back(detection_indexs[assign[i]]);
            }else{
                matches[tracker_indexs[i]] = detection_indexs[assign[i]];
            }          
        }else{
            um_trackers.push_back(tracker_indexs[i]);
        }
    }

    return 1;

}

int LinearAssignment::matching_cascade(DistanceMetric dm, KalmanTrackerV2 kf,
        int cascade_depth, std::vector<cv::Rect_<float>> detection_boxes,
        std::map<int, int>& matches, std::vector<int>& um_detections, std::vector<int>& um_trackers,
        std::vector<BaseTracker> tracker_list, std::vector<std::vector<float>> detection_feature,
        std::vector<int> tracker_index, std::vector<int> detection_index){
    
    matches.clear();

    std::vector<int> detection_candidate;
    std::vector<int> tracker_candidate;
    detection_candidate.clear();
    tracker_candidate.clear();
    
    detection_candidate = detection_index;
    for(int i=0; i < cascade_depth; i++){
        if(detection_candidate.size() == 0) break;

        debug<<" detection candidate :";
        for(auto i : detection_candidate) std::cout<<i<<" ";
        std::cout<<std::endl;

        tracker_candidate.clear();
        for(int j=0; j < tracker_index.size(); j++){
            if(tracker_list[tracker_index[j]].get_time_since_update() == i + 1){
                debug<<"deep : "<<i<<" candidate tracker : "<<tracker_list[tracker_index[j]].get_id()<<debugend;
                tracker_candidate.push_back(tracker_index[j]);
            }
        }
        if(tracker_candidate.size() == 0) continue;

        std::map<int, int> temp_match;
        um_detections.clear();
        min_cost_matching(dm, kf, detection_boxes,
            temp_match, um_trackers, um_detections,
            detection_feature, detection_candidate,
            tracker_list, tracker_candidate);

        debug<<" temp um detection :";
        for(auto i : um_detections) std::cout<<i<<" ";
        std::cout<<std::endl;

        detection_candidate = um_detections;
        for(std::map<int,int>::iterator iter = temp_match.begin(); iter != temp_match.end(); iter++){
            matches[iter->first] = iter->second;
        }
        
    }

    for(int i=0; i < detection_candidate.size(); i++){
        um_detections = detection_candidate;
    }

    um_trackers.clear();
    for(int i=0; i < tracker_index.size(); i++){
        if(!matches.count(tracker_index[i])) um_trackers.push_back(tracker_index[i]);
    }

}

int LinearAssignment::gate_cost_matrix(KalmanTrackerV2 kf, std::vector<std::vector<double>>& cost_matrix,
    std::vector<BaseTracker> tracker_list, std::vector<int> tracker_indexes,
    std::vector<cv::Rect_<float>> detection_boxes, std::vector<int> detection_indexs){
    
    double distance;
    debug<<"gate distance "<<debugend;
    for(int i=0; i < tracker_indexes.size(); i++){
        for(int j=0; j < detection_indexs.size(); j++){
            distance = kf.gating_distance(tracker_list[tracker_indexes[i]].get_mean(), tracker_list[tracker_indexes[i]].get_covariance(),
                detection_boxes[detection_indexs[j]]);
            std::cout<<distance<<" ";
            if(distance > gating_threshold) cost_matrix[i][j] = gating_cost;
        }
        std::cout<<std::endl;
    }

    return 1;
}

int LinearAssignment::matching_by_iou(DistanceMetric dm,
    std::map<int, int>& matches, std::vector<int>& um_detections, std::vector<int>& um_trackers,
    std::vector<cv::Rect_<float>> detect_boxes, std::vector<int> detect_indexs,
    std::vector<BaseTracker> tracker_list, std::vector<int> tracker_indexs){
    
    if(tracker_indexs.size() == 0 || detect_indexs.size() == 0){
       um_detections = detect_indexs;
       um_trackers = tracker_indexs;
       return 1;
    }

    std::vector<std::vector<double>> cost_matrix;
    cost_matrix.clear();

    std::vector<cv::Rect_<float>> temp_boxes;
    std::vector<BaseTracker> temp_track;

    for(int i=0; i < detect_indexs.size(); i++){
        debug<<"detectin_indexs :"<<detect_indexs[i]<<debugend;
        temp_boxes.push_back(detect_boxes[detect_indexs[i]]);
    }
    for(int i=0; i < tracker_indexs.size(); i++){
        debug<<"tracker_indexs :"<<tracker_indexs[i]<<debugend;
        temp_track.push_back(tracker_list[tracker_indexs[i]]);
    }
    
    dm.distance(cost_matrix, temp_boxes, temp_track);
    debug<<"after gate cost matrix"<<debugend;
    for(int i=0; i < cost_matrix.size(); i++){
        for(int j=0; j < cost_matrix[i].size(); j++){
            std::cout<<cost_matrix[i][j]<<" "; 
        }
        std::cout<<std::endl;
    }

    //Hungari Algorithm
    std::vector<int> assign;
    assign.clear();
    hung_algo.Solve(cost_matrix, assign);

    debug<<" assign ";
    for(auto i : assign)  std::cout<< i <<" ";
    std::cout<<std::endl;

    // debug<<" max_distance is "<< max_distance <<debugend;

    for(int i=0; i < detect_indexs.size(); i++){
        if(std::find(assign.begin(), assign.end(), i) == assign.end()){
            um_detections.push_back(detect_indexs[i]);
        }
    }

    for(int i=0; i < assign.size(); i++){
        if(assign[i] != -1){
            if(cost_matrix[i][assign[i]] > max_iou_distance){
                um_trackers.push_back(tracker_indexs[i]);
                um_detections.push_back(detect_indexs[assign[i]]);
            }else{
                matches[tracker_indexs[i]] = detect_indexs[assign[i]];
            }          
        }else{
            um_trackers.push_back(tracker_indexs[i]);
        }
    }

    return 1;

}