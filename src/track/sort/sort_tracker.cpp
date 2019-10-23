#include "sort_tracker.h"

int SORT_tracker::inference(std::vector<cv::Rect_<float>> det_boxes, std::vector<float> detection_score, 
            byavs::TrackeObjectCPUs& result){

    std::vector<int> activated_trackers;
    std::vector<int> refind_trackers;
    std::vector<int> lost_trackers;
    std::vector<int> removed_trackers;

    std::vector<cv::Rect_<float>> detection_boxes
    detection_boxes.clear();
    detection_boxes = det_boxes;
    /*
     * Step 1 : join the activate tracker and lost trackers
     *          generate the candidate tracker
     * */
    std::vector<int> unconfirmed_trackers;
    std::vector<int> tracked_trackers;
    unconfirmed_trackers.clear();                                   //new Tracker
    tracked_trackers.clear();
    for(auto i : m_tracked_trackers){
        if(m_trackers[i].isActivate()){
            tracked_trackers.push_back(i);
        }else{
            unconfirmed_trackers.push_back(i);
        }
    }
    
    //`candidate_tracakers` save the indexs of the `m_trackers`
    std::vector<int> candidate_trackers;
    candidate_trackers.clear();

    generate_candidate_trackers(candidate_trackers, tracked_trackers, m_lost_trackers);

    /*
     * Step 2 : tracker predict
     *
     * */
    for(int i=0; i < candidate_trackers.size(); i++){
        m_trackers[candidate[i]].predict();
    }

    /*
     * Step 3 : if using deep feature, compute the feature distacne
     *          otherwise only iou cost will be compute
     * */
    std::map<int, int> matches;
    std::vector<int> um_tracker;
    std::vector<int> um_detection;

    matches.clear();
    um_tracker.clear();
    um_detection.clear();

    if(sort::USE_DEEP){
        // deep feature match
        cv::Mat image;  // TODO: Get the image from DetectionObject
        std::vector<std::vector<float>> det_feature;
        compute_detection_feature(image, detection_boxes, det_feature);

        std::vector<std::vector<float>> feature_cost_matrix;
        feature_cost_matrix.clear();

        compute_feature_distance(feature_cost_matrix, candidate, det_feature);
        matching(feature_cost_matrix, matches, um_tracker, um_detection);
         
        for(int i=0; i < matches.size(); i++){
            if(m_tracker[candidate[matches[i].first]].getState() == STATE_TRACKED){
                m_trackers[candidate[matches[i].first]].update(detection_boxs[matches[i].second]);
                activated_trackers.push_back(candidate[matches[i].first]);
            }else{
                m_trackers[candidate[matches[i].first]].re_activate(detection_boxs[matches[i].second]);
                refind_trackers.push_back(candidate[matches[i].first]);
            }
        }

    }
    
    // iou match
    std::vector<int> remain_tracker;
    std::vector<cv::Rect_<float>> remain_boxes;
    remain_tracker.clear();
    remain_boxes.clear();
    if(sort::USE_DEEP){
        for(int i=0; i < um_tracker.size(); i++){
            if(m_trackers[candidate[um_tracker[i]]].getState() == STATE_TRACKED){
                remain_tracker.push_back(candidate[um_tracker[i]]);
            }
        }
        //TODO: Only using the miss time is equal to 1, others mark as losted
        for(int i=0; i < um_detection.size(); i++){
            remain_boxes.push_back(detection_boxes[um_detection[i]]);
        }
    }else{
        remain_tracker = candidate;
        remain_boxes = detection_boxes;
    }

    std::vector<std::vector<double>> iou_cost_matrix;
    iou_cost_matrix.clear();
    
    matches.clear();
    um_tracker.clear();
    um_detecion.clear();

    compute_iou_distance(iou_cost_matrix, remain_tracker, remain_boxes);
    matching(iou_cost_matrix, matches, um_tracker, um_detecion);

    for(int i=0; i < matches.size(); i++){
        if(m_trackers[remain_tracker[matches[i].first].getState() == STATE_TRACKED){
            m_trackers[remain_tracker[matches[i].first]].update(remain_boxes[matches[i].second]);
            activated_trackers.push_back(remain_tracker[matches[i].first]);
        }else{
            m_trackers[remain_tracker[matches[i].first]].re_activate(remain_boxes[matches[i].first]);
            refind_trackers.push_back(remain_tracker[matches[i].first]);
        }
    }

    for(int i=0; i < um_tracker.size(); i++){

        if(m_trackers[remain_tracker[um_tracker[i]].getState() != STATE_LOST){
            m_trackers[remain_tracker[um_tracker[i]].mark_lost();
            lost_trackers.push_back(remain_tracker[um_tracker[i]]);
        }
    }

    /*
    *  Step 4 : deal with unconfirmed tracker 
    */
    remain_boxes.clear();
    iou_cost_matrix.clear();

    for(int i=0; i < um_detection.size(); i++){
        remain_boxes.push_back(detection_boxes[um_detection[i]]);
    }

    matches.clear();
    um_tracker.clear();
    um_detecion.clear();

    compute_iou_distance(iou_cost_matrix, unconfirmed_trackers, remain_boxes);
    matching(iou_cost_matrix, matches, um_tracker, um_detection);

    for(int i=0; i < matches.size(); i++){
        m_trackers[unconfirmed_trackers[matches[i].first]].update(remain_boxes.push_back(matches[i].second));
        activated_trackers.push_back(unconfirmed_trackers[matches[i].first]);
    }

    for(int i=0; i < um_tracker.size(); i++){
        m_trackers[unconfrimed_trackers[um_tracker[i]]].mark_removed();
        removed_trackers.push_back(unconfrimed_trackers[um_tracker[i]]);
    }

    /*
    *  Step 5 : Init new trackers
    */
   std::vector<Tracker> new_tracker_list;
   new_tracker_list.clear();
   for(int i=0; i < um_detection.size(); i++){
       Tracker new_tracker = new Tracker(remain_boxes[um_detection[i]]);
       new_tracker_list.push_back(new_tracker);
   }

   /*
   *  Step 6 : Mark the removed target when miss time achieve the threshold
   * */
   for(int i=0; i < lost_trackers.size(); i++){
       if(m_trackers[lost_trackers[i]].getMissTime() > sort::MAX_MISS_TIME){
            m_trackers[lost_trackers[i]].mark_removed();
            removed_trackers.push_back(lost_trackers[i]);
       }
   }

   /*
   *  Step 7 : deal the tracker states
   * */
   m_tracked_trackers.clear();
   for(auto i : activated_trackers){
       m_tracked_trackers.push_back(i);
   }

   for(auto i : refind_trackers){
       m_tracked_trackers.push_back(i);
   }

   //remove the re-find tracker from `m_lost_tracker`
   deal_duplicate_tracker(m_lost_tracker, m_tracked_trackers);

   for(auto i : lost_trackers){
       m_lost_trackers.push_back(i);
   }

   //remove the need to delete tracker from `m_lost_tracker`
   deal_duplicate_tracker(m_lost_tracker, removed_trackers);

   /**
    * Step 8 : output
   */

   //output and delete the tracker that need to remove
   for(int i=0; i < m_trackers.size(); i++){
       //output and delete
       if(m_trackers[i].getState() == STATE_REMOVE){
            //TODO: Set the finish tracking flag to True 
            byavs::TrackeObjectCPU remove_object;
            remove_object.camID = 0;
            remove_object.channelID = 0;
            remove_object.id = 0;
            remove_object.label = 0;
            remove_object.box = {m_trackers[i].getBox().x, m_trackers[i].y,
                    m_trackers[i].getBox().width, m_trackers[i].getBox().height};
            remove_object.return_state = true;
            remove_object.match_flag = 0;
            remove_object.score = 0.0;
            result.push_back(remove_object);
            m_trackers.earse(m_tracker.begin()+i);
       }
   }

   //add new tracker
   for(int i=0; i < new_tracker_list.size(); i++){
       m_trackers.push_back(new_tracker_list[i]);
   }

   for(int i=0; i < m_trackers.size(); i++){
       if(m_trackers[i].getState() == STATE_TRACKED){
           //output
            byavs::TrackeObjectCPU output_object;
            output_object.camID = 0;
            output_object.channelID = 0;
            output_object.id = 0;
            output_object.label = 0;
            output_object.box = {m_trackers[i].getBox().x, m_trackers[i].y,
                    m_trackers[i].getBox().width, m_trackers[i].getBox().height};
            output_object.return_state = true;
            output_object.match_flag = 0;
            output_object.score = 0.0;
            result.push_back(output_object);
       }
   }
}

int inference(const std::string& model_dir, const byavs::TrackeParas& pas, 
              const int gpu_id){
    return 1;
}

int SORT_tracker::generate_candidate_trackers(std::vector<int>& candidate, 
                std::vector<int> tracked_tracker, std::vector<int> m_lost_trackers){
    
    candidate.clear();

    for(int i=0; i<tracked_tracker.size();i++){
        int * location_index = find(m_lost_trackers.begin(), m_lost_trackers.end(), tracked_tracker[i]);
        if((location_index - m_lost_trackers.begin()) < m_lost_trackers.size()){
           m_lost_trackers.earse(location_index);
        }
        candidate.push_back(tracked_tracker[i]);
    }

    for(auto i : m_lost_trackers){
        candidate.push_back(i);
    }

}

int SORT_tracker::compute_detection_feature(cv::Mat image, std::vector<cv::Rect_<float>> detection_boxes, 
                std::vector<std::vector<float>>& detection_features){
    //TODO: Using a single ReID model to extract the object feature
}

int SORT_tracker::compute_feature_distance(std::vector<std::vector<double>>& cost_matrix, std::vector<int> traker_index, 
                std::vector<std::vector<float>> detection_features){
    //TODO: compute the distance of features
}



int SORT_tracker::compute_iou_distance(std::vector<std::vector<double>> &cost_matrix, 
                std::vector<int> tracker_index, std::vector<cv::Rect_<float>> detection_boxes){
    
    cost_matrix.clear();
    cost_matrix.reshape(tracker_index.size(), detection_boxes.size());
    for(int i=0; i < tracker_index.size(); i++){
        for(int j=0; j < detection_boxes.size(); j++){
            cost_matrix[i][j] = get_iou(m_trackers[tracker_index[i]].getBox(), detection_boxes[j]);
        }
    }

}

int SORT_tracker::matching(std::vector<std::vector<double>> cost_matrix, std::map<int, int>& matches, 
                std::vector<int>& um_tracker, std::vector<int>& um_detection){
   
    if(cost_matrix.size() == 0)  return 1;

    std::set<int> all_detection;
    std::set<int> matched;
    std::set<int> unmatched_detection;

    all_detection.clear();
    matched_detection.clear();
    unmatched_detection.clear();

    for(int i=0; i < cost_matrix[0].size(); i++){
        all_detection.insert(i);
    }

    std::vector<int> assign;
    assign.clear();

    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(cost_matrix, assign);

    for(int i=0; i < assign.size(); i++){      
        if(assign[i] == -1){
            um_tracker.insert(i);
        }else{
            if(cost_matrix[i][assign[i]] < 0.7){
                matches[i] = assign[i];
                matched_detection.insert(assign[i]);
            }else{
                um_tracker.insert(i);
            }

        }

    }

    if(matched_detection.size() < all_detection.size()){

        std::set_difference(all_detection.begin(), all_detection.end(),
            matched_detection.begin(), matched_detection.end(),
            std::insert_iterator<std::set<int>>(unmatched_detection, unmatched_detection.begin()));

        for(auto i : unmatched_detection){
            um_detection.push_back(i);
        }
    }

}

int deal_duplicate_tracker(std::vector<int>& lost_tracker, std::vector<int> tracker_list){

    for(int i=0; i < lost_tracker.size(); i++){
        int * location_index = find(tracker_list.begin(), tracker_list.end(), lost_tracker[i]);
        if((location_index - tracker_list.begin()) < tracker_list.size()){
           lost_tracker.earse(lost_tracker.begin()+i);
        }
    }

}