#include "sort_tracker.h"

int SORT_tracker::inference(byavs::GpuImg image){

    std::vector<int> activated_trackers;
    std::vector<int> refind_trackers;
    std::vector<int> lost_trackers;
    std::vector<int> removed_trackers;

    std::vector<cv::Rect_<float>> detection_boxes
    detection_boxes.clear();

    /*
     * Step 1 : join the activate tracker and lost trackers
     *          generate the candidate tracker
     * */
    std::vector<int> unconfirmed_trackers;
    std::vector<int> tracked_trackers;
    unconfirmed_trackers.clear();                                   //new Tracker
    tracked_trackers.clear();
    for(auto i : m_tracked_trackers){
        if(m_trackers[i].IsActivate()){
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

    if(USE_DEEP){
        // deep feature match
        std::vector<std::vector<float>> det_feature;
        compute_detection_feature(image, det_feature);

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
    if(USE_DEEP){
        for(int i=0; i < um_tracker.size(); i++){
            if(m_trackers[candidate[um_tracker[i]]].getState() == STATE_TRACKED){
                remain_tracker.push_back(candidate[um_tracker[i]]);
            }
        }

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
   *  Step 6 : Update states
   * */
   for(int i=0; i < lost_trackers.size(); i++){
       if(m_trackers[lost_trackers[i]].getBeginFrame() - currect_frame > m_miss_time){
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
   deal_reactivate_tracker(m_lost_tracker, m_tracked_trackers);

   for(auto i : lost_trackers){
       m_lost_trackers.push_back(i);
   }

   //remove the need to delete tracker from `m_lost_tracker`
   deal_remove_tracker(m_lost_tracker, removed_trackers);

   //delete the tracker that need to remove
   for(int i=0; i < removed_trackers.size(); i++){
       //remove
   }

   /**
    * Step 8 : output
   */

   //add new tracker
   for(int i=0; i < new_tracker_list.size(); i++){
       m_trackers.push_back(new_tracker_list[i]);
   }

   for(int i=0; i < m_trackers.size(); i++){
       if(m_trackers[i].getState() == STATE_TRACKED){
           //output
       }
   }
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