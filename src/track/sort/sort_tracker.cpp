#include "sort_tracker.h"

int SortTracker::inference(std::vector<cv::Rect_<float>> det_boxes, std::vector<float> detection_score, 
            byavs::TrackeObjectCPUs& result){

    std::vector<int> activated_trackers;
    std::vector<int> refind_trackers;
    std::vector<int> lost_trackers;
    std::vector<int> removed_trackers;

    std::vector<cv::Rect_<float>> detection_boxes;
    detection_boxes.clear();
    detection_boxes = det_boxes;

    /*
     * Step 1 : tracker predict
     *
     * */

    // for(int i=0; i < m_trackers.size(); i++){
    //      m_trackers[i].predict();
    //
    // }



    /*
     * Step 2 : join the activate tracker and lost trackers
     *          generate the candidate tracker
     * */
    std::vector<int> unconfirmed_trackers;
    std::vector<int> tracked_trackers;
    unconfirmed_trackers.clear();                                   //new Tracker
    tracked_trackers.clear();
    for(int i = 0 ; i < m_trackers.size(); i++){
        if(m_trackers[i].is_activate()){
            tracked_trackers.push_back(i);
        }else{
            unconfirmed_trackers.push_back(i);
        }
    }
    
    //`candidate_tracakers` save the indexs of the `m_trackers`
    std::vector<int> candidate;
    candidate.clear();

    generate_candidate_trackers(candidate, tracked_trackers, m_lost_trackers);

    //std::cout<<m_trackers.size()<<" "<<tracked_trackers.size()<<" "<<unconfirmed_trackers.size()<<std::endl;

    for(int i=0; i < candidate.size(); i++){  
        std::cout<<"Size is "<<candidate.size()<<" Currect is "<<i<<std::endl;  
        std::cout<<m_trackers[candidate[i]].get_box()<<std::endl;
        m_trackers[candidate[i]].predict();
        std::cout<<"tracked tracker predict finished"<<std::endl;
    }
    std::cout<<"step 2 middel"<<std::endl;
    for(int i=0; i < unconfirmed_trackers.size(); i++){
        m_trackers[unconfirmed_trackers[i]].predict();
        std::cout<<"unconfirm tracker predict finished"<<std::endl;
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
        cv::Mat image;  // TODO: Get the image from DetectionObject
        std::vector<std::vector<float>> det_feature;
        compute_detection_feature(image, detection_boxes, det_feature);

        std::vector<std::vector<double>> feature_cost_matrix;
        feature_cost_matrix.clear();

        if(candidate.size() != 0 && det_feature.size() != 0){
            compute_feature_distance(feature_cost_matrix, candidate, det_feature);
            matching(feature_cost_matrix, matches, um_tracker, um_detection);

            for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
                if(m_trackers[candidate[iter->first]].get_state() == STATE_TRACKED){
                    m_trackers[candidate[iter->first]].update(detection_boxes[iter->second]);
                    activated_trackers.push_back(candidate[iter->first]);
                }else{
                    m_trackers[candidate[iter->first]].update(detection_boxes[iter->second]);
                    refind_trackers.push_back(candidate[iter->first]);
                }
            }
        }else{
            for(int i=0; i < candidate.size(); i++){
                um_tracker.push_back(i);
            }
            for(int i=0; i < detection_boxes.size(); i++){
                um_detection.push_back(i);
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
            if(m_trackers[candidate[um_tracker[i]]].get_state() == STATE_TRACKED){
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
    um_detection.clear();

    if(remain_tracker.size() != 0 && remain_boxes.size() != 0){
        compute_iou_distance(iou_cost_matrix, remain_tracker, remain_boxes);
        matching(iou_cost_matrix, matches, um_tracker, um_detection);
        for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
            if(m_trackers[remain_tracker[iter->first]].get_state() == STATE_TRACKED){
                m_trackers[remain_tracker[iter->first]].update(remain_boxes[iter->second]);
                activated_trackers.push_back(remain_tracker[iter->first]);
            }else{
                m_trackers[remain_tracker[iter->first]].update(remain_boxes[iter->second]);
                refind_trackers.push_back(remain_tracker[iter->first]);
            }
        } 
    }else{
        for(int i=0; i < remain_tracker.size(); i++){
            um_tracker.push_back(i);
        }
        for(int i=0; i < remain_boxes.size(); i++){
            um_detection.push_back(i);
        }
    }

    for(int i=0; i < um_tracker.size(); i++){

        if(m_trackers[remain_tracker[um_tracker[i]]].get_state() != STATE_LOST){
            m_trackers[remain_tracker[um_tracker[i]]].mark_lost();
            lost_trackers.push_back(remain_tracker[um_tracker[i]]);
        }
    }

    /*
    *  Step 4 : deal with unconfirmed tracker 
    */
    //std::cout<<"step 4"<<std::endl;
    remain_boxes.clear();
    iou_cost_matrix.clear();

    for(int i=0; i < um_detection.size(); i++){
        remain_boxes.push_back(detection_boxes[um_detection[i]]);
    }

    matches.clear();
    um_tracker.clear();
    um_detection.clear();
    
    if(unconfirmed_trackers.size() !=0 && remain_boxes.size() != 0){
        compute_iou_distance(iou_cost_matrix, unconfirmed_trackers, remain_boxes);
        matching(iou_cost_matrix, matches, um_tracker, um_detection);
        
        for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){       

            m_trackers[unconfirmed_trackers[iter->first]].update(remain_boxes[iter->second]);
            // std::cout<<"Unconfirmed Tracker Id : "<< m_trackers[unconfirmed_trackers[iter->first]].get_id();
            // std::cout<<m_trackers[unconfirmed_trackers[iter->first]].get_box()<<std::endl;
            activated_trackers.push_back(unconfirmed_trackers[iter->first]);
        }
        
    }else{

        for(int i=0; i < unconfirmed_trackers.size(); i++){
            um_tracker.push_back(i);
        }
        for(int i=0; i < remain_boxes.size(); i++){
            um_detection.push_back(i);
        }
    }

    for(int i=0; i < um_tracker.size(); i++){
        m_trackers[unconfirmed_trackers[um_tracker[i]]].mark_removed();
        removed_trackers.push_back(unconfirmed_trackers[um_tracker[i]]);
    }

    /*
    *  Step 5 : Init new trackers
    */
   //std::cout<<"step 5"<<std::endl;
   std::vector<Tracker> new_tracker_list;
   new_tracker_list.clear();
   
   int detection_box_size = um_detection.size();
   if(m_trackers.size() == 0){
       detection_box_size = detection_boxes.size();
       remain_boxes = detection_boxes;
   }

   //debug<<"Tracker size : "<< m_trackers.size() <<" detection_box_size :" << debugend;

   for(int i=0; i < detection_box_size; i++){
       Tracker new_tracker = Tracker(remain_boxes[i], 0);
       new_tracker_list.push_back(new_tracker);
    //    std::cout<<" New Tracker Id : "<< new_tracker.get_id();
    //    std::cout<<" Box : "<<new_tracker.get_box()<<std::endl;
   }

   /*
   *  Step 6 : Mark the removed target when miss time achieve the threshold
   * */
   //std::cout<<"step 6"<<std::endl;
   for(int i=0; i < lost_trackers.size(); i++){
       if(m_trackers[lost_trackers[i]].get_miss_time() > MAX_MISS_TIME){
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
   deal_duplicate_tracker(m_lost_trackers, m_tracked_trackers);

   for(auto i : lost_trackers){
       m_lost_trackers.push_back(i);
   }

   //remove the need to delete tracker from `m_lost_tracker`
   deal_duplicate_tracker(m_lost_trackers, removed_trackers);

   /**
    * Step 8 : output
   */
   
   //output and delete the tracker that need to remove
   for(auto it = m_trackers.begin(); it != m_trackers.end();){
       //output and delete
       
       if((*it).get_state() == STATE_REMOVE){
            //TODO: Set the finish tracking flag to True 
            byavs::TrackeObjectCPU remove_object;
            remove_object.camID = 0;
            remove_object.channelID = 0;
            remove_object.id = it->get_id();
            remove_object.label = 0;
            remove_object.box = {int(it->get_box().x), int(it->get_box().y),
                    int(it->get_box().width), int(it->get_box().height)};
            remove_object.return_state = true;
            remove_object.match_flag = 0;
            remove_object.score = 0.0;
            result.push_back(remove_object);
            it = m_trackers.erase(it);
       }else{
           it++;
       }
   }

   //add new tracker
   for(int i=0; i < new_tracker_list.size(); i++){
       m_trackers.push_back(new_tracker_list[i]);
   }
   if(m_trackers.size() == new_tracker_list.size()){
       for(int i=0; i < m_trackers.size(); i++){
            byavs::TrackeObjectCPU output_object;
            output_object.camID = 0;
            output_object.channelID = 0;
            output_object.id = m_trackers[i].get_id();
            output_object.label = 0;
            output_object.box = {int(m_trackers[i].get_box().x), int(m_trackers[i].get_box().y),
                    int(m_trackers[i].get_box().width), int(m_trackers[i].get_box().height)};
            output_object.return_state = true;
            output_object.match_flag = 0;
            output_object.score = 0.0;
            result.push_back(output_object);
       }
   }
   
   for(int i=0; i < m_trackers.size(); i++){
       
       if(m_trackers[i].get_state() == STATE_TRACKED){
           //output
            //std::cout<<"Id : "<< m_trackers[i].get_id() << " Box :"<< m_trackers[i].get_box()<<std::endl;
            byavs::TrackeObjectCPU output_object;
            output_object.camID = 0;
            output_object.channelID = 0;
            output_object.id = m_trackers[i].get_id();
            output_object.label = 0;
            output_object.box = {int(m_trackers[i].get_box().x), int(m_trackers[i].get_box().y),
                    int(m_trackers[i].get_box().width), int(m_trackers[i].get_box().height)};
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

int SortTracker::generate_candidate_trackers(std::vector<int>& candidate, 
                std::vector<int> tracked_tracker, std::vector<int> m_lost_trackers){
    
    candidate.clear();

    for(int i=0; i<tracked_tracker.size();i++){
        std::vector<int>::iterator location_index = find(m_lost_trackers.begin(), m_lost_trackers.end(), tracked_tracker[i]);
        if((location_index - m_lost_trackers.begin()) < m_lost_trackers.size()){
           m_lost_trackers.erase(location_index);
        }
        candidate.push_back(tracked_tracker[i]);
    }

    for(auto i : m_lost_trackers){
        candidate.push_back(i);
    }

}

int SortTracker::compute_detection_feature(cv::Mat image, std::vector<cv::Rect_<float>> detection_boxes, 
                std::vector<std::vector<float>>& detection_features){
    //TODO: Using a single ReID model to extract the object feature
}

int SortTracker::compute_feature_distance(std::vector<std::vector<double>>& cost_matrix, std::vector<int> traker_index, 
                std::vector<std::vector<float>> detection_features){
    //TODO: compute the distance of features
}



int SortTracker::compute_iou_distance(std::vector<std::vector<double>> &cost_matrix, 
                std::vector<int> tracker_index, std::vector<cv::Rect_<float>> detection_boxes){
    
    cost_matrix.clear();
    cost_matrix.resize(tracker_index.size(), 
            std::vector<double>(detection_boxes.size(), 0.00));
    for(int i=0; i < tracker_index.size(); i++){
        for(int j=0; j < detection_boxes.size(); j++){
            cost_matrix[i][j] = 1 - get_iou(m_trackers[tracker_index[i]].get_box(), detection_boxes[j]);
        }
    }

}

int SortTracker::matching(std::vector<std::vector<double>> cost_matrix, std::map<int, int>& matches, 
                std::vector<int>& um_tracker, std::vector<int>& um_detection){
   
    if(cost_matrix.size() == 0) return 0;

    std::set<int> all_detection;
    std::set<int> matched_detection;
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
            um_tracker.push_back(i);
        }else{
            if(cost_matrix[i][assign[i]] < 0.7){
                matches[i] = assign[i];
                matched_detection.insert(assign[i]);
            }else{
                um_tracker.push_back(i);
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

    return 1;
}

int SortTracker::deal_duplicate_tracker(std::vector<int>& lost_tracker, std::vector<int> tracker_list){

    for(int i=0; i < lost_tracker.size(); i++){
        std::vector<int>::iterator location_index = find(tracker_list.begin(), tracker_list.end(), lost_tracker[i]);
        if((location_index - tracker_list.begin()) < tracker_list.size()){
           lost_tracker.erase(lost_tracker.begin()+i);
        }
    }

}