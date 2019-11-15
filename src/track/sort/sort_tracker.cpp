#include "sort_tracker.h"

int SortTracker::init(byavs::PedFeatureParas ped_feature_paras, std::string ped_model_dir, int gpu_id){

    if(USE_DEEP){
        extractor.init(ped_feature_paras, ped_model_dir, gpu_id);
    }
    Tracker::reset_static_id();
}

int SortTracker::inference(const byavs::TrackeInputGPU input, byavs::TrackeObjectGPUs& output){

    std::vector<int> activated_trackers;
    std::vector<int> refind_trackers;
    std::vector<int> lost_trackers;
    std::vector<int> removed_trackers;

    std::vector<cv::Rect_<float>> detection_boxes;
    std::vector<int> detection_label;
    std::vector<std::vector<float>> detection_feature;
    detection_boxes.clear();
    detection_label.clear();
    detection_feature.clear();

    for(int i=0; i < input.objs.size(); i++){

        cv::Rect_<float> box = cv::Rect_<float>(input.objs[i].box.topLeftX, input.objs[i].box.topLeftY,
                                input.objs[i].box.width, input.objs[i].box.height);
        int label = input.objs[i].label;
        detection_boxes.push_back(box);
        detection_label.push_back(label);

    }
    /*
     * Step 1 : tracker predict
     *
     * */

    // for(int i=0; i < m_trackers.size(); i++){
         
    //      m_trackers[i].predict();
    //      //debug<<"Id "<<m_trackers[i].get_id()<<" box is "<<m_trackers[i].get_box()<<debugend;
    // }

    for(auto it = m_trackers.begin(); it != m_trackers.end();){
        it->predict();
        debug<<"Id "<<it->get_id()<<" box is "<<it->get_box()<<debugend;
        if(it->get_box().x < 0 || it->get_box().y < 0){
             debug<<"Remove Id "<<it->get_id()<<debugend;
             it = m_trackers.erase(it);
         }else{
             it++;
         }
    }


    /*
     * Step 2 : join the activate tracker and lost trackers
     *          generate the candidate tracker
     * */
    std::vector<int> unconfirmed_trackers;
    std::vector<int> tracked_trackers;
    unconfirmed_trackers.clear();                                   //new Tracker
    tracked_trackers.clear();

    std::vector<int> candidate;
    candidate.clear();

    for(int i = 0 ; i < m_trackers.size(); i++){
        if(m_trackers[i].is_activate()){
            candidate.push_back(i);
        }else{
            unconfirmed_trackers.push_back(i);
        }
    }
    
    //`candidate_tracakers` save the indexs of the `m_trackers`
    // std::vector<int> candidate;
    // candidate.clear();

    // generate_candidate_trackers(candidate, tracked_trackers, m_lost_trackers);
    //debug<<"candidate size :"<<candidate.size()<<debugend;

    //std::cout<<m_trackers.size()<<" "<<tracked_trackers.size()<<" "<<unconfirmed_trackers.size()<<std::endl;
    /*
    *   Temply do this
    */
    // for(int i=0; i < candidate.size(); i++){  
    //     //std::cout<<m_trackers[candidate[i]].get_box()<<std::endl;
    //     m_trackers[candidate[i]].predict();
    // }

    // for(int i=0; i < unconfirmed_trackers.size(); i++){
    //     m_trackers[unconfirmed_trackers[i]].predict();
    // }

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
        
        // bdavs::AVSGPUMat image; // TODO: Get the image from DetectionObject
        compute_detection_feature(input.gpuImg, detection_boxes, detection_feature);
        assert(detection_feature.size() == detection_boxes.size());
        std::vector<std::vector<double>> feature_cost_matrix;
        feature_cost_matrix.clear();

        //debug<<candidate.size() << ", "<< detection_feature.size() << debugend;

        if(candidate.size() != 0 && detection_feature.size() != 0){

            compute_feature_distance(feature_cost_matrix, candidate, detection_boxes, detection_feature);
            matching(feature_cost_matrix, matches, um_tracker, um_detection);

            // normalization(feature_cost_matrix);
            // compute_iou_distance(feature_cost_matrix, candidate, detection_boxes);
            // matching(feature_cost_matrix, matches, um_tracker, um_detection);

            debug<<debugend;
            for(int i=0; i < feature_cost_matrix.size(); i++){
                for(int j=0; j < feature_cost_matrix[i].size(); j++){
                    std::cout<<feature_cost_matrix[i][j]<<" ";
                }
                std::cout<<std::endl;
            }
            //debug<<"match size is "<< matches.size() << debugend;
            for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
                if(m_trackers[candidate[iter->first]].get_state() == STATE_TRACKED){
                    // if(m_trackers[candidate[iter->first]].get_id() == 5){
                    //     debug<<"currect box is "<< m_trackers[candidate[iter->first]].get_box() << ", udpate box is "<< detection_boxes[iter->second] << debugend;
                    //     debug<<"box iou is" << get_iou(m_trackers[candidate[iter->first]].get_box(), detection_boxes[iter->second]) << debugend;
                    //     for(int i=0; i<feature_cost_matrix[iter->first].size(); i++) std::cout<<" "<<feature_cost_matrix[iter->first][i]<<" ";
                    //     std::cout<<std::endl;
                    // }
                    m_trackers[candidate[iter->first]].update(detection_boxes[iter->second],detection_feature[iter->second]);
                    activated_trackers.push_back(candidate[iter->first]);
                }else{
                    m_trackers[candidate[iter->first]].update(detection_boxes[iter->second],detection_feature[iter->second]);
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
    std::vector<int> remain_detbox_index;           // Using the detbox_index, to record the index of detection box, for record the box label
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
            remain_detbox_index.push_back(um_detection[i]);
            remain_boxes.push_back(detection_boxes[um_detection[i]]);
        }
        
    }else{
        remain_tracker = candidate;
        remain_boxes = detection_boxes;
        for(int i=0; i<detection_boxes.size();i++){
            remain_detbox_index.push_back(i);
        }
    }

    assert(remain_detbox_index.size() == remain_boxes.size());
    // debug<<debugend;
    // std::cout<<"Detection boxes size : "<<detection_boxes.size()<<" Matched boxes size : "<<matches.size()<<" Remian Box size :"<<remain_boxes.size()<<std::endl;
    // for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
    //     std::cout<<"Detection Box :"<<detection_boxes[iter->second]<<std::endl;
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<remain_boxes.size(); i++){
    //     std::cout<<"Detection Box :"<<remain_boxes[i]<<" Index boxes is :"<<detection_boxes[remain_detbox_index[i]]<<std::endl;
    // }


    std::vector<std::vector<double>> iou_cost_matrix;
    iou_cost_matrix.clear();
    
    matches.clear();
    um_tracker.clear();
    um_detection.clear();

    if(remain_tracker.size() != 0 && remain_boxes.size() != 0){
        compute_iou_distance(iou_cost_matrix, remain_tracker, remain_boxes);
        matching(iou_cost_matrix, matches, um_tracker, um_detection);
        //debug<<"compute iou distance"<<debugend;
        for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
            if(m_trackers[remain_tracker[iter->first]].get_state() == STATE_TRACKED){
                m_trackers[remain_tracker[iter->first]].reset_kf_update(remain_boxes[iter->second], detection_feature[remain_detbox_index[iter->second]]);
                activated_trackers.push_back(remain_tracker[iter->first]);
            }else{
                m_trackers[remain_tracker[iter->first]].reset_kf_update(remain_boxes[iter->second], detection_feature[remain_detbox_index[iter->second]]);
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
            //debug<<"Lost Tracker ID: "<< m_trackers[remain_tracker[um_tracker[i]]].get_id()<<debugend;
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

    std::vector<int> remain_temp_indexs;
    remain_temp_indexs.clear();

    for(int i=0; i < um_detection.size(); i++){
        remain_boxes.push_back(detection_boxes[remain_detbox_index[um_detection[i]]]);
        remain_temp_indexs.push_back(remain_detbox_index[um_detection[i]]);
    }

    remain_detbox_index = remain_temp_indexs;

    assert(remain_detbox_index.size() == remain_boxes.size());
    // debug<<debugend;
    // std::cout<<"Detection boxes size : "<<detection_boxes.size()<<" Matched boxes size : "<<matches.size()<<" Remian Box size :"<<remain_boxes.size()<<std::endl;
    // for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
    //     std::cout<<"Detection Box :"<<detection_boxes[iter->second]<<std::endl;
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<remain_boxes.size(); i++){
    //     std::cout<<"Detection Box :"<<remain_boxes[i]<<" Index boxes is :"<<detection_boxes[remain_detbox_index[i]]<<std::endl;
    // }

    matches.clear();
    um_tracker.clear();
    um_detection.clear();
    
    if(unconfirmed_trackers.size() !=0 && remain_boxes.size() != 0){
        compute_iou_distance(iou_cost_matrix, unconfirmed_trackers, remain_boxes);
        matching(iou_cost_matrix, matches, um_tracker, um_detection);
        
        for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){       

            m_trackers[unconfirmed_trackers[iter->first]].reset_kf_update(remain_boxes[iter->second], detection_feature[remain_detbox_index[iter->second]]);
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
   
   remain_temp_indexs.clear();

   for(int i=0; i < um_detection.size(); i++){
        remain_boxes.push_back(detection_boxes[remain_detbox_index[um_detection[i]]]);
        remain_temp_indexs.push_back(remain_detbox_index[um_detection[i]]);
   }

   remain_detbox_index = remain_temp_indexs;
   
   int detection_box_size = um_detection.size();
   if(m_trackers.size() == 0){
       detection_box_size = detection_boxes.size();
       remain_boxes = detection_boxes;
   }

   if(USE_DEEP){
        for(int i=0; i < remain_detbox_index.size(); i++){
            Tracker new_tracker = Tracker(detection_boxes[remain_detbox_index[i]], detection_label[remain_detbox_index[i]], 
                                    detection_feature[remain_detbox_index[i]]);
            new_tracker_list.push_back(new_tracker);
        }
   }else{
        for(int i=0; i < remain_detbox_index.size(); i++){
            Tracker new_tracker = Tracker(detection_boxes[remain_detbox_index[i]], detection_label[remain_detbox_index[i]]);
            new_tracker_list.push_back(new_tracker);
        }
   }

//    debug<<"Tracker size is "<< m_trackers.size()<<" New Tracker size is "<<new_tracker_list.size()<<debugend;

   /*
   *  Step 6 : Mark the removed target when miss time achieve the threshold
   * */
   //std::cout<<"step 6"<<std::endl;
//    for(int i=0; i < lost_trackers.size(); i++){
//        if(m_trackers[lost_trackers[i]].get_miss_time() > MAX_MISS_TIME){
//             m_trackers[lost_trackers[i]].mark_removed();
//             removed_trackers.push_back(lost_trackers[i]);
//        }
//    }

    for(int i=0; i < m_trackers.size(); i++){
        if(m_trackers[i].get_state() == STATE_LOST && m_trackers[i].get_miss_time() > MAX_MISS_TIME){
            m_trackers[i].mark_removed();
        }
    }

   /*
   *  Step 7 : deal the tracker states
   * */
//    m_tracked_trackers.clear();
//    for(auto i : activated_trackers){
//        m_tracked_trackers.push_back(i);
//    }

//    for(auto i : refind_trackers){
//        m_tracked_trackers.push_back(i);
//    }
   
//    //remove the re-find tracker from `m_lost_tracker`
//    deal_duplicate_tracker(m_lost_trackers, m_tracked_trackers);

//    for(auto i : lost_trackers){
//        m_lost_trackers.push_back(i);
//    }

//    //remove the need to delete tracker from `m_lost_tracker`
//    deal_duplicate_tracker(m_lost_trackers, removed_trackers);

   /**
    * Step 8 : output
   */
   
   //output and delete the tracker that need to remove
   for(auto it = m_trackers.begin(); it != m_trackers.end();){
       //output and delete
       if((*it).get_state() == STATE_REMOVE){
            //TODO: Set the finish tracking flag to True 
            byavs::TrackeObjectGPU remove_object;
            remove_object.camID = input.camID;
            remove_object.channelID = input.channelID;
            remove_object.id = it->get_id();
            remove_object.label = it->get_label();
            remove_object.box = {int(it->get_box().x), int(it->get_box().y),
                    int(it->get_box().width), int(it->get_box().height)};
            remove_object.return_state = true;
            remove_object.match_flag = 0;
            remove_object.score = 0.0;
            //debug<<"remove object id is "<<remove_object.id<<debugend;
            output.push_back(remove_object);
            it = m_trackers.erase(it);
       }else{
           it++;
       }
   }

   //add new tracker
   for(int i=0; i < new_tracker_list.size(); i++){
       //debug<<"id is "<<new_tracker_list[i].get_id()<<debugend;
       m_trackers.push_back(new_tracker_list[i]);
   }
   // if the number of currect trackers equal the new trackers,
   // it means this time is first to tracking and should output
   // the result as first frame result.
   if(m_trackers.size() == new_tracker_list.size()){
       for(int i=0; i < m_trackers.size(); i++){
            byavs::TrackeObjectGPU output_object;
            output_object.camID = input.camID;
            output_object.channelID = input.channelID;
            output_object.id = m_trackers[i].get_id();
            output_object.label = m_trackers[i].get_label();
            output_object.box = {int(m_trackers[i].get_box().x), int(m_trackers[i].get_box().y),
                    int(m_trackers[i].get_box().width), int(m_trackers[i].get_box().height)};
            output_object.gpuImg = input.gpuImg;
            output_object.return_state = false;
            output_object.match_flag = 1;
            output_object.score = 0.0;
            output.push_back(output_object);
       }
   }
   
   for(int i=0; i < m_trackers.size(); i++){
       
       if(m_trackers[i].get_state() == STATE_TRACKED){
           //output
            //debug<<"Id : "<< m_trackers[i].get_id() << " Box :"<< m_trackers[i].get_box()<<debugend;
            if(int(m_trackers[i].get_box().x) > 0 && int(m_trackers[i].get_box().y) > 0){
                byavs::TrackeObjectGPU output_object;
                output_object.camID = input.camID;
                output_object.channelID = input.channelID;
                output_object.id = m_trackers[i].get_id();
                output_object.label = m_trackers[i].get_label();
                output_object.box = {int(m_trackers[i].get_box().x), int(m_trackers[i].get_box().y),
                        int(m_trackers[i].get_box().width), int(m_trackers[i].get_box().height)};
                output_object.gpuImg = input.gpuImg;
                output_object.return_state = false;
                output_object.match_flag = 1;
                output_object.score = 0.0;
                output.push_back(output_object);
            }
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

    for(auto i : m_lost_trackers){
        debug<<"have index "<< i << debugend;
        candidate.push_back(i);
    }

    for(int i=0; i<tracked_tracker.size();i++){
        
        std::vector<int>::iterator location_index = find(m_lost_trackers.begin(), m_lost_trackers.end(), tracked_tracker[i]);
        if((location_index - m_lost_trackers.begin()) < m_lost_trackers.size()){
            debug<<"find index "<< tracked_tracker[i]<<" remove index is "<<(*location_index) << debugend;
           m_lost_trackers.erase(location_index);
        }
        candidate.push_back(tracked_tracker[i]);
    }

    for(auto i : m_lost_trackers){
        debug<<"Push index "<< i << debugend;
        candidate.push_back(i);
    }

}

int SortTracker::compute_detection_feature(byavs::GpuMat image, std::vector<cv::Rect_<float>> detection_boxes, 
                std::vector<std::vector<float>>& detection_features){
 
    std::vector<bdavs::AVSGPUMat> gpu_mats;
    //crop_gpu_mat do malloc, so we need to free memory
    crop_gpu_mat(image, detection_boxes, gpu_mats);
    extractor.inference(gpu_mats, detection_features);
    //TODO: Free `gpu_mats`
    release_avs_gpu_mat(gpu_mats);


}

int SortTracker::compute_feature_distance(std::vector<std::vector<double>>& cost_matrix, std::vector<int> tracker_index, 
                std::vector<cv::Rect_<float>> detection_boxes,std::vector<std::vector<float>> detection_features){
    //TODO: compute the distance of features
    int row = tracker_index.size();
    int col = detection_features.size();

    cost_matrix.resize(row, std::vector<double>(col, 100.0));
    for(int i=0; i < row; i++){
        for(int j=0; j < col; j++){
          
            double iou = get_iou(m_trackers[tracker_index[i]].get_box(), detection_boxes[j]);
            //double feature_dis = get_feature_distance_cosine(m_trackers[tracker_index[i]].get_features(), detection_features[j]);
            double feature_dis = get_feature_distance_euclidean(m_trackers[tracker_index[i]].get_features(), detection_features[j]);
            // if(iou < 0.3 || feature_dis > 1.0){
            //     cost_matrix[i][j] = 1000;
            //     continue;
            // }
            cost_matrix[i][j] = feature_dis;
        }
    }

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