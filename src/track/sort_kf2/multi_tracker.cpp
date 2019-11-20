#include "multi_tracker.h"

int MultiTracker::init(byavs::PedFeatureParas ped_feature_paras, std::string ped_model_dir, int gpu_id){
     
    extractor.init(ped_feature_paras, ped_model_dir, gpu_id);
}

int MultiTracker::inference(const byavs::TrackeInputGPU input, byavs::TrackeObjectGPUs& output){

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
    *   Step 1 : Predict and Extract Feature
    */
    // debug<<"step1"<<debugend;
    for(int i=0; i < m_tracker_list.size(); i++){
        m_tracker_list[i].predict(kf);
    }

    compute_detection_feature(input.gpuImg, detection_boxes, detection_feature);

    /*
    *   Step 2 : Match
    * */
    // debug<<"step2"<<debugend;
    std::map<int, int> matches;
    std::vector<int> um_tracker;
    std::vector<int> um_detection;

    match(matches, um_tracker, um_detection, detection_boxes, detection_feature);

    /*
    *   Step 3 : Update State
    * */
//    debug<<"step3"<<debugend;
   for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
       m_tracker_list[iter->first].update(kf, detection_boxes[iter->second], detection_feature[iter->second]);
   }

   for(int i=0; i < um_tracker.size(); i++){
       m_tracker_list[um_tracker[i]].mark_missed();
   }

   for(int i=0; i < um_detection.size(); i++){
       initiate_tracker(detection_label[um_detection[i]], detection_boxes[um_detection[i]], detection_feature[um_detection[i]]);
   }

    // debug<<"tracker size : "<<m_tracker_list.size()<<debugend;

    debug<<debugend;
    for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
        std::cout<<"id : "<<m_tracker_list[iter->first].get_id()<<" matched "<< iter->second <<std::endl;
    }
    std::cout<<std::endl;

   /*
   *    Step 4 : Push Result and delete the deleted trackers
   */
//   debug<<"step4"<<debugend;
   for(auto it = m_tracker_list.begin(); it != m_tracker_list.end();){
       //output and delete
       if((*it).is_deleted()){
            //TODO: Set the finish tracking flag to True
            if((*it).get_hits() > n_init){
                byavs::TrackeObjectGPU remove_object;
                remove_object.camID = input.camID;
                remove_object.channelID = input.channelID;
                remove_object.id = it->get_id();
                remove_object.label = it->get_label();
                remove_object.box = {int(it->to_rect().x), int(it->to_rect().y),
                        int(it->to_rect().width), int(it->to_rect().height)};
                remove_object.return_state = true;
                remove_object.match_flag = 0;
                remove_object.score = 0.0;
                //debug<<"remove object id is "<<remove_object.id<<debugend;
                output.push_back(remove_object);
            }
            it = m_tracker_list.erase(it);
       }else{
           if((*it).is_confirmed()){
                byavs::TrackeObjectGPU remove_object;
                remove_object.camID = input.camID;
                remove_object.channelID = input.channelID;
                remove_object.id = it->get_id();
                remove_object.label = it->get_label();
                remove_object.box = {int(it->to_rect().x), int(it->to_rect().y),
                        int(it->to_rect().width), int(it->to_rect().height)};
                remove_object.return_state = true;
                remove_object.match_flag = 0;
                remove_object.score = 0.0;
                //debug<<"remove object id is "<<remove_object.id<<debugend;
                output.push_back(remove_object);
           }
           it++;
       }
   }

   /*
   *    Step 5 : Update distance metric
   * */
//    debug<<"step5"<<debugend;
   
   std::vector<int> tracker_ids;
   std::vector<std::vector<float>> features;
   tracker_ids.clear();
   features.clear();

    // debug<<"step5.1"<<debugend;
   //get the featrue and confirmed tracker id list
   for(int i =0; i < m_tracker_list.size(); i++){
       if(m_tracker_list[i].is_confirmed()){
           for(int j=0; j < m_tracker_list[i].get_features().size(); j++){
               tracker_ids.push_back(m_tracker_list[i].get_id());
               features.push_back(m_tracker_list[i].get_features()[j]);
           }
           m_tracker_list[i].clear_features();
       }
   }
//    debug<<"step6"<<debugend;

   dm.partial_fit(features, tracker_ids); 

}

int MultiTracker::match(std::map<int, int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detection,
        std::vector<cv::Rect_<float>> detection_boxes, std::vector<std::vector<float>> detection_feature){
    
    std::vector<int> confirmed_trackers;
    std::vector<int> unconfirmed_trackers;
    confirmed_trackers.clear();
    unconfirmed_trackers.clear();

    for(int i=0; i < m_tracker_list.size(); i++){
        if(m_tracker_list[i].is_confirmed()){
            confirmed_trackers.push_back(i);
        }else{
            unconfirmed_trackers.push_back(i);
        } 
    }

    std::vector<int> detection_index;
    detection_index.clear();
    for(int i=0; i < detection_boxes.size(); i++){
        detection_index.push_back(i);
    }

    std::map<int, int> match_a;
    std::vector<int> um_detection_a;
    std::vector<int> um_trackers_a;
    match_a.clear();
    um_detection_a.clear();
    um_trackers_a.clear();

    la.matching_cascade(dm, kf,
        cascade_depth, detection_boxes,
        match_a, um_detection_a, um_trackers_a,
        m_tracker_list, detection_feature,
        confirmed_trackers, detection_index
    );

    debug<<"Un match Tracekr :";
    for(auto i : um_trackers_a) std::cout<< m_tracker_list[confirmed_trackers[i]].get_id()<< " ";
    std::cout<<std::endl;

    debug<<"Un match Detection :";
    for(auto i : um_detection_a) std::cout<< i << " ";
    std::cout<<std::endl;

    debug<<"Match A size :"<<match_a.size()<<debugend;

    for(std::map<int, int>::iterator iter= match_a.begin(); iter != match_a.end(); iter++){
        std::cout<<m_tracker_list[iter->first].get_id()<<" match "<< iter->second <<std::endl;
        matches[iter->first] = iter->second;
    }

    std::vector<int> iou_tracker_candidate;
    iou_tracker_candidate.clear();
    for(auto i : unconfirmed_trackers) iou_tracker_candidate.push_back(i);

    for(int i=0; i < um_trackers_a.size();){
        if(m_tracker_list[um_trackers_a[i]].get_time_since_update() == 1){
            iou_tracker_candidate.push_back(um_trackers_a[i]);
            um_trackers_a.erase(um_trackers_a.begin()+i);
        }else{
            i++;
        }
    }

    // debug<<" iou tracker candidate is :";
    // for(auto i : iou_tracker_candidate) std::cout<< i << " ";
    // std::cout<<std::endl;

    debug<<"************IOU************"<<debugend;

    std::map<int, int> match_b;
    std::vector<int> um_detection_b;
    std::vector<int> um_trackers_b;
    match_b.clear();
    um_detection_b.clear();
    um_trackers_b.clear();

    la.matching_by_iou(dm,
        match_b, um_detection_b, um_trackers_b,
        detection_boxes, um_detection_a,
        m_tracker_list, iou_tracker_candidate
    );

    debug<<"Match B size :"<<match_b.size()<<debugend;
    for(std::map<int, int>::iterator iter= match_b.begin(); iter != match_b.end(); iter++){
        assert(matches.count(iter->first) == 0);
        std::cout<<m_tracker_list[iter->first].get_id()<<" match "<< iter->second <<std::endl;
        matches[iter->first] = iter->second;
    }

    for(int i=0; i < um_trackers_a.size(); i++){
        um_trackers.push_back(um_trackers_a[i]);
    }
    for(int i=0; i < um_trackers_b.size(); i++){
        um_trackers.push_back(um_trackers_b[i]);
    }

    um_detection = um_detection_b;

    return 1;
}

int MultiTracker::initiate_tracker(int label, cv::Rect_<float> detection_box, std::vector<float> detection_feature){

    std::vector<cv::Mat> init_mean_cova = kf.initiate(detection_box);

    BaseTracker new_tracker(init_mean_cova[0], init_mean_cova[1], next_id, label, detection_feature);
    m_tracker_list.push_back(new_tracker);

    next_id += 1;

    return 1;
}

int MultiTracker::compute_detection_feature(byavs::GpuMat image, std::vector<cv::Rect_<float>> detection_boxes, 
                std::vector<std::vector<float>>& detection_features){
 
    std::vector<bdavs::AVSGPUMat> gpu_mats;
    //crop_gpu_mat do malloc, so we need to free memory
    crop_gpu_mat(image, detection_boxes, gpu_mats);
    extractor.inference(gpu_mats, detection_features);
    //TODO: Free `gpu_mats`
    release_avs_gpu_mat(gpu_mats);


}