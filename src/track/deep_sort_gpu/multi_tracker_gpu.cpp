#include "multi_tracker_gpu.h"

int MultiTrackerGPU::init(byavs::PedFeatureParas ped_feature_paras, std::string ped_model_dir, int gpu_id){
     
    extractor.init(ped_feature_paras, ped_model_dir, gpu_id);
    size_t size = FEATURE_SIZE * sizeof(float);
    cudaMalloc(&m_single_feature, size);

    size = 50 * FEATURE_SIZE * sizeof(float);
    cudaMalloc(&detection_feature.elements, size);

}

int MultiTrackerGPU::inference(const byavs::TrackeInputGPU input, byavs::TrackeObjectGPUs& output){

    std::vector<cv::Rect_<float>> detection_boxes;
    std::vector<int> detection_label;
    //FeatureMatrix detection_feature;
    detection_boxes.clear();
    detection_label.clear();

    for(int i=0; i < input.objs.size(); i++){

        cv::Rect_<float> box = cv::Rect_<float>(input.objs[i].box.topLeftX, input.objs[i].box.topLeftY,
                                input.objs[i].box.width, input.objs[i].box.height);
        int label = input.objs[i].label;
        detection_boxes.push_back(box);
        detection_label.push_back(label);
    }

    detection_feature.height = FEATURE_SIZE;
    detection_feature.width = detection_boxes.size();
    debug<<"detection size : "<<detection_boxes.size()<<debugend;

    /*     
    *   Step 1 : Predict and Extract Feature
    */
    // debug<<"step1"<<debugend;
    double start, end;
    start = clock();
    for(int i=0; i < m_tracker_list.size(); i++){
        m_tracker_list[i].predict(kf);
    }
    end = clock();
    debug<<"kalmanfilter cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;

    start = clock();
    compute_detection_feature(input.gpuImg, detection_boxes, detection_feature);

    end = clock();
    debug<<"compute detection feature cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
   
    //show_device_data(detection_feature, "detect_feature");

    /*
    *   Step 2 : Match
    * */
    // debug<<"step2"<<debugend;
    std::map<int, int> matches;
    std::vector<int> um_tracker;
    std::vector<int> um_detection;
    
    start = clock();
    match(matches, um_tracker, um_detection, detection_boxes, detection_feature);
    end = clock();
    debug<<"match cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;

    /*
    *   Step 3 : Update State
    * */
   //debug<<"step3"<<debugend;
   start = clock();
   for(std::map<int,int>::iterator iter=matches.begin(); iter != matches.end(); iter++){
       get_object_feature(iter->second, detection_feature, m_single_feature);
       m_tracker_list[iter->first].update(kf, detection_boxes[iter->second], m_single_feature);
   }

   for(int i=0; i < um_tracker.size(); i++){
       m_tracker_list[um_tracker[i]].mark_missed();
   }

   //show_device_data(detection_feature, "detection_feature");
   for(int i=0; i < um_detection.size(); i++){
       get_object_feature(um_detection[i], detection_feature, m_single_feature);
       //show_device_data(m_single_feature, "get_object_feature");
       initiate_tracker(detection_label[um_detection[i]], detection_boxes[um_detection[i]], m_single_feature);
   }
    end = clock();
    debug<<"update and initiate cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
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
    start = clock();
   for(auto it = m_tracker_list.begin(); it != m_tracker_list.end();){
       //output and delete
       if((*it).is_deleted()){
            //TODO: Set the finish tracking flag to True
            //TODO: free the samples in dm
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
                dm.remove_object(remove_object.id);
                it->release_memory();
                output.push_back(remove_object);
            }
            it = m_tracker_list.erase(it);
       }else{
           if((*it).is_confirmed() && (*it).get_time_since_update() < 5){
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
   end = clock();
   debug<<"send cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;

   /*
   *    Step 5 : Update distance metric
   * */
//    debug<<"step5"<<debugend;

   debug<<"tracker size is "<< m_tracker_list.size() << debugend;
   
   double update_start = clock();
   start = clock();
   std::vector<int> tracker_ids;
   std::vector<std::vector<float>> features;
   tracker_ids.clear();
   features.clear();

   // debug<<"step5.1"<<debugend;
   //get the featrue and confirmed tracker id list
   for(int i =0; i < m_tracker_list.size(); i++){
       if(m_tracker_list[i].is_confirmed()){
           for(int j=0; j < m_tracker_list[i].get_features().size(); j++){
                double single_start = clock();
                dm.partial_fit(m_tracker_list[i].get_id(), m_tracker_list[i].get_features()[j]);
                debug<<"single update feature cost time : "<<(double)(clock() - single_start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
           }
           double tracker_free_start = clock();
           m_tracker_list[i].clear_features();
           debug<<"tracker free feature cost time : "<<(double)(clock() - tracker_free_start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
       }
   }
//    debug<<"step6"<<debugend;
   end = clock();
   debug<<"update feature cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
   
   start = clock();
   //cudaFree(detection_feature.elements);
   end = clock();
   debug<<"free feature cost time : "<<(double)(end -start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;

   end = clock();
   debug<<"update cost time : "<<(double)(end -update_start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;

}

int MultiTrackerGPU::match(std::map<int, int>& matches, std::vector<int>& um_trackers, std::vector<int>& um_detection,
        std::vector<cv::Rect_<float>> detection_boxes, FeatureMatrix detection_feature){
    
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

int MultiTrackerGPU::initiate_tracker(int label, cv::Rect_<float> detection_box, float* detection_feature){

    std::vector<cv::Mat> init_mean_cova = kf.initiate(detection_box);

    BaseTrackerGPU new_tracker(init_mean_cova[0], init_mean_cova[1], next_id, label, detection_feature);
    m_tracker_list.push_back(new_tracker);

    next_id += 1;

    return 1;
}

int MultiTrackerGPU::compute_detection_feature(byavs::GpuMat image, std::vector<cv::Rect_<float>> detection_boxes, 
                FeatureMatrix& detection_features){
 
    std::vector<bdavs::AVSGPUMat> gpu_mats;
    std::vector<std::vector<float>> temp_detection_features;
    temp_detection_features.clear();
    //crop_gpu_mat do malloc, so we need to free memory
    crop_gpu_mat(image, detection_boxes, gpu_mats);
    extractor.inference(gpu_mats, temp_detection_features);
    //TODO: Free `gpu_mats`
    release_avs_gpu_mat(gpu_mats);

    int width = temp_detection_features.size();
    int height = FEATURE_SIZE;
    size_t size =  width * height * sizeof(float);
    float *convert_feature = (float*)malloc(size);

    debug<<debugend;
    // std::cout<<temp_detection_features[0].size()<<std::endl;
    for(int i=0; i < width; i++){
        for(int j=0; j < height; j++){
            // if(i==0){
            //     std::cout<<temp_detection_features[i][j]<<" ";
            //     if(j % 256 == 0) std::cout<<std::endl;
            // }
            convert_feature[j*width+i] = temp_detection_features[i][j];
        }
    }
    // std::cout<<std::endl;
    // debug<< "convert_feature" <<debugend;
    // for(int i=0; i < height; i++){
    //     for(int j=0; j < width; j++){
    //         if(j==0){
    //             std::cout<<convert_feature[i*width+j]<<" ";
    //         }
    //     }
    // }
    // std::cout<<std::endl;


    cudaMemcpy(detection_features.elements, convert_feature, size, cudaMemcpyHostToDevice);

    // float *h_featrue = (float*)malloc(size*sizeof(float));
    // cudaMemcpy(h_featrue, detection_features.elements, size, cudaMemcpyDeviceToHost);
    // debug<<debugend;
    
    // std::cout<<std::endl;
    // debug<< "h_feature" <<debugend;
    // for(int i=0; i < height; i++){
    //     for(int j=0; j < width; j++){
    //         if(j==0){
    //             std::cout<<convert_feature[i*width+j]<<" ";
    //         }
    //     }
    // }
    // std::cout<<std::endl;

    
}

int MultiTrackerGPU::get_object_feature(int index, FeatureMatrix detect_feature, float *object_feature){

    GetObjectFeature(index, detect_feature, object_feature);

}

int MultiTrackerGPU::release(){

    cudaFree(m_single_feature);
    cudaFree(detection_feature.elements);
}