#include "nn_matching_gpu.h"

int DistanceMetricGPU::partial_fit(int tracker_id, int feature_idx, FeatureMatrix feature){

    if(m_samples.count(tracker_id)){
        UpdateFeature(m_samples[tracker_id], feature_idx, feature);
        m_samples[tracker_id].height += 1;
        if(m_samples[tracker_id].height > 30) m_samples[tracker_id].height = 30;
        m_samples[tracker_id].update_row = (m_samples[tracker_id].update_row + 1) % m_budget;
    }else{
        FeatureMatrix new_sample;
        new_sample.id = tracker_id;
        new_sample.width = FEATURE_SIZE;
        new_sample.update_row = 0;
        size_t size = new_sample.width * m_budget * sizeof(float);
        cudaMalloc(&new_sample.elements, size);
        UpdateFeature(new_sample, feature_idx, feature);
        new_sample.height = 1;
        new_sample.update_row = (new_sample.update_row + 1) % m_budget;
        m_samples[tracker_id] = new_sample;
    }
}

int DistanceMetricGPU::partial_fit(int tracker_id, float* feature){

    if(m_samples.count(tracker_id)){
        UpdateFeature(m_samples[tracker_id], feature);
        m_samples[tracker_id].height += 1;
        if(m_samples[tracker_id].height > 30) m_samples[tracker_id].height = 30;
        m_samples[tracker_id].update_row = (m_samples[tracker_id].update_row + 1) % m_budget;
    }else{
        FeatureMatrix new_sample;
        new_sample.id = tracker_id;
        new_sample.width = FEATURE_SIZE;
        new_sample.update_row = 0;
        size_t size = new_sample.width * m_budget * sizeof(float);
        cudaMalloc(&new_sample.elements, size);
        UpdateFeature(new_sample, feature);
        new_sample.height = 1;
        new_sample.update_row = (new_sample.update_row + 1) % m_budget;
        m_samples[tracker_id] = new_sample;
    }
}

/***
 *  The colmon of FeatureMatrix `det_feature` is a 2048*1 feature vector, so
 *  `det_feature`'s size is a 2048 * size of detection_box 
***/
int DistanceMetricGPU::distance(std::vector<std::vector<double>>& cost_matrix, std::vector<int> detection_indexs, 
                    FeatureMatrix det_feature, std::vector<int> tracker_ids){
    
    size_t size;
    int row = tracker_ids.size();
    int col = detection_indexs.size();

    cost_matrix.clear();
    cost_matrix.resize(row, std::vector<double>(col, 10000));

    //TODO: Malloc and free memory in the initalization and release of class
    int *detection_idxs, *d_detection_idxs;
    detection_idxs = (int*)malloc(col * sizeof(int));
    for(int i=0; i<detection_indexs.size(); i++){
        detection_idxs[i] = detection_indexs[i];
    }
    size = col * sizeof(int);
    cudaMalloc(&d_detection_idxs, size);
    cudaMemcpy(d_detection_idxs, detection_idxs, size,
                cudaMemcpyHostToDevice);

    FeatureMatrix all_cost;
    all_cost.width = col; all_cost.height = row;
    all_cost.elements = (float*)malloc(all_cost.width*all_cost.height*sizeof(float));
    for(int i=0; i < (all_cost.width * all_cost.height); i++){
        all_cost.elements[i] = 1e5;
    }

    FeatureMatrix d_single_cost;
    d_single_cost.width = FEATURE_SIZE; d_single_cost.height = m_budget;
    size = d_single_cost.width * d_single_cost.height * sizeof(float);
    cudaMalloc(&d_single_cost.elements, size);

    FeatureMatrix d_all_cost;
    d_all_cost.width = col; d_all_cost.height = row;
    size = d_all_cost.width * d_all_cost.height * sizeof(float);
    cudaMalloc(&d_all_cost.elements, size);
    cudaMemcpy(d_all_cost.elements, all_cost.elements, size,
                cudaMemcpyHostToDevice);

    //show_device_data(det_feature, "det_feature "+std::to_string(d_detection_idxs[0]));
    for(int i=0; i < tracker_ids.size(); i++){
        //if(tracker_ids[i] == 16) show_device_data(m_samples[tracker_ids[i]], "id_16_feature");
        d_single_cost.height = m_samples[tracker_ids[i]].height;
        EuclideanMetric(i, m_samples[tracker_ids[i]], col, d_detection_idxs, det_feature, d_single_cost, d_all_cost);
        //if(tracker_ids[i] == 16) show_device_data(d_single_cost, "id_16_cost_matrix");
    }

    size = d_all_cost.width * d_all_cost.height * sizeof(float);
    cudaMemcpy(all_cost.elements, d_all_cost.elements, size, cudaMemcpyDeviceToHost);

    for(int i=0; i < row; i++){
        for(int j=0; j < col; j++){
            cost_matrix[i][j] = all_cost.elements[i*all_cost.width+j];
        }
    }

    free(all_cost.elements);
    cudaFree(d_single_cost.elements);
    cudaFree(d_all_cost.elements);
}

int DistanceMetricGPU::distance(std::vector<std::vector<double>>& cost_matrix, std::vector<cv::Rect_<float>> detect_box, std::vector<BaseTrackerGPU> tracker){

    cost_matrix.clear();
    cost_matrix.resize(tracker.size(), std::vector<double>(detect_box.size(), 0.00));
    for(int i=0; i < tracker.size(); i++){
        for(int j=0; j < detect_box.size(); j++){
            cost_matrix[i][j] = 1 - get_iou(tracker[i].to_rect(), detect_box[j]);
        }
    }

}

int DistanceMetricGPU::remove_object(int object_id){

    cudaFree(m_samples[object_id].elements);
    m_samples[object_id].elements = nullptr;
    m_samples.erase(object_id);
}