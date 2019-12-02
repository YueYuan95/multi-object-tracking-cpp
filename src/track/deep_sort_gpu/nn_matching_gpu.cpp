#include "nn_matching_gpu.h"

DistanceMetricGPU::DistanceMetricGPU(){

    size_t size = 50 * sizeof(int);
    m_detection_idxs = (int*)malloc(size);
    cudaMalloc(&m_d_detection_idxs, size);

    m_all_cost.width = 0;
    m_all_cost.height = 0;
    m_all_cost.elements = (float*)malloc(m_max_tracker*m_max_detection*sizeof(float));
    for(int i=0; i < (m_max_tracker * m_max_detection); i++){
        m_all_cost.elements[i] = 1e5;
    }

    m_d_single_cost.width = 0;
    m_d_single_cost.height = 30;
    size = m_budget * m_max_detection * sizeof(float);
    cudaMalloc(&m_d_single_cost.elements, size);

    m_d_all_cost.width = 0;
    m_d_all_cost.height = 0;
    size = m_max_tracker * m_max_detection * sizeof(float);
    cudaMalloc(&m_d_all_cost.elements, size);
    cudaMemcpy(m_d_all_cost.elements, m_all_cost.elements, size,
                cudaMemcpyHostToDevice);

}

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
    for(int i=0; i<detection_indexs.size(); i++){
        m_detection_idxs[i] = detection_indexs[i];
    }
    size = col * sizeof(int);
    cudaMemcpy(m_d_detection_idxs, m_detection_idxs, size,
                cudaMemcpyHostToDevice);

    for(int i=0; i < (m_max_tracker * m_max_detection); i++){
        m_all_cost.elements[i] = 1e5;
    }
    size = m_max_tracker * m_max_detection * sizeof(float);
    cudaMemcpy(m_d_all_cost.elements, m_all_cost.elements, size,
                cudaMemcpyHostToDevice);

    m_all_cost.width = col; 
    m_all_cost.height = row;

    m_d_single_cost.width = col; 
    m_d_single_cost.height = m_budget;

    m_d_all_cost.width = col; 
    m_d_all_cost.height = row;

    //show_device_data(det_feature, "det_feature "+std::to_string(d_detection_idxs[0]));
    for(int i=0; i < tracker_ids.size(); i++){
        // if(tracker_ids[i] == 10) show_device_data(m_samples[tracker_ids[i]], "id_10_feature");
        // if(tracker_ids[i] == 24) show_device_data(m_samples[tracker_ids[i]], "id_24_feature");
        m_d_single_cost.height = m_samples[tracker_ids[i]].height;
        EuclideanMetric(i, m_samples[tracker_ids[i]], col, m_d_detection_idxs, det_feature, m_d_single_cost, m_d_all_cost);
        // if(tracker_ids[i] == 10) show_device_data(m_d_single_cost, "id_10_cost_matrix");
        // if(tracker_ids[i] == 24) show_device_data(m_d_single_cost, "id_24_cost_matrix");
    }

    size = m_d_all_cost.width * m_d_all_cost.height * sizeof(float);
    cudaMemcpy(m_all_cost.elements, m_d_all_cost.elements, size, cudaMemcpyDeviceToHost);

    for(int i=0; i < row; i++){
        for(int j=0; j < col; j++){
            cost_matrix[i][j] = m_all_cost.elements[i*m_all_cost.width+j];
        }
    }

    // free(m_all_cost.elements);
    // cudaFree(m_d_single_cost.elements);
    // cudaFree(m_d_all_cost.elements);
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

int DistanceMetricGPU::release(){

    free(m_all_cost.elements);
    cudaFree(m_d_single_cost.elements);
    cudaFree(m_d_all_cost.elements);
}