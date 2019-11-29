#include "base_tracker_gpu.h"

BaseTrackerGPU::BaseTrackerGPU(cv::Mat mean, cv::Mat covariance, int tracker_id, int tracker_label,
    float* feature){

    m_mean = mean;
    m_covariance = covariance;
    m_id = tracker_id;
    m_label = tracker_label;
    m_hits = 1;
    m_age = 1;
    m_time_since_update = 0;

    m_state = Tentative;

    size = FEATURE_SIZE * sizeof(float);
    m_free_d_memory.clear();
    m_feature.clear();
    for(int i=0; i < n_init; i++){
        float *temp_feature;
        cudaMalloc(&temp_feature, size);
        m_free_d_memory.push_back(temp_feature);
    }

    cudaMemcpy(m_free_d_memory[0], feature, size,
            cudaMemcpyHostToDevice);
    m_feature.push_back(m_free_d_memory[0]);
    m_free_d_memory.erase(m_free_d_memory.begin());
    
}

int BaseTrackerGPU::predict(KalmanTrackerV2 kf){
    
    kf.predict(m_mean, m_covariance);
    m_age += 1;
    m_time_since_update += 1;

}

int BaseTrackerGPU::update(KalmanTrackerV2 kf, cv::Rect_<float> detection, 
            float* feature){

    kf.update(m_mean, m_covariance, detection);
    //TODO: Malloc in the construct function
    cudaMemcpy(m_free_d_memory[0], feature, size,
            cudaMemcpyHostToDevice);
    m_feature.push_back(m_free_d_memory[0]);
    m_free_d_memory.erase(m_free_d_memory.begin());

    m_hits += 1;
    m_time_since_update = 0;

    if(m_state == Tentative && m_hits >= n_init){
        m_state = Confirmed;
    }
}

cv::Rect_<float> BaseTrackerGPU::to_rect(){

    float x = m_mean.at<float>(0,0);
    float y = m_mean.at<float>(0,1);
    float a = m_mean.at<float>(0,2);
    float h = m_mean.at<float>(0,3);
    float w = a * h;

    x = x - (w / 2.0);
    y = y - (h / 2.0);

    return cv::Rect_<float>(x,y,w,h);

}

int BaseTrackerGPU::mark_missed(){

    if(m_state == Tentative){
        m_state = Deleted;
    }else{
        if(m_time_since_update > max_age){
            m_state = Deleted;
        }
    }
}

int BaseTrackerGPU::is_tentative(){
    return m_state == Tentative;
}

int BaseTrackerGPU::is_confirmed(){
    return m_state == Confirmed;
}

int BaseTrackerGPU::is_deleted(){
    return m_state == Deleted;
}

int BaseTrackerGPU::get_id(){
    return m_id;
}

int BaseTrackerGPU::get_hits(){
    return m_hits;
}

int BaseTrackerGPU::get_age(){
    return m_age;
}

int BaseTrackerGPU::get_label(){
    return m_label;
}

int BaseTrackerGPU::get_time_since_update(){
    return m_time_since_update;
}

cv::Mat BaseTrackerGPU::get_mean(){
    return m_mean;
}

cv::Mat BaseTrackerGPU::get_covariance(){
    return m_covariance;
}

std::vector<float*> BaseTrackerGPU::get_features(){
    return m_feature;
}

void BaseTrackerGPU::clear_features(){
    // NOT really clear
    double start, end;
    for(int i=0; i<m_feature.size(); i++){
        start = clock();
        m_free_d_memory.push_back(m_feature[i]);
        end = clock();
        debug<<"tracker single cuda free cost time : "<<(double)(end - start)/CLOCKS_PER_SEC*1000<<" ms" <<debugend;
    }
    m_feature.clear();
}

void BaseTrackerGPU::release_memory(){

    for(int i=0; i < m_feature.size(); i++){
        cudaFree(m_feature[i]);
    }
    for(int j=0; j < m_free_d_memory.size(); j++){
        cudaFree(m_free_d_memory[j]);
    }

    m_feature.clear();
    m_free_d_memory.clear();
}
