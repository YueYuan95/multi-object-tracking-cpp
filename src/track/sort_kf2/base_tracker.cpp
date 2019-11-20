#include "base_tracker.h"

BaseTracker::BaseTracker(cv::Mat mean, cv::Mat covariance, int tracker_id, int tracker_label,
    std::vector<float> feature){

    m_mean = mean;
    m_covariance = covariance;
    m_id = tracker_id;
    m_label = tracker_label;
    m_hits = 1;
    m_age = 1;
    m_time_since_update = 0;

    m_state = Tentative;

    if(feature.size() != 0){
        m_feature.push_back(feature);
    }

}

int BaseTracker::predict(KalmanTrackerV2 kf){
    
    kf.predict(m_mean, m_covariance);
    m_age += 1;
    m_time_since_update += 1;

}

int BaseTracker::update(KalmanTrackerV2 kf, cv::Rect_<float> detection, 
            std::vector<float> feature){

    kf.update(m_mean, m_covariance, detection);
    m_feature.push_back(feature);

    m_hits += 1;
    m_time_since_update = 0;

    if(m_state == Tentative && m_hits >= n_init){
        m_state = Confirmed;
    }
}

cv::Rect_<float> BaseTracker::to_rect(){

    float x = m_mean.at<float>(0,0);
    float y = m_mean.at<float>(0,1);
    float a = m_mean.at<float>(0,2);
    float h = m_mean.at<float>(0,3);
    float w = a * h;

    x = x - (w / 2.0);
    y = y - (h / 2.0);

    return cv::Rect_<float>(x,y,w,h);

}

int BaseTracker::mark_missed(){

    if(m_state == Tentative){
        m_state = Deleted;
    }else{
        if(m_time_since_update > max_age){
            m_state = Deleted;
        }
    }
}

int BaseTracker::is_tentative(){
    return m_state == Tentative;
}

int BaseTracker::is_confirmed(){
    return m_state == Confirmed;
}

int BaseTracker::is_deleted(){
    return m_state == Deleted;
}

int BaseTracker::get_id(){
    return m_id;
}

int BaseTracker::get_hits(){
    return m_hits;
}

int BaseTracker::get_age(){
    return m_age;
}

int BaseTracker::get_label(){
    return m_label;
}

int BaseTracker::get_time_since_update(){
    return m_time_since_update;
}

cv::Mat BaseTracker::get_mean(){
    return m_mean;
}

cv::Mat BaseTracker::get_covariance(){
    return m_covariance;
}

std::vector<std::vector<float>> BaseTracker::get_features(){
    return m_feature;
}

void BaseTracker::clear_features(){
    m_feature.clear();
}
