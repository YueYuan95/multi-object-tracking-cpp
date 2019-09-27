#include "kalman_tracker.h"

int KalmanTracker::trk_count = 1;

KalmanTracker::KalmanTracker(){};

KalmanTracker::KalmanTracker(cv::Rect_<float> init_box, int label) {
    m_bbox = init_box;
    m_id = std::to_string(trk_count);
    trk_count++;
    m_label = label;

    //State : 1 is tracking , -1 is tracking over
    m_state = 0;
    //How many times this object being tracked
    m_age = 0;
    //How many times this object was not update
    m_time_since_update = 0;
    m_history.clear();
    m_history.push_back(init_box);

    //Init KalmanFilter
    int stateNum = 7;
    int measureNum = 4;
    m_kalman_filter = cv::KalmanFilter(stateNum, measureNum, 0);
    m_measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    m_kalman_filter.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1);

    cv::setIdentity(m_kalman_filter.measurementMatrix);
    cv::setIdentity(m_kalman_filter.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(m_kalman_filter.measurementNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(m_kalman_filter.errorCovPost, cv::Scalar::all(1));

    // State is [center_x, center_y, area, width/height]
    m_kalman_filter.statePost.at<float>(0, 0) = init_box.x + init_box.width/2;
    m_kalman_filter.statePost.at<float>(1, 0) = init_box.y + init_box.height/2;
    m_kalman_filter.statePost.at<float>(2, 0) = init_box.area();
    m_kalman_filter.statePost.at<float>(3, 0) = init_box.width/init_box.height;

}

int KalmanTracker::predict() {  
    cv::Mat predicted_mat = m_kalman_filter.predict();
    //std::cout<<"success"<<std::endl;
    m_age += 1;
    m_time_since_update += 1;
   
    cv::Rect_<float> predict_box = getRectBox(predicted_mat.at<float>(0,0), 
                                                predicted_mat.at<float>(1,0),
                                                predicted_mat.at<float>(2,0), 
                                                predicted_mat.at<float>(3,0));
    
    //Is this mean predicted history not corrected history?
    //Which history can inital a sot tracker?
    m_history.push_back(predict_box);

    m_bbox = predict_box;

    return 1;

}

int KalmanTracker::update(cv::Rect_<float> det_result) {
    //measurement
    m_measurement.at<float>(0,0) = det_result.x + det_result.width / 2;
    m_measurement.at<float>(1,0) = det_result.y + det_result.height / 2;
    m_measurement.at<float>(2,0) = det_result.area();
    m_measurement.at<float>(3,0) = det_result.width / det_result.height;

    //states
    m_time_since_update = 0;

    m_kalman_filter.correct(m_measurement);

    //use corrected position to do something
    cv::Mat corrected_mat = m_kalman_filter.statePost;

    cv::Rect_<float> corrected_box = getRectBox(corrected_mat.at<float>(0,0), 
                                                corrected_mat.at<float>(1,0),
                                                corrected_mat.at<float>(2,0), 
                                                corrected_mat.at<float>(3,0));
                     
    m_bbox = corrected_box;
    m_history.pop_back();
    m_history.push_back(m_bbox);

    return 1;
}

cv::Rect_<float> KalmanTracker::getRectBox(float cx, float cy, float area, 
                                           float ratio) {
    float w, h;
    if (area < 0 || ratio < 0) {      
        w = m_history[m_history.size()-1].width;
        h = m_history[m_history.size()-1].height;
    } else {
        w = sqrt(area * ratio);
        h = area / w;
    }

    float x = (cx - w/2);
    float y = (cy - h/2);

    if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

    return cv::Rect_<float>(int(x),int(y),int(w),int(h));
}

int KalmanTracker::setState(int state){
    m_state = state;
    return m_state;
}

int KalmanTracker::setDescriptor(std::vector<float> descriptor) {
    m_descriptor = descriptor;
    return 1;
}

cv::Rect_<float> KalmanTracker::getBbox() {
    return m_bbox;  
}

std::vector<float> KalmanTracker::getDescriptor() {
    return m_descriptor;
}

std::string KalmanTracker::getId() {
    return m_id;
}

int KalmanTracker::getLabel() {
    return m_label;
}

int KalmanTracker::getState() {
    return m_state;
}

int KalmanTracker::getTime() {
    return m_time_since_update;
}

std::vector<cv::Rect_<float>> KalmanTracker::getHistory() {
    return m_history;
}

cv::KalmanFilter KalmanTracker::getKalmanFilter() {
    return m_kalman_filter;
}
