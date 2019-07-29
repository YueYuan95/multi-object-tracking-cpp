#include "kalman_tracker.h"

int KalmanTracker::trk_count = 1;

KalmanTracker::KalmanTracker(cv::Rect_<float> init_box, int label){
    
    m_bbox = init_box;
    m_id = std::to_string(trk_count);
    trk_count++;
    //0 is car, 1 is tricycle, 2 is pedestrian, 3 is rider
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
    m_kalman_filter.statePost.at<float>(0, 0) = init_box.x + init_box.width / 2;
    m_kalman_filter.statePost.at<float>(1, 0) = init_box.y + init_box.height/ 2;
    m_kalman_filter.statePost.at<float>(2, 0) = init_box.area();
    m_kalman_filter.statePost.at<float>(3, 0) = init_box.width / init_box.height;

}

int KalmanTracker::predict(cv::Mat image){

    cv::Mat predicted_mat = m_kalman_filter.predict();

    m_age += 1;
    m_time_since_update += 1;

    cv::Rect_<float> predict_box = getRectBox(image, predicted_mat.at<float>(0,0), predicted_mat.at<float>(1,0),
                                             predicted_mat.at<float>(2,0), predicted_mat.at<float>(3,0));

    //Is this mean predicted history not corrected history?
    //Which history can inital a sot tracker?
    m_history.push_back(predict_box);

    m_bbox = predict_box;

    return 1;

}

int KalmanTracker::update(cv::Mat image, cv::Rect_<float> det_result){

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

    cv::Rect_<float> corrected_box = getRectBox(image, corrected_mat.at<float>(0,0), corrected_mat.at<float>(1,0),
                                             corrected_mat.at<float>(2,0), corrected_mat.at<float>(3,0));
                     
    m_bbox = corrected_box;
    m_history.pop_back();
    m_history.push_back(m_bbox);

    return 1;

}

cv::Rect_<float> KalmanTracker::getRectBox(cv::Mat image, float cx, float cy, float area, float ratio){

    float w = sqrt(area * ratio);
    float h = area / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    x = std::min(image.rows, std::max(x,0));
    y = std::min(image.cols, std::max(y,0));
    w = (image.rows - (x+w)) > 0 ? w : image.rows;
    h = (image.cols - (y+h)) > 0 ? h : image.cols;

    return cv::Rect_<float>(x,y,w,h);

}

int KalmanTracker::setState(int state){

    m_state = state;

    return m_state;
}

int KalmanTracker::setDescriptor(std::shared_ptr<float> descriptor){

    m_descriptor = descriptor;
    LOG(INFO) << "Descriptor shared ptr use count is" << m_descriptor.use_count();

    return 1;
}

cv::Rect_<float> KalmanTracker::getBbox(){

    return m_bbox;  
}

std::shared_ptr<float> KalmanTracker::getDescriptor(){

    return m_descriptor;
}

std::string KalmanTracker::getId(){

    return m_id;
}

int KalmanTracker::getLabel(){

    return m_label;
}

int KalmanTracker::getState(){

    return m_state;
}

int KalmanTracker::getTime(){

    return m_time_since_update;
}

std::vector<cv::Rect_<float>> KalmanTracker::getHistory(){

    return m_history;
}