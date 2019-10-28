#include "sort_base_tracker.h"

using namespace sort;

int Tracker::id = 1;

Tracker::Tracker(cv::Rect_<float> init_box, int label){

    m_id = id;
    id++;
    m_state = STATE_NEW;
    m_time_since_update = 0;
    m_label = label;
    m_activate = false;
    m_box = init_box;

    //Init KalmanFilter
    int stateNum = 7;
    int measureNum = 4;
    m_kalman_filter = cv::KalmanFilter(stateNum, measureNum, CV_32FC1);
    m_measurement = cv::Mat::zeros(measureNum, 1, CV_32FC1);

    m_kalman_filter.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1);
        
    // std::cout<<m_kalman_filter.transitionMatrix.type()<<std::endl;
    // std::cout<<CV_64FC1<<std::endl;
    // std::cout<<CV_32FC2<<std::endl;
    // std::cout<<CV_64FC2<<std::endl;

    cv::setIdentity(m_kalman_filter.measurementMatrix);
    cv::setIdentity(m_kalman_filter.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(m_kalman_filter.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(m_kalman_filter.errorCovPost, cv::Scalar::all(1));

    // State is [center_x, center_y, area, width/height]
    m_kalman_filter.statePost.at<float>(0, 0) = init_box.x + init_box.width/2;
    m_kalman_filter.statePost.at<float>(1, 0) = init_box.y + init_box.height/2;
    m_kalman_filter.statePost.at<float>(2, 0) = init_box.area();
    m_kalman_filter.statePost.at<float>(3, 0) = init_box.width/init_box.height;

}

bool Tracker::is_activate(){
    return m_activate;
}

int Tracker::predict(){
    
    cv::Mat predicted_mat = m_kalman_filter.predict();
    std::cout<<std::endl;

    m_time_since_update += 1;
   
    cv::Rect_<float> predict_box = get_rect_box(predicted_mat.at<float>(0,0), 
                                                predicted_mat.at<float>(1,0),
                                                predicted_mat.at<float>(2,0), 
                                                predicted_mat.at<float>(3,0));

    m_box = predict_box;

    return 1;
}

int Tracker::update(cv::Rect_<float> det_result) {

    //measurement
    m_measurement.at<float>(0,0) = det_result.x + det_result.width / 2;
    m_measurement.at<float>(1,0) = det_result.y + det_result.height / 2;
    m_measurement.at<float>(2,0) = det_result.area();
    m_measurement.at<float>(3,0) = det_result.width / det_result.height;

    // std::cout<<"Correct：";
    // std::cout<<m_measurement<<std::endl;
    // cv::Mat uncorrect_mat = m_kalman_filter.statePost;
    // std::cout<<"x: "<<uncorrect_mat.at<float>(0,0)<<" y: "<<uncorrect_mat.at<float>(1,0);
    // std::cout<<" area: "<<uncorrect_mat.at<float>(2,0)<<" ratio: "<<uncorrect_mat.at<float>(3,0)<<std::endl;

    m_kalman_filter.correct(m_measurement);

    //use corrected position to do something
    cv::Mat corrected_mat = m_kalman_filter.statePost;

    // std::cout<<"x: "<<m_measurement.at<float>(0,0)<<" y: "<<m_measurement.at<float>(1,0);
    // std::cout<<" area: "<<m_measurement.at<float>(2,0)<<" ratio: "<<m_measurement.at<float>(3,0)<<std::endl;

    // std::cout<<"x: "<<corrected_mat.at<float>(0,0)<<" y: "<<corrected_mat.at<float>(1,0);
    // std::cout<<" area: "<<corrected_mat.at<float>(2,0)<<" ratio: "<<corrected_mat.at<float>(3,0)<<std::endl;

    cv::Rect_<float> corrected_box = get_rect_box(corrected_mat.at<float>(0,0), 
                                                corrected_mat.at<float>(1,0),
                                                corrected_mat.at<float>(2,0), 
                                                corrected_mat.at<float>(3,0));
                     
    m_box = corrected_box;
    

    //states
    m_time_since_update = 0;
    m_state = STATE_TRACKED;
    m_activate = true;

    return 1;
}

int Tracker::get_id(){
    return m_id;
}

int Tracker::get_state(){
    return m_state;
}

int Tracker::get_label(){
    return m_label;
}

int Tracker::get_miss_time(){
    return m_time_since_update;
}

cv::Rect_<float> Tracker::get_box(){
    return m_box;
}

int Tracker::mark_lost(){
    m_state = STATE_LOST;
    return STATE_LOST;
}

int Tracker::mark_removed(){
    m_state = STATE_REMOVE;
    return STATE_REMOVE;
}

cv::Rect_<float> Tracker::get_rect_box(float cx, float cy, float area, 
                                    float ratio) {

    float w = sqrt(area * ratio);
    float h = area / w;
    float x = (cx - w/2);
    float y = (cy - h/2);

    if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

    return cv::Rect_<float>(int(x),int(y),int(w),int(h));
}