#include "kalman_tracker_v2.h"

KalmanTrackerV2::KalmanTrackerV2(){

    ndim = 4;
    dt = 1.0;
    std_weight_position = 1. / 20;
    std_weight_velocity = 1. / 160;

    motion_mat = cv::Mat::eye(2*ndim, 2*ndim, CV_32F);
    for(int i=0; i < ndim; i++){
        motion_mat.at<float>(i, ndim+i) = dt;
    }
    // std::cout<<"motion_mat:"<<std::endl;
    // std::cout<<motion_mat<<std::endl;

    update_mat = cv::Mat::eye(ndim, 2*ndim, CV_32F);

    // std::cout<<"update_mat:"<<std::endl;
    // std::cout<<update_mat<<std::endl;
}

std::vector<cv::Mat> KalmanTrackerV2::initiate(cv::Rect_<float> measurement){
/*
 Parameters
 ----------
 measurement : Bounding box coordinates (x, y, a, h) with center position (x, y),
               aspect ratio a, and height h.
 Returns
 ----------
 [cv::Mat, cv::Mat]
    Return the mean vector (8 dimensional) and covariance matrix(8 x 8 dimensional)
    of the new track. Unobserved velocities are initialized to 0 mean.
*/
    std::vector<cv::Mat> return_mean_cov;
    return_mean_cov.clear();

    std::vector<float> measurement_vector = {
        measurement.x + measurement.width/2,
        measurement.y + measurement.height/2,
        measurement.width/measurement.height,
        measurement.height
    };

    cv::Mat mean_pos = (cv::Mat_<float>(ndim, 1) << 
        measurement_vector[0],
        measurement_vector[1],
        measurement_vector[2],
        measurement_vector[3]);

    cv::Mat mean_vel = cv::Mat::zeros(ndim, 1, CV_32F);
    cv::Mat mean;
    vconcat(mean_pos, mean_vel, mean);
    return_mean_cov.push_back(mean);

    cv::Mat std_cov = (cv::Mat_<float>(2 * ndim, 1) <<
        2 * std_weight_position * measurement_vector[3],
        2 * std_weight_position * measurement_vector[3],
        1e-2,
        2 * std_weight_position * measurement_vector[3],
        10 * std_weight_velocity * measurement_vector[3],
        10 * std_weight_velocity * measurement_vector[3],
        1e-5,
        10 * std_weight_velocity * measurement_vector[3]
    );

    cv::Mat squre_std_cov = std_cov.mul(std_cov);
    cv::Mat covariance = cv::Mat::diag(squre_std_cov);
    return_mean_cov.push_back(covariance);

    return return_mean_cov;
}

void KalmanTrackerV2::predict(cv::Mat& mean, cv::Mat& covariance){

    cv::Mat std_pos_vel = (cv::Mat_<float>(1, 2 * ndim) << 
        std_weight_position * mean.at<float>(3,0),
        std_weight_position * mean.at<float>(3,0),
        1e-2,
        std_weight_position * mean.at<float>(3,0),
        std_weight_velocity * mean.at<float>(3,0),
        std_weight_velocity * mean.at<float>(3,0),
        1e-5,
        std_weight_velocity * mean.at<float>(3,0)
    );

    cv::Mat motion_cov = std_pos_vel.mul(std_pos_vel);
    motion_cov = cv::Mat::diag(motion_cov);

    mean = motion_mat * mean;
    covariance = motion_mat*covariance*motion_mat.t() + motion_cov;

}

void KalmanTrackerV2::project(cv::Mat mean, cv::Mat covariance, 
                        cv::Mat& project_mean, cv::Mat& project_cova){

    cv::Mat std_pos = (cv::Mat_<float>(ndim, 1)<<
        std_weight_position * mean.at<float>(3, 0),
        std_weight_position * mean.at<float>(3, 0),
        1e-1,
        std_weight_position * mean.at<float>(3, 0)
    );

    std_pos = std_pos.mul(std_pos);
    cv::Mat innovation_cov = cv::Mat::diag(std_pos);

    project_mean = update_mat * mean;
    project_cova = update_mat * covariance * update_mat.t() + innovation_cov;

}

void KalmanTrackerV2::update(cv::Mat& mean, cv::Mat& cova, cv::Rect_<float> measurement){
    
    cv::Mat project_mean, project_cova;
    project(mean, cova, project_mean, project_cova);

    cv::Mat measurement_mat = (cv::Mat_<float>(ndim, 1) <<
        measurement.x + measurement.width/2,
        measurement.y + measurement.height/2,
        measurement.width/measurement.height,
        measurement.height
    );

    cv::Mat innovation = measurement_mat - project_mean;

    cv::Mat kalman_gain =  cova * update_mat.t() * project_cova.inv();

    mean = mean + kalman_gain * innovation;

    cv::Mat identity = cv::Mat::eye(2*ndim, 2*ndim, CV_32F);
    cova = (identity - kalman_gain * update_mat) * cova;

}

double KalmanTrackerV2::gating_distance(cv::Mat mean, cv::Mat covariance, 
    cv::Rect_<float> measurement, int only_position){

    cv::Mat project_mean, project_cova;
    project(mean, covariance, project_mean, project_cova);

    cv::Mat measurement_mat = (cv::Mat_<float>(ndim, 1) <<
        measurement.x + measurement.width/2,
        measurement.y + measurement.height/2,
        measurement.width/measurement.height,
        measurement.height
    );

    cv::Mat distance = measurement_mat - project_mean;
    cv::Mat z = distance.t() * project_cova.inv() * distance;
    assert(z.rows == z.cols == 1);

    return (double)z.at<float>(0,0);
}