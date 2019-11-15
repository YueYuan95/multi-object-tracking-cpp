#include "test.h"

int get_kalman_test_boxes(std::vector<cv::Rect_<float>>& object_boxes){

    cv::Rect_<float> box(1363,569,103,241);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(371,410,80,239);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(103,549,83,251);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(1501,35,41,127);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(822,293,68,207);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(1342,10,44,136);
    object_boxes.push_back(box);
    box = cv::Rect_<float>(1324,488,68,207);
    object_boxes.push_back(box);
}

int test_kf_initiate(){

    cv::Rect_<float> box(68,36,64,128);
    KalmanTrackerV2 kf;
    std::vector<cv::Mat> mean_cov = kf.initiate(box);
    cv::Mat mean = mean_cov[0];
    cv::Mat covn = mean_cov[1];

    std::cout<<"mean :"<<std::endl;
    std::cout<<mean<<std::endl;

    std::cout<<"covn :"<<std::endl;
    std::cout<<covn<<std::endl;

    std::cout<<std::endl;

    kf.predict(mean, covn);

    std::cout<<"mean :"<<std::endl;
    std::cout<<mean<<std::endl;

    std::cout<<"covn :"<<std::endl;
    std::cout<<covn<<std::endl;

    cv::Mat project_mean;
    cv::Mat project_cova;
    kf.project(mean, covn, project_mean, project_cova);

    std::cout<<"project mean"<<std::endl;
    std::cout<<project_mean<<std::endl;
    std::cout<<"project cova"<<std::endl;
    std::cout<<project_cova<<std::endl;

    cv::Rect_<float> det_box(100,100,64,128);
    kf.update(mean, covn, det_box);

    std::cout<<"update mean"<<std::endl;
    std::cout<<mean<<std::endl;
    std::cout<<"update conv"<<std::endl;
    std::cout<<covn<<std::endl;

    cv::Rect_<float> gate_box(60,50,80,100);
    kf.gating_distance(mean, covn, gate_box);

}

int test_kalman_batch_test(){

    std::vector<cv::Rect_<float>> object_boxes;
    get_kalman_test_boxes(object_boxes);

    cv::Rect_<float> box(68,36,64,128);
    KalmanTrackerV2 kf;
    std::vector<cv::Mat> mean_cov = kf.initiate(box);
    cv::Mat mean = mean_cov[0];
    cv::Mat covn = mean_cov[1];
    std::cout<<"Mean :"<<std::endl;
    std::cout<<mean<<std::endl;
    std::cout<<"covn :"<<std::endl;
    std::cout<<covn<<std::endl;

    for(int i=0; i<object_boxes.size(); i++){
        std::cout<<"**************************"<<std::endl;
        kf.predict(mean, covn);
        std::cout<<"Mean :"<<std::endl;
        std::cout<<mean<<std::endl;
        std::cout<<"covn :"<<std::endl;
        std::cout<<covn<<std::endl;
        double distance = kf.gating_distance(mean, covn, object_boxes[i]);
        std::cout<<"distance :"<<distance<<std::endl;
        kf.update(mean, covn, object_boxes[i]);
        std::cout<<"Mean :"<<std::endl;
        std::cout<<mean<<std::endl;
        std::cout<<"covn :"<<std::endl;
        std::cout<<covn<<std::endl;
        std::cout<<"**************************"<<std::endl;
        std::cout<<std::endl;
    }
}