#include "test.h"

std::vector<cv::Mat> get_cpu_mats(){

    std::vector<cv::Mat> img_mats;
    cv::Mat img;
    // img = cv::imread("/root/data/image/person_reid/person_test_sample_1.jpg"); // car
    // img_mats.push_back(img);
    // img = cv::imread("/root/data/image/person_reid/person_test_sample_2.jpg");  // woman
    // img_mats.push_back(img);
    // img = cv::imread("/root/data/image/person_reid/person_test_sample_3.jpg");  // same with above
    // img_mats.push_back(img);
    // img = cv::imread("/root/data/image/person_reid/person_test_sample_4.jpg");  // man
    // img_mats.push_back(img);

    img = cv::imread("/root/data/image/person_reid/person_test_sample_5.jpg");  // red woman
    img_mats.push_back(img);
    img = cv::imread("/root/data/image/person_reid/person_test_sample_6.jpg");  // red woman
    img_mats.push_back(img);
    img = cv::imread("/root/data/image/person_reid/person_test_sample_8.jpg");  // Purple woman 
    img_mats.push_back(img);
    img = cv::imread("/root/data/image/person_reid/person_test_sample_9.jpg");  // Purple woman
    img_mats.push_back(img);


    return img_mats;
}

cv::Mat get_cpu_mat(){

    cv::Mat img = cv::imread("/root/data/image/person_reid/full_image.jpg");
    return img;

}

int get_gpu_mats(std::vector<bdavs::AVSGPUMat>& gpu_mats){

    convert_image_from_cpu_to_gpu(gpu_mats, get_cpu_mats());

}

int get_gpu_mat(bdavs::AVSGPUMat& gpu_mat){

    convert_image_from_cpu_to_gpu(gpu_mat, get_cpu_mat());
}

int get_gpu_mat(byavs::GpuMat& gpu_mat){

    convert_image_from_cpu_to_gpu(gpu_mat, get_cpu_mat());
}

int get_bounding_boxes(std::vector<cv::Rect_<float>>& object_boxes){

    cv::Rect_<float> box(1363,569,103,241);
    object_boxes.push_back(box);
    // box = cv::Rect_<float>(371,410,80,239);
    // object_boxes.push_back(box);
    // box = cv::Rect_<float>(103,549,83,251);
    // object_boxes.push_back(box);
}

int test_crop_gpu_mat(){

    bdavs::AVSGPUMat gpu_mat;
    std::vector<bdavs::AVSGPUMat> gpu_mats;
    std::vector<cv::Rect_<float>> object_boxes;

    get_gpu_mat(gpu_mat);
    get_bounding_boxes(object_boxes);
    crop_gpu_mat(gpu_mat, object_boxes, gpu_mats);

    //TODO: cudaFree(gpu_mat), cudaFree(gpu_mats)
}

