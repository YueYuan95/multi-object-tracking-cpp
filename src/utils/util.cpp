/***************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/common
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The following functions are tools, in every chapter
****************************************************************************/
#include "util.h"

/**
 *   TODO: Move followed code to visualization class and delete some unnesseary code
 * *

/*
Pretraversal
*/
int preorderTraversal (treeNode tree_node, std::vector<int>& path, 
                       std::vector<std::vector<int>>& path_list) {

    path.push_back(tree_node.index);
    if (tree_node.children.size() == 0){
        path_list.push_back(path);
    } else {
        for (int i = 0; i < tree_node.children.size(); i++) {
            preorderTraversal(*(tree_node.children[i]), path, path_list);
            path.pop_back();
        }
    }
}

/*Post traversal*/
int backTraversal (treeNode tree_node, std::shared_ptr<treeNode> head_node,
         std::vector<int>& path, std::vector<std::vector<int>>& path_list, 
         int N) {

    path.push_back(tree_node.index);
    
    //If the depth of the tree is not big than N
    if (tree_node.parent == head_node){
        path.push_back(tree_node.parent->index);
        if (path.size() <= N) {
            for (int i = N-path.size()+1; i > 0; i--) {
                path.push_back(0);
            }
        }
        path_list.push_back(path);
        return 1;
    }
   //If this node is a root node
    if (tree_node.parent == NULL &&  path.size() <= N) {
        for (int i = N-path.size()+1; i > 0; i--){
            path.push_back(0);
        }
        path_list.push_back(path);
        return 1;
    }
    if (tree_node.parent != NULL) {
        backTraversal(*(tree_node.parent), head_node, path, path_list, N);
    }
}

/* Transfers a tree to a graph*/
int TreeToGraph(std::vector<Tree> tree_list, Graph& graph) {


    std::vector<int> path; 
    std::vector<std::vector<int>> path_list;
    std::vector<VexNode> graph_node_list;
    
    for (auto tree : tree_list) {
        std::cout << "Tree No." << tree.getId() << std::endl;
        //preorderTraversal(tree.getHead(),path, path_list);
        for (auto leaf : tree.getLeafNode()) {
            path.clear();
            backTraversal(*(leaf), tree.getHead(), 
                        path, path_list, tree.getN());
        }
        for (auto path : path_list) {
            VexNode graph_node;
            graph_node.path.clear();
            for(int i = path.size()-1; i >= 0; i--){
                std::cout << path[i] << " ";
                graph_node.id = tree.getId();
                graph_node.path.push_back(path[i]);
            }
            std::cout << std::endl;
            graph_node_list.push_back(graph_node);
         }
         path_list.clear();
    }
    graph = Graph(graph_node_list);
}

/*Visualizes the tracking results */
int visualize (int frame, cv::Mat img, byavs::TrackeObjectGPUs results, 
                std::string result_dir) {

    for (int j=0; j < results.size(); j++) {
        int id = results[j].id;
        // std::cout<<"id : "<<id<<std::endl;
        //debug<<results[j].box.topLeftX<<", "<<results[j].box.topLeftY<<std::endl;
        cv::Point left_top = cv::Point(float(results[j].box.topLeftX), 
                                        float(results[j].box.topLeftY));
        cv::Point right_bottom = cv::Point(float(results[j].box.topLeftX 
                                            + results[j].box.width), 
                                           float(results[j].box.topLeftY
                                            + results[j].box.height));
        //std::cout<<"create point"<<std::endl;
        // std::cout<<left_top<<std::endl;
        // std::cout<<right_bottom<<std::endl;
        cv::putText(img, std::to_string(id), left_top, 
                    cv::FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(0,255,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,0,0), 3, 8, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2,img.rows/2),0,00, cv::INTER_LINEAR);
    cv::imwrite(result_dir + std::to_string(frame) + ".jpg", img);
}

/*
Visualizes the tracking results in only in the script test.cpp
Input:frame: the frame number; img: the kth image; results: tracking results
*/
int visualize (int frame, cv::Mat img, byavs::TrackeObjectCPUs results) {

    for(int j=0; j < results.size(); j++) {
        int id = results[j].id;

        cv::Point left_top = cv::Point(results[j].box.topLeftX, 
                                        results[j].box.topLeftY);
        cv::Point right_bottom = cv::Point(results[j].box.topLeftX
                                            + results[j].box.width, 
                                           results[j].box.topLeftY
                                            + results[j].box.height);
       
        cv::putText(img, std::to_string(id), left_top, 
                    cv::FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(255,0,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,0,0), 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0,00, cv::INTER_LINEAR);
    cv::imwrite("tracking_result_0925/MOT16-11/" 
                + std::to_string(frame) + ".jpg", img);
}

/*
Saves the detection result after NMS
DEPRECATED, it was previously used to save det_result after NMS
*/
int visualize (int frame, cv::Mat img, std::vector<cv::Rect_<float>> detect_result) {  


    for (int j=0; j < detect_result.size(); j++) {
        std::string id = std::to_string(j+1);

        cv::putText(img, id, cv::Point(detect_result[j].x + detect_result[j].width, 
                                        detect_result[j].y), 
                    cv::FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(0,0,255), 3, 8);
        cv::rectangle(img, detect_result[j], cv::Scalar(0, 255, 0), 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0, 00, cv::INTER_LINEAR);
    cv::imwrite("det_result_MOT16-11" + std::to_string(frame) + ".jpg", img);
}

/*
save the Kalman prediction if Kalman filter is used
DEPRECATED,beacase it is not stable
*/
int visualize (int frame, cv::Mat img, 
                byavs::TrackeObjectCPUs results, char filter) {  

    for (int j=0; j < results.size(); j++) {
        std::string id = std::to_string(results[j].id);

        cv::Point left_top = cv::Point(results[j].box.topLeftX, 
                                        results[j].box.topLeftY);
        cv::Point right_bottom = cv::Point(results[j].box.topLeftX 
                                            + results[j].box.width, 
                                            results[j].box.topLeftY
                                            + results[j].box.height);
        cv::putText(img, id, left_top, cv::FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(255, 100, 0), 3, 8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,100,0), 
                        3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0, 00, 
                cv::INTER_LINEAR);
    cv::imwrite("Kalman_predict/" + std::to_string(frame) + ".jpg", img);
}

/*
Opens a dir and gets information
Input:dir name, filename and default
*/
void listDir (const char *name, std::vector<std::string> &fileNames, 
                bool lastSlash) {

    DIR *dir;
    struct dirent *entry;
    struct stat statbuf;
    struct tm *tm;
    time_t rawtime;
    if (!(dir = opendir(name))) {
        std::cout << "Couldn't open the file or dir" << name << "\n";
        return;
    }
    if (!(entry = readdir(dir))) {
        std::cout<<"Couldn't read the file or dir"<<name<<"\n";
        return;
    }
        do
    {
        std::string slash = "";
        if (!lastSlash)
          slash = "/";

        std::string parent(name);
        std::string file(entry->d_name);
        std::string final = parent + slash + file;
        if (stat(final.c_str(), &statbuf) == -1) {
            std::cout << "Couldn't get the stat info of file or dir: " << final 
                        << "\n";
            return;
        }
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        } else { 
                fileNames.push_back(final);
        }
    } while (entry = readdir(dir));
        closedir(dir);
}

/* 
compare the vex score between A and B and return the bigger
*/
bool VexSort (VexNode a, VexNode b) {

    return (a.score > b.score);
}

/* 
compare the vex score between A and B and return the smaller
*/
bool VexSortUp (VexNode a, VexNode b) {

    return (a.score < b.score);
}

double get_iou (cv::Rect_<float> detection, cv::Rect_<float> tracker) {
   float in = (tracker & detection).area();
   float un = tracker.area() + detection.area() - in;
  
   if (un < 0.001) {
        return 0.0;
   }
  
   return (double)(in/un);
}

/* 
Calculate the overlap base on rect1
*/
double get_ov_n1 (cv::Rect_<float> rec1, cv::Rect_<float> rec2) {

    float in = (rec1 & rec2).area();
    float un = rec1.area();
    return (double)(in/un);
}

/*
Calculate the overlap base on rect2
*/
double get_ov_n2 (cv::Rect_<float> rec1, cv::Rect_<float> rec2) {

    float in = (rec1 & rec2).area();
    float un = rec2.area();
    return (double)(in/un);
}

/*
Write result in files
Input: frame number, tracking results, dir to save and txt to save
*/
int writeResult (int frame, byavs::TrackeObjectGPUs tracking_results, 
                std::string result_dir, std::string txt_name) {

    std::ofstream outfile(result_dir + txt_name, std::ios::app);

    if (!outfile.is_open()) {
        std::cerr << "Error: can not find file " << std::endl;
    }

    if (outfile.is_open()) {
        for (int i=0; i<tracking_results.size(); i++) {
            
            outfile << frame<<","<< tracking_results[i].id <<","
                << tracking_results[i].box.topLeftX << "," 
                << tracking_results[i].box.topLeftY << ","
                << tracking_results[i].box.width << ","
                << tracking_results[i].box.height << ","
                << "-1," << "-1," << "-1," << "-1" << "\n";
        }
        outfile.close();
    }
    return 0;
}

/*
Write result in files
Input: frame number, tracking results, dir to save and txt to save
*/
int writeResult (int frame, byavs::TrackeObjectCPUs tracking_results, 
                std::string result_dir, std::string txt_name) {

    std::ofstream outfile(result_dir + txt_name, std::ios::app);

    if (!outfile.is_open()) {
        std::cerr << "Error: can not find file " << std::endl;
    }
    if (outfile.is_open()) {
        for (int i=0; i<tracking_results.size(); i++) {
            
            outfile << frame<<","<< tracking_results[i].id <<","
                << tracking_results[i].box.topLeftX << "," 
                << tracking_results[i].box.topLeftY << ","
                << tracking_results[i].box.width << ","
                << tracking_results[i].box.height << ","
                << "-1," << "-1," << "-1," << "-1" << "\n";
            
        }
        outfile.close();
    }
    return 0;
}

/***
 *      Moved End
 * 
 * ****/

bool convert_image_from_cpu_to_gpu(std::vector<byavs::GpuMat>& gpu_mats, std::vector<cv::Mat> img_mats){

    for(int i=0; i<img_mats.size(); i++){

        byavs::GpuMat avs_gpu_mat;
        avs_gpu_mat.width = img_mats[i].cols;
        avs_gpu_mat.height = img_mats[i].rows;
        avs_gpu_mat.channels = 3;
    
        unsigned char *temp_gpu_data;
        cudaMalloc((void **)&temp_gpu_data, 3 * img_mats[i].cols * img_mats[i].rows);
        cudaMemcpy(temp_gpu_data, img_mats[i].data, 3 * img_mats[i].cols * img_mats[i].rows, cudaMemcpyHostToDevice);
        avs_gpu_mat.data = temp_gpu_data;
        gpu_mats.push_back(avs_gpu_mat);
    }

}

bool convert_image_from_cpu_to_gpu(std::vector<bdavs::AVSGPUMat>& gpu_mats, std::vector<cv::Mat> img_mats){

    for(int i=0; i<img_mats.size(); i++){

        bdavs::AVSGPUMat avs_gpu_mat;
        avs_gpu_mat.width = img_mats[i].cols;
        avs_gpu_mat.height = img_mats[i].rows;
        avs_gpu_mat.channels = 3;
    
        unsigned char *temp_gpu_data;
        cudaMalloc((void **)&temp_gpu_data, 3 * img_mats[i].cols * img_mats[i].rows);
        cudaMemcpy(temp_gpu_data, img_mats[i].data, 3 * img_mats[i].cols * img_mats[i].rows, cudaMemcpyHostToDevice);
        avs_gpu_mat.data = temp_gpu_data;
        gpu_mats.push_back(avs_gpu_mat);
    }
}

bool convert_image_from_cpu_to_gpu(bdavs::AVSGPUMat & gpu_mat, cv::Mat img_mat){

    gpu_mat.width = img_mat.cols;
    gpu_mat.height = img_mat.rows;
    gpu_mat.channels = 3;
    
    unsigned char *temp_gpu_data;
    cudaMalloc((void **)&temp_gpu_data, 3 * img_mat.cols * img_mat.rows);
    cudaMemcpy(temp_gpu_data, img_mat.data, 3 * img_mat.cols * img_mat.rows, cudaMemcpyHostToDevice);
    gpu_mat.data = temp_gpu_data;
    
}

bool convert_image_from_cpu_to_gpu(byavs::GpuMat & gpu_mat, cv::Mat img_mat){

    gpu_mat.width = img_mat.cols;
    gpu_mat.height = img_mat.rows;
    gpu_mat.channels = 3;
    
    unsigned char *temp_gpu_data;
    cudaMalloc((void **)&temp_gpu_data, 3 * img_mat.cols * img_mat.rows);
    cudaMemcpy(temp_gpu_data, img_mat.data, 3 * img_mat.cols * img_mat.rows, cudaMemcpyHostToDevice);
    gpu_mat.data = temp_gpu_data;
    
}

double get_feature_distance_euclidean(std::vector<float> feature_a, std::vector<float> feature_b){

    assert(feature_a.size() == feature_b.size());
    double distance;
    for(int i=0; i<feature_a.size(); i++){
        distance += pow((feature_a[i]-feature_b[i]), 2);
    }
    //distance = sqrt(distance);
    return (double)distance;
}

double get_feature_distance_cosine(std::vector<float> feature_a, std::vector<float> feature_b){

    assert(feature_a.size() == feature_b.size());
    
    double denominator = get_pow_sum(feature_a) * get_pow_sum(feature_b);
    if(denominator == 0) return 1.00;
    double distance = get_vector_time(feature_a, feature_b);
    distance = distance / denominator;

    return distance;
}

int get_feature_cost_matrix(std::vector<std::vector<double>>& cost_matrix, std::vector<std::vector<float>> feature_list_a, 
                std::vector<std::vector<float>> feature_list_b){
    
    int row = feature_list_a.size();
    int col = feature_list_b.size();

    cost_matrix.resize(row, std::vector<double>(col, 1000.0));
    for(int i=0; i < row; i++){
        for(int j=0; j < col; j++){
            if(i == j) continue;
            cost_matrix[i][j] = get_feature_distance_euclidean(feature_list_a[i], feature_list_b[j]);
        }
    }

}

int crop_avs_mat(byavs::GpuMat gpu_mat, cv::Rect_<float> box, bdavs::AVSGPUMat& roi){

    unsigned  char *cropImg=nullptr;
    cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels*sizeof(unsigned char));
 
    bdavs::cudaCropImage(gpu_mat.data,gpu_mat.width,gpu_mat.height,gpu_mat.channels,
                  cropImg,
                  box.x,box.y,box.width,box.height);

    roi.data = cropImg;
    roi.width = box.width;
    roi.height = box.height;
    roi.channels = gpu_mat.channels;
    return 0;

}

int crop_avs_mat(bdavs::AVSGPUMat gpu_mat, cv::Rect_<float> box, bdavs::AVSGPUMat& roi){

    unsigned  char *cropImg=nullptr;
    cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels*sizeof(unsigned char));
 
    bdavs::cudaCropImage(gpu_mat.data,gpu_mat.width,gpu_mat.height,gpu_mat.channels,
                  cropImg,
                  box.x,box.y,box.width,box.height);

    roi.data = cropImg;
    roi.width = box.width;
    roi.height = box.height;
    roi.channels = gpu_mat.channels;
    return 0;

}

int crop_gpu_mat(byavs::GpuMat gpu_mat, std::vector<cv::Rect_<float>> object_boxes, std::vector<bdavs::AVSGPUMat>& gpu_mats){

    for(int i=0; i<object_boxes.size(); i++){   
        cv::Rect_<float> box = object_boxes[i];
        bdavs::AVSGPUMat roi;
        if(box.x < 0) box.x = 0;
        if(box.y < 0) box.y = 0;
        if((box.x + box.width)>gpu_mat.width) box.width = gpu_mat.width - box.x;
        if((box.y + box.height)>gpu_mat.height) box.height = gpu_mat.height - box.y;
        assert(box.x >= 0 && box.y >=0 && (box.x + box.width)<=gpu_mat.width && (box.y + box.height)<=gpu_mat.height);
        //cropAVSGPUMat(image_gpu, box, roi);
        crop_avs_mat(gpu_mat, box, roi);
        gpu_mats.push_back(roi);
    }

}

int crop_gpu_mat(bdavs::AVSGPUMat gpu_mat, std::vector<cv::Rect_<float>> object_boxes, std::vector<bdavs::AVSGPUMat>& gpu_mats){

    for(int i=0; i<object_boxes.size(); i++){   
        cv::Rect_<float> box = object_boxes[i];
        bdavs::AVSGPUMat roi;
        assert(box.x >= 0 && box.y >=0 && (box.x + box.width)<=gpu_mat.width && (box.y + box.height)<=gpu_mat.height);
        //cropAVSGPUMat(image_gpu, box, roi);
        crop_avs_mat(gpu_mat, box, roi);
        gpu_mats.push_back(roi);
    }

}

int convert_to_tracking_input(std::string img_pth, std::vector<cv::Rect_<float>> detection_boxes, 
                        std::vector<float> detection_scores, byavs::TrackeInputGPU& objects){
    
    cv::Mat img = cv::imread(img_pth);
    byavs::GpuMat gpu_img;
    convert_image_from_cpu_to_gpu(gpu_img, img);
    objects.camID = 0;
    objects.channelID = 0;
    objects.gpuImg = gpu_img;
    for(int i=0; i< detection_boxes.size(); i++){
        byavs::DetectObject object;
        object.label = 0;
        object.score = detection_scores[i];
        object.box = {int(detection_boxes[i].x), int(detection_boxes[i].y),
                    int(detection_boxes[i].width), int(detection_boxes[i].height)};
        objects.objs.push_back(object);
    }

}

int release_avs_gpu_mat(std::vector<bdavs::AVSGPUMat> &image_list){
    for (int i =0;i<image_list.size();i++)
    {
        if (image_list[i].data!=NULL)
        {
            cudaFree(image_list[i].data);
        }
    }
}


int normalization(std::vector<std::vector<double>>& cost_matrix){

    double max = -100.00, min = 100.00;
    for(int i=0; i < cost_matrix.size(); i++){
        for(int j=0; j < cost_matrix[i].size(); j++){
            if(max < cost_matrix[i][j]) max = cost_matrix[i][j];
            if(min > cost_matrix[i][j]) min = cost_matrix[i][j];
        }
    }

    for(int i=0; i < cost_matrix.size(); i++){
        for(int j=0; j < cost_matrix[i].size(); j++){
            cost_matrix[i][j] = (cost_matrix[i][j] - min) / (max - min);
        }
    }

}

std::vector<float> norm(std::vector<float> data){


    float sum = 0.0000;
    std::vector<float> norm_data(data.size(), 0);

    for(int i=0; i < data.size(); i++){
        sum += pow(data[i], 2);
    }

    sum = sqrt(sum);

    for(int i=0; i < data.size(); i++){
        if(sum == 0.000){
            norm_data[i] = 0.000;
            continue;
        }
        norm_data[i] = data[i] / sum;
    }

    return norm_data;
    
}

double get_pow_sum(std::vector<float> data){

    double sum = 0.000;
    for(int i=0; i < data.size(); i++){
        sum += pow(data[i], 2);
    }

    sum = sqrt(sum);

    return sum;
}

double get_vector_time(std::vector<float> a, std::vector<float> b){

    assert(a.size() == b.size());
   
    double distance = 0.000;
    for(int i=0; i < a.size(); i++){
        distance += a[i] * b[i];
    }

    return distance;
}

int show_device_data(FeatureMatrix feature_matrix, std::string flag){
    
    debug<<flag<<debugend;
    size_t size = feature_matrix.height*feature_matrix.width*sizeof(float);
    float *feature = (float*)malloc(size);
    cudaMemcpy(feature, feature_matrix.elements, size, cudaMemcpyDeviceToHost);
    for(int i=0; i < feature_matrix.height; i++){
        for(int j=0; j < feature_matrix.width; j++){
            if(i < 40){
                std::cout<<feature[i*feature_matrix.width+j]<<" ";
            }
        }
        if(i < 40) std::cout<<std::endl;
    }
    std::cout<<std::endl;
    free(feature);
}

int show_device_data(float* feature_float, std::string flag){
    
    debug<<flag<<debugend;
    size_t size = FEATURE_SIZE*sizeof(float);
    float *feature = (float*)malloc(size);
    cudaMemcpy(feature, feature_float, size, cudaMemcpyDeviceToHost);
    for(int i=0; i < FEATURE_SIZE; i++){
        std::cout<<feature[i]<<" ";
    }
    std::cout<<std::endl;
    free(feature);
}