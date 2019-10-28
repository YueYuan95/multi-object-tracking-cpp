/***************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/common
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The following functions are tools, in every chapter
****************************************************************************/
#include "util.h"

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
int visualize (int frame, cv::Mat img, byavs::TrackeObjectCPUs results, 
                std::string result_dir) {
    std::cout<<"results size: "<<results.size()<<std::endl;
    for (int j=0; j < results.size(); j++) {
        int id = results[j].id;
        // std::cout<<"id : "<<id<<std::endl;
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
                    CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(0,255,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,0,0), 3, 8, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2,img.rows/2),0,00, CV_INTER_LINEAR);
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
                    CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(255,0,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,0,0), 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0,00, CV_INTER_LINEAR);
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
                    CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(0,0,255), 3, 8);
        cv::rectangle(img, detect_result[j], cv::Scalar(0, 255, 0), 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0, 00, CV_INTER_LINEAR);
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
        cv::putText(img, id, left_top, CV_FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(255, 100, 0), 3, 8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,100,0), 
                        3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2, img.rows/2), 0, 00, 
                CV_INTER_LINEAR);
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

/*
WriteResult a specific file
*/
int writeResult (int frame, byavs::TrackeObjectCPUs tracking_results) {

    std::ofstream outfile("tracking_result_0925/MOT16-11/MOT16-11.txt", 
                            std::ios::app);
    //std::ofstream outfile(result_dir + txt_name, std::ios::app);

    if (!outfile.is_open()) {
        std::cerr << "Error: can not find file " << std::endl;
    }
    if (outfile.is_open()) {
        if (frame==1) {
            std::cout << tracking_results.size() << std::endl;
        }
        for (int i=0; i < tracking_results.size(); i++) {
            
            outfile << frame << ", " << tracking_results[i].id << ", "
                << tracking_results[i].box.topLeftX << ", "
                << tracking_results[i].box.topLeftY << ", "
                << tracking_results[i].box.width << ", "
                << tracking_results[i].box.height << ", "
                << "-1, " << "-1, " << "-1, " << "-1, " << "\n";
            
        }
        outfile.close();
    }
    return 0;
}
