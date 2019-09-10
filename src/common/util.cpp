#include "util.h"


/*
 *===================================
 *  Tree convert to Graph
 *
 *===================================
*/

int preorderTraversal(treeNode tree_node, std::vector<int>& path, 
                std::vector<std::vector<int>>& path_list){
    
    path.push_back(tree_node.index);
    if(tree_node.children.size() == 0){
        path_list.push_back(path);
    }else{
        for(int i=0; i < tree_node.children.size(); i++){
            preorderTraversal(*(tree_node.children[i]), path, path_list);
            path.pop_back();
        }
    }

}

int backTraversal(treeNode tree_node, std::shared_ptr<treeNode> head_node,
         std::vector<int>& path, std::vector<std::vector<int>>& path_list, int N){

    path.push_back(tree_node.index);
    
    /*When the depth of the tree is not big than N*/
    if(tree_node.parent == head_node){
        path.push_back(tree_node.parent->index);
        if(path.size() <= N){
            for(int i=N-path.size()+1;i>0;i--){
                path.push_back(0);
            }
        }
        path_list.push_back(path);
        return 1;
    }
   /*When this node is a root node*/
    if(tree_node.parent == NULL &&  path.size() <= N){
        
        for(int i=N-path.size()+1; i > 0; i--){
            path.push_back(0);
        }
        path_list.push_back(path);
        return 1;

    }
    if(tree_node.parent != NULL){

        backTraversal(*(tree_node.parent), head_node, path, path_list, N);
    }
}

int TreeToGraph(std::vector<Tree> tree_list, Graph& graph){

    std::vector<int> path; 
    std::vector<std::vector<int>> path_list;
    std::vector<VexNode> graph_node_list;
    
    for(auto tree : tree_list){
        
        std::cout<<"Tree No."<<tree.getId()<<std::endl;
        //preorderTraversal(tree.getHead(),path, path_list);
        for(auto leaf : tree.getLeafNode()){
            path.clear();
            backTraversal(*(leaf), tree.getHead(), path, path_list, tree.getN());
        }
        for(auto path : path_list){
            VexNode graph_node;
            graph_node.path.clear();
            for(int i = path.size()-1; i >=0; i--){
                std::cout<<path[i]<<" ";
                graph_node.id = tree.getId();
                graph_node.path.push_back(path[i]);
            }
            std::cout<<std::endl;
            graph_node_list.push_back(graph_node);
         }
         path_list.clear();
    }
    
    graph = Graph(graph_node_list);
}

int visualize(int frame, cv::Mat img, byavs::TrackeObjectCPUs results)
{

    for(int j=0; j < results.size(); j++)
    {

        std::string id = std::to_string(results[j].id);

        cv::Point left_top = cv::Point(results[j].box.topLeftX, results[j].box.topLeftY);////////////
        cv::Point right_bottom = cv::Point(results[j].box.topLeftX+results[j].box.width, results[j].box.topLeftY+results[j].box.height);
        cv::putText(img, id, left_top, CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(255,0,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,0,0), 3, 1, 0);
        //cv::rectangle(img, left_top, right_bottom, ss, 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2,img.rows/2),0,00, CV_INTER_LINEAR);
    //cv::imshow("test",img);
    //cv::imwrite("result/"+std::to_string(frame)+".jpg", img);
    cv::imwrite("tracking_result_mending/MOT16-13/"+std::to_string(frame)+".jpg", img);
    
    //cv::waitKey(1);

}

int visualize(int frame, cv::Mat img, std::vector<cv::Rect_<float>> detect_result)
{

    for(int j=0; j < detect_result.size(); j++)
    {

        std::string id = std::to_string(j+1);

        cv::putText(img, id, cv::Point(detect_result[j].x + detect_result[j].width, detect_result[j].y), CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(0,0,255),3,8);
        cv::rectangle(img, detect_result[j], cv::Scalar(0,255,0), 3, 1, 0);
        //cv::rectangle(img, left_top, right_bottom, ss, 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2,img.rows/2),0,00, CV_INTER_LINEAR);
    //cv::imshow("test",img);
    //cv::imwrite("result/"+std::to_string(frame)+".jpg", img);
    cv::imwrite("det_result_MOT16-13/"+std::to_string(frame)+".jpg", img);
    
    //cv::waitKey(1);

}

int visualize(int frame, cv::Mat img, byavs::TrackeObjectCPUs results, char filter)
{

    for(int j=0; j < results.size(); j++)
    {

        std::string id = std::to_string(results[j].id);

        cv::Point left_top = cv::Point(results[j].box.topLeftX, results[j].box.topLeftY);////////////
        cv::Point right_bottom = cv::Point(results[j].box.topLeftX+results[j].box.width, results[j].box.topLeftY+results[j].box.height);
        cv::putText(img, id, left_top, CV_FONT_HERSHEY_SIMPLEX, 1 ,cv::Scalar(255,100,0),3,8);
        cv::rectangle(img, left_top, right_bottom, cv::Scalar(255,100,0), 3, 1, 0);
        //cv::rectangle(img, left_top, right_bottom, ss, 3, 1, 0);
    }
    cv::resize(img, img, cv::Size(img.cols/2,img.rows/2),0,00, CV_INTER_LINEAR);
    //cv::imshow("test",img);
    //cv::imwrite("result/"+std::to_string(frame)+".jpg", img);
    cv::imwrite("Kalman_predict/"+std::to_string(frame)+".jpg", img);
    
    //cv::waitKey(1);

}

void listDir(const char *name, std::vector<std::string> &fileNames, bool lastSlash)
{
    DIR *dir;
    struct dirent *entry;
    struct stat statbuf;
    struct tm      *tm;
    time_t rawtime;
    if (!(dir = opendir(name)))
    {
        std::cout<<"Couldn't open the file or dir"<<name<<"\n";
        return;
    }
    if (!(entry = readdir(dir)))
    {
        std::cout<<"Couldn't read the file or dir"<<name<<"\n";
        return;
    }
        do
    {
        std::string slash="";
        if(!lastSlash)
          slash = "/";

        std::string parent(name);
        std::string file(entry->d_name);
        std::string final = parent + slash + file;
        if(stat(final.c_str(), &statbuf)==-1)
        {
            std::cout<<"Couldn't get the stat info of file or dir: "<<final<<"\n";
            return;
        }
                if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) //its a directory
        {
                        //skip the . and .. directory
            //if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            //    continue;
                        //listDir(final.c_str(), fileNames, false);
                        continue;
                }
                else // it is a file
                {
                        fileNames.push_back(final);
                }
        }while (entry = readdir(dir));
        closedir(dir);
}

bool VexSort(VexNode a, VexNode b){

    return (a.score > b.score);
}

bool VexSortUp(VexNode a, VexNode b){

    return (a.score < b.score);
}

double get_iou(cv::Rect_<float> detection, cv::Rect_<float> tracker){
  
   float in = (tracker & detection).area();
   float un = tracker.area() + detection.area() - in;
  
   if(un < 0.001){
        return 0.0;
   }
  
   return (double)(in/un);
}

double get_ov_n1(cv::Rect_<float> rec1, cv::Rect_<float> rec2)
{
    float in = (rec1 & rec2).area();
    float un = rec1.area();
    return (double)(in/un);
}

double get_ov_n2(cv::Rect_<float> rec1, cv::Rect_<float> rec2)
{
    float in = (rec1 & rec2).area();
    float un = rec2.area();
    return (double)(in/un);
}

int writeResult(int frame, byavs::TrackeObjectCPUs tracking_results)
{
    // std::string resultFileName;
    // resultFileName = "/home/lihy/multiple-object-tracking-cpp/build/MOT16-04.txt ";
    std::ofstream outfile("tracking_result_mending/MOT16/MOT16-13.txt", std::ios::app);
    //outfile.open(resultFileName);

    if (!outfile.is_open())
    {
        std::cerr << "Error: can not find file " <<std::endl;
    }
    //outfile.open("data.txt");
    if(outfile.is_open())
    {
        if(frame==1)
        {
            std::cout<<tracking_results.size()<<std::endl;
        }
        for(int i=0; i<tracking_results.size(); i++)
        {
            
            outfile << frame<<", "<<tracking_results[i].id<<", "
                <<tracking_results[i].box.topLeftX<<", "<<tracking_results[i].box.topLeftY<<", "<<tracking_results[i].box.width<<", "<<tracking_results[i].box.height<<", "
                <<"-1, "<<"-1, "<<"-1, "<<"-1, "<<"\n";
            
        }
        outfile.close();
    }
    
    return 0;
}
