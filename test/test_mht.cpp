/***************************************************************************
Copyright(C)：AVS
  *FileName:  // multiple-object-tracking-cpp/common
  *Author:  // Li Haoying
  *Version:  // 2
  *Date:  //2019-10-16
  *Description:  //*The following functions are to test functions in 
                    every chapter
****************************************************************************/

#include "test.h"

int test_graph() {
    std::vector<VexNode> vex_node_list;
    VexNode temp_node;

    temp_node = {0.0, 0, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 1, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 2, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 3, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {2.18402e-09, 4, {2,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.1, 5, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 6, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 7, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 6, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 7, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 8, {-1,0,2,0,0,4,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 9, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 10, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {0.0, 12, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.0, 13, {0,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {0.01, 14, {1,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 15, {3,0,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 16, {4,3,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 17, {0,1,1,1,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 18, {0,2,0,0,0,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 19, {0,0,0,0,1,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 20, {0,0,0,0,2,3,3,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 21, {0,0,0,0,3,0,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 22, {0,0,0,0,0,1,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 23, {0,0,0,0,0,2,0,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 24, {0,0,0,0,0,0,1,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 25, {0,0,0,0,0,0,2,0,0,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 26, {0,0,0,0,0,0,0,1,1,0}};
    vex_node_list.push_back(temp_node);

    temp_node = {0.01, 27, {0,0,0,0,0,0,0,0,0,1}};
    vex_node_list.push_back(temp_node);

    
    Graph A(vex_node_list);
    A.printGraph();
    std::map<int, std::vector<int>> routes;
    A.mwis_greed(routes);
}

int test_treeTograph() {

    /*Fake Tree No.1*/
    treeNode root = {{10,9,8,7},6,1,1,NULL};
    std::shared_ptr<treeNode> root_ptr(new treeNode(root));

    Tree test_tree(root_ptr,1,2);

    treeNode node_a = {{10,9,8,7},6,2,3,root_ptr};
    treeNode node_b = {{10,9,8,7},6,2,2,root_ptr};

    std::shared_ptr<treeNode> node_a_ptr(new treeNode(node_a));
    std::shared_ptr<treeNode> node_b_ptr(new treeNode(node_b));
    
    treeNode node_c = {{10,9,8,7},6,3,4,node_a_ptr};
    treeNode node_d = {{10,9,8,7},6,3,2,node_a_ptr};
    treeNode node_e = {{10,9,8,7},6,3,1,node_b_ptr};
    treeNode node_f = {{10,9,8,7},6,3,3,node_b_ptr};
    
    std::shared_ptr<treeNode> node_c_ptr(new treeNode(node_c));
    std::shared_ptr<treeNode> node_d_ptr(new treeNode(node_d));
    std::shared_ptr<treeNode> node_e_ptr(new treeNode(node_e));
    std::shared_ptr<treeNode> node_f_ptr(new treeNode(node_f));
    

    std::vector<std::shared_ptr<treeNode>> node_list;
    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict;
    
    node_list.push_back(node_a_ptr);
    node_list.push_back(node_b_ptr);
    
    dict[0] = node_list;
    test_tree.addNode(dict);
    
    node_list.clear();
    node_list.push_back(node_c_ptr);
    node_list.push_back(node_d_ptr);
    dict[0] = node_list;

    node_list.clear();
    node_list.push_back(node_e_ptr);
    node_list.push_back(node_f_ptr);
    dict[1] = node_list;

    test_tree.addNode(dict);

    dict.clear();
   
    /*Fake Tree No.2*/
    treeNode root_2 = {{10,9,8,7},6,1,2,NULL};
    std::shared_ptr<treeNode> root_2_ptr(new treeNode(root_2));

    Tree test_tree_2(root_2_ptr,2,2);

    treeNode node_2_a = {{10,9,8,7},6,2,2,root_2_ptr};
    treeNode node_2_b = {{10,9,8,7},6,2,1,root_2_ptr};

    std::shared_ptr<treeNode> node_2_a_ptr(new treeNode(node_2_a));
    std::shared_ptr<treeNode> node_2_b_ptr(new treeNode(node_2_b));
    
    treeNode node_2_c = {{10,9,8,7},6,3,1,node_2_a_ptr};
    treeNode node_2_d = {{10,9,8,7},6,3,3,node_2_a_ptr};
    treeNode node_2_e = {{10,9,8,7},6,3,0,node_2_b_ptr};
    
    std::shared_ptr<treeNode> node_2_c_ptr(new treeNode(node_2_c));
    std::shared_ptr<treeNode> node_2_d_ptr(new treeNode(node_2_d));
    std::shared_ptr<treeNode> node_2_e_ptr(new treeNode(node_2_e));

    node_list.clear();
    
    node_list.push_back(node_2_a_ptr);
    node_list.push_back(node_2_b_ptr);
    
    dict[0] = node_list;
    test_tree_2.addNode(dict);
    
    node_list.clear();
    node_list.push_back(node_2_c_ptr);
    node_list.push_back(node_2_d_ptr);
    dict[0] = node_list;

    node_list.clear();
    node_list.push_back(node_2_e_ptr);
    dict[1] = node_list;

    test_tree_2.addNode(dict);
    
    dict.clear();
    /*Fake Tree No.3*/
    treeNode root_3 = {{10,9,8,7},6,1,5,NULL};
    std::shared_ptr<treeNode> root_3_ptr(new treeNode(root_3));

    Tree test_tree_3(root_3_ptr,3,2);
    
    treeNode node_3_a = {{10,9,8,7},6,2,2,root_3_ptr};
    treeNode node_3_b = {{10,9,8,7},6,2,1,root_3_ptr};

    std::shared_ptr<treeNode> node_3_a_ptr(new treeNode(node_3_a));
    std::shared_ptr<treeNode> node_3_b_ptr(new treeNode(node_3_b));
    
    node_list.clear();
    node_list.push_back(node_3_a_ptr);
    node_list.push_back(node_3_b_ptr);
    
    dict[0] = node_list;
    test_tree_3.addNode(dict);
    

    //map<int, vector<std::shared_ptr<treeNode>>>:: iterator it;
    //dict.insert(pair<int, std::vector<std::shared_ptr<treeNode>>> 1,testTree);
    //test_tree.printTree(root_ptr);
    //test_tree(root1,1,3);
    //printf("")
    // std::cout<<"id:"<<test_tree.getId()<<std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    // std::cout<<"leaf_node:"<<&(test_tree.getLeafNode()[0])<<std::endl;
    // std::cout<<"head_node:"<<test_tree.getHead()<<std::endl;
    // std::cout<<"box:"<<test_tree.getRoot()->box<<std::endl;
    //test_tree.printTree(root_ptr);
    std::vector<Tree> tree_list;
    tree_list.push_back(test_tree);
    tree_list.push_back(test_tree_2);
    tree_list.push_back(test_tree_3);

    std::map<int, std::vector<int>> routes;
    routes.clear();
    Graph graph;
    TreeToGraph(tree_list, graph);
    graph.printGraph();
    graph.mwis(routes);
    for (int j=0; j < tree_list.size(); j++) {
        if (routes.count(tree_list[j].getId())) {
            tree_list[j].pruning(routes[tree_list[j].getId()]);
            tree_list[j].printTree(tree_list[j].getRoot());
        }   
    }
    for (int i=0; i < tree_list.size(); i++) {
        cv::Rect_<float> result_vector;
        if (tree_list[i].sentResult(result_vector)) {
            std::cout << "ID is " << tree_list[i].getId() << ", Result is "
                        << result_vector << std::endl;
        }
    }
    for (int i=0; i < tree_list.size(); i++) {
        cv::Rect_<float> result_vector;
        if (tree_list[i].sentResult(routes[tree_list[i].getId()], result_vector)) {
            std::cout << "ID is "<< tree_list[i].getId() << ", Result is "
                        << result_vector << std::endl;
        }
    }
  
}

int test_tree() {

    treeNode root = {{10,9,8,7},6,1,1,NULL};
    std::shared_ptr<treeNode> root_ptr(new treeNode(root));

    Tree test_tree(root_ptr,1,3);

    treeNode node_a = {{10,9,8,7},6,2,3,root_ptr};
    treeNode node_b = {{10,9,8,7},6,2,2,root_ptr};

    std::shared_ptr<treeNode> node_a_ptr(new treeNode(node_a));
    std::shared_ptr<treeNode> node_b_ptr(new treeNode(node_b));
    
    treeNode node_c = {{10,9,8,7},6,3,4,node_a_ptr};
    treeNode node_d = {{10,9,8,7},6,3,2,node_a_ptr};
    treeNode node_e = {{10,9,8,7},6,3,1,node_b_ptr};
    treeNode node_f = {{10,9,8,7},6,3,3,node_b_ptr};
    
    std::shared_ptr<treeNode> node_c_ptr(new treeNode(node_c));
    std::shared_ptr<treeNode> node_d_ptr(new treeNode(node_d));
    std::shared_ptr<treeNode> node_e_ptr(new treeNode(node_e));
    std::shared_ptr<treeNode> node_f_ptr(new treeNode(node_f));
    

    std::vector<std::shared_ptr<treeNode>> node_list;
    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict;
    
    node_list.push_back(node_a_ptr);
    node_list.push_back(node_b_ptr);
    
    dict[0] = node_list;
    test_tree.addNode(dict);
    
    node_list.clear();
    node_list.push_back(node_c_ptr);
    node_list.push_back(node_d_ptr);
    dict[0] = node_list;

    node_list.clear();
    node_list.push_back(node_e_ptr);
    node_list.push_back(node_f_ptr);
    dict[1] = node_list;

    test_tree.addNode(dict);

    //dict.clear();

    std::cout << "id:" << test_tree.getId() << std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    //std::cout<<"root_node:"<<test_tree.getRoot()<<std::endl;
    // std::cout<<"leaf_node:"<<&(test_tree.getLeafNode()[0])<<std::endl;
    //std::cout<<"head_node:"<<test_tree.getHead()<<std::endl;
    // std::cout<<"box:"<<test_tree.getRoot()->box<<std::endl;
    /*if(test_tree.getRoot()==test_tree.getHead()){
        std::cout<<"yes"<<std::endl;
    }*/
    
    
    //test_tree.addNode(dict);
    //test_tree.addNode(dict);
    test_tree.printTree(root_ptr);

    //create a route:1-2-1
    std::vector<int> route_list;
    route_list.push_back(1);
    route_list.push_back(2);
    route_list.push_back(1);

    test_tree.pruning(route_list);
    test_tree.printTree(root_ptr);
}

int test_gating()
{
    //fake det_result:
    std::vector<cv::Rect_<float>> det_result;
    cv::Rect_<float> box1 = cv::Rect(100,110,120,130);
    cv::Rect_<float> box2 = cv::Rect(100,110,140,150);
    cv::Rect_<float> box3 = cv::Rect(500,510,160,170);
    cv::Rect_<float> box4 = cv::Rect(100,110,180,190);

    det_result.push_back(box1);
    det_result.push_back(box2);
    det_result.push_back(box3);
    det_result.push_back(box4);

    //true det_result
    /*std::vector<cv::Rect_<float>> det_result;
    int frame = 1050;
    //std::vector<cv::Rect_<float>> destination;//random
    Detector detector;
    detector.read_txt();
    detector.inference(frame, det_result);*/

    //std::vector<cv::Rect_<float>> temp_det_result;
    //for(int i=0; i < 30; i++){
    //    temp_det_result.push_back(det_result[i]);
    //}
    //det_result = temp_det_result;
    /*std::cout<<frame<<" ";
    int i;
    for(i=0;i<det_result.size();i++)
    {
        std::cout<<det_result[i];
    }
    std::cout<<std::endl;*/

    //create trees
    std::vector<Tree> tree_list;
    //tree No.1
    treeNode root = {{100,90,80,70},6,1,1,NULL};
    std::shared_ptr<treeNode> root_ptr(new treeNode(root));
   
    Tree test(root_ptr,1,3);

    treeNode node_a = {{100,90,85,75},6,2,3,root_ptr};
    treeNode node_b = {{100,90,85,70},6,2,2,root_ptr};

    std::shared_ptr<treeNode> node_a_ptr(new treeNode(node_a));
    std::shared_ptr<treeNode> node_b_ptr(new treeNode(node_b));
    
    treeNode node_c = {{100,90,100,130},6,3,4,node_a_ptr};
    treeNode node_d = {{100,110,120,130},6,3,2,node_a_ptr};
    treeNode node_e = {{100,110,120,130},6,3,2,node_b_ptr};
    treeNode node_f = {{100,90,120,110},6,3,3,node_b_ptr};
    
    std::shared_ptr<treeNode> node_c_ptr(new treeNode(node_c));
    std::shared_ptr<treeNode> node_d_ptr(new treeNode(node_d));
    std::shared_ptr<treeNode> node_e_ptr(new treeNode(node_e));
    std::shared_ptr<treeNode> node_f_ptr(new treeNode(node_f));
    

    std::vector<std::shared_ptr<treeNode>> node_list;
    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict;
    
    node_list.push_back(node_a_ptr);
    node_list.push_back(node_b_ptr);
    
    dict[0] = node_list;
    test.addNode(dict);
    
    node_list.clear();
    node_list.push_back(node_c_ptr);
    node_list.push_back(node_d_ptr);
    dict[0] = node_list;

    node_list.clear();
    node_list.push_back(node_e_ptr);
    node_list.push_back(node_f_ptr);
    dict[1] = node_list;

    test.addNode(dict);
    test.printTree(root_ptr);

    //tree No.2,100->110
    treeNode root2 = {{110,90,80,70},6,1,2,NULL};//1
    std::shared_ptr<treeNode> root_ptr2(new treeNode(root2));

    Tree test2(root_ptr2,1,3);

    treeNode node_a2 = {{110,90,85,75},6,2,3,root_ptr2};
    treeNode node_b2 = {{110,90,85,70},6,2,2,root_ptr2};

    std::shared_ptr<treeNode> node_a_ptr2(new treeNode(node_a2));
    std::shared_ptr<treeNode> node_b_ptr2(new treeNode(node_b2));
    
    treeNode node_c2 = {{110,90,100,130},6,3,4,node_a_ptr2};
    treeNode node_d2 = {{100,110,140,150},6,3,2,node_a_ptr2};
    treeNode node_e2 = {{100,110,140,150},6,3,2,node_b_ptr2};
    treeNode node_f2 = {{110,90,120,110},6,3,3,node_b_ptr2};
    
    std::shared_ptr<treeNode> node_c_ptr2(new treeNode(node_c2));
    std::shared_ptr<treeNode> node_d_ptr2(new treeNode(node_d2));
    std::shared_ptr<treeNode> node_e_ptr2(new treeNode(node_e2));
    std::shared_ptr<treeNode> node_f_ptr2(new treeNode(node_f2));
    

    std::vector<std::shared_ptr<treeNode>> node_list2;
    std::map<int, std::vector<std::shared_ptr<treeNode>>> dict2;
    
    node_list2.push_back(node_a_ptr2);
    node_list2.push_back(node_b_ptr2);
    
    dict2[0] = node_list2;
    test2.addNode(dict2);
    
    node_list2.clear();
    node_list2.push_back(node_c_ptr2);
    node_list2.push_back(node_d_ptr2);
    dict2[0] = node_list2;

    node_list2.clear();
    node_list2.push_back(node_e_ptr2);
    node_list2.push_back(node_f_ptr2);
    dict2[1] = node_list2;

    test2.addNode(dict2);
    test2.printTree(root_ptr2);

    tree_list.push_back(test);
    tree_list.push_back(test2);

    MHT_tracker test_tracker;
    //test_tracker.gating(det_result,tree_list);

    int i;
    std::cout << "After Gating :" << std::endl;
    for (i=0; i<tree_list.size(); i++) {
        tree_list[i].printTree(tree_list[i].getRoot());
        std::cout<<std::endl;
    }
    
    std::cout << "" << std::endl;
    for (i=0; i<tree_list.size(); i++) {
        for (auto iter :tree_list[i].getLeafNode()) {
            std::cout<< iter->index ;
        }
        std::cout<<std::endl;
    }

    //test graph
    std::map<int, std::vector<int>> routes;
    routes.clear();
    Graph graph;
    TreeToGraph(tree_list, graph);
    graph.printGraph();
    graph.mwis(routes);
    for (int j = 0; j < tree_list.size(); j++) {
        if (routes.count(tree_list[j].getId())) {
            tree_list[j].pruning(routes[tree_list[j].getId()]);
            tree_list[j].printTree(tree_list[j].getRoot());
        }   
    }
    for (i=0; i < tree_list.size(); i++) {
        cv::Rect_<float> result_vector;
        if (tree_list[i].sentResult(result_vector)) {
            std::cout << "ID is "<< tree_list[i].getId() << ", Result is "
                        << result_vector<<std::endl;
        }
    }
    for (i=0; i < tree_list.size(); i++) {
        cv::Rect_<float> result_vector;
        if (tree_list[i].sentResult(routes[tree_list[i].getId()], 
            result_vector)) {
            std::cout << "ID is " << tree_list[i].getId() << ", Result is "
                        << result_vector<<std::endl;
        }
    }
}

int test_read_txt()
{
    
    std::string root = "/nfs-data/tracking/MOT16/train/";
    std::string seq = "MOT16-07";
    root = root + seq + "/";
    std::string imgPath = root + "img1/";
    std::string detPath = root + "det/det.txt";

    Detector detector;
    detector.read_txt(detPath);

    int i;
    std::map<int, std::vector<cv::Rect_<float>>>::iterator it;
    for (it = detector.frame_det_map.begin(); it != detector.frame_det_map.end();
         it++) {
        auto value1 = it->first;
        auto value2 = it->second;
        std::cout << value1 <<" ";
        for (i=0; i<value2.size(); i++) {
            std::cout << value2[i];
        }
        std::cout<<std::endl;
    }
    

}

int test_detector_inference() {
    int frame = 1050;
    std::vector<cv::Rect_<float>> destination;
    std::vector<float> destination_score;

    std::string root = "/nfs-data/tracking/MOT16/train/";
    std::string seq = "MOT16-07";
    root = root + seq + "/";
    std::string imgPath = root + "img1/";
    std::string detPath = root + "det/det.txt";

    Detector detector;
    detector.read_txt(detPath);

    detector.inference(frame, destination, destination_score);
    
    std::cout << frame << " ";
    int i;
    for(i = 0; i < destination.size(); i++) {
        std::cout << destination[i];
        std::cout << " score " << destination_score[i];
    }
    std::cout<<std::endl;
}

int test_NMS() {
    int N=5;

    std::string root = "/nfs-data/tracking/MOT16/train/";
    std::string seq = "MOT16-09";
    root = root + seq + "/";
    std::string imgPath = root + "img1/";
    std::string detPath = root + "det/det.txt";

    Detector detector;
    detector.read_txt(detPath);

    MHT_tracker tracker;
    std::vector<cv::Rect_<float>> det_result;
    std::vector<float> det_result_score;
    std::vector<cv::Rect_<float>> det_result_selected;

    byavs::TrackeObjectCPUs tracking_results;

    std::vector<std::string> files;
    listDir(imgPath.c_str(), files, true);
    sort(files.begin(), files.end());

    std::string curr_img;
    cv::Mat img;

    for(int frame = 1; frame < 30; frame++)
    {
        detector.inference(frame, det_result, det_result_score);
        std::cout << "frame:" << frame << " det_result size:" << det_result.size()
                    << std::endl;
        
        det_result_selected = tracker.NMS(det_result, det_result_score);
        for (int j = 0; j < det_result_selected.size(); j++) {
            std::cout << " det_selected " << j+1 << " box:" << det_result_selected[j] 
                        << std::endl;
        }
        std::cout << std::endl;
        det_result.clear();
        det_result_score.clear();
    }
}

int test_all() {
    int N=10;
    //detection file
    int filelength;
    std::string root = "/nfs-data/tracking/MOT16/train/";
    std::string seq = "MOT16-13";
    root = root + seq + "/";
    std::string imgPath = root + "img1/";
    std::string detPath = root + "det/det.txt";
    std::vector<std::string> files;
    listDir(imgPath.c_str(), files, true);
    sort(files.begin(), files.end());

    Detector detector;
    MHT_tracker tracker;
    detector.read_txt(detPath);

    std::vector<cv::Rect_<float>> det_result;
    std::vector<float> det_result_score;
    byavs::TrackeObjectCPUs tracking_results;

    std::string curr_img;
    cv::Mat img;
    int filesize = files.size();
    std::cout << "total frame:" << filesize << std::endl;
    for (int frame =1; frame < filesize + N; frame++) {
        detector.inference(frame, det_result, det_result_score);
        std::cout << "frame:" << frame << " det_result size:" << det_result.size()
                    <<std::endl;

        tracker.inference(det_result, det_result_score, tracking_results);
        std::cout << "after inference, tracking_results size:" << tracking_results.size()
                    << std::endl;

        //save tracking results
        if (frame >= N) {
            curr_img = files[frame-N];
            img = cv::imread(curr_img);
            visualize(frame-N+1, img, tracking_results);
            //writeResult(frame-N+1,tracking_results);
        }
        
        det_result.clear();
        tracking_results.clear();
    }
}

int test_mwis(){

    int N=10;
    int count_n = 0;
    int filelength;
    std::string root = "/nfs-data/tracking/MOT16/train/";
    std::string seq = "MOT16-04";
    root = root + seq + "/";
    std::string imgPath = root + "img1/";
    std::string detPath = root + "det/det.txt";

    Detector detector;
    MHT_tracker tracker;
    detector.read_txt(detPath);

    std::vector<std::string> files;
    listDir(imgPath.c_str(), files, true);
    sort(files.begin(), files.end()); 
}

int test_km(){

    int row = 30;
    int col = 40;
    std::vector<int> assign;
    assign.clear();
    std::vector<std::vector<double>> cost_matrix_hung;
    cost_matrix_hung.resize(row, std::vector<double>(col, 0.0));
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            cost_matrix_hung[i][j] = rand()%101/(double)101;;
            //std::cout<<cost_matrix_hung[i][j]<<" ";
        }
        //std::cout<<std::endl;
    }

    HungarianAlgorithm algorithm;
    algorithm.Solve(cost_matrix_hung, assign);
    for(auto i : assign){
        std::cout<<i<<","<<assign[i]<<std::endl;
    }

    std::cout<<"HungarianAlgorithm is over"<<std::endl;
    
    assign.clear();
    std::vector<std::vector<float>> cost_matrix;
    cost_matrix.resize(row, std::vector<float>(col, 0.0));
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            cost_matrix[i][j] = rand()%101/(float)101;;
            //std::cout<<cost_matrix[i][j]<<" ";
        }
        //std::cout<<std::endl;
    }
    
    KM km_algorithm;
    km_algorithm.solve(cost_matrix, assign);

    for(auto i : assign){
        std::cout<<i<<","<<assign[i]<<std::endl;
    }
}