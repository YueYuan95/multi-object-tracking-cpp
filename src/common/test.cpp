#include "test.h"
/* *===================================
 * Test Unit 
 *
 *===================================
*/

int test_graph(){
    std::vector<VexNode> vex_node_list;
    VexNode temp_node = {1.0, 1, {1,3,4}};
    vex_node_list.push_back(temp_node); 
    temp_node = {1.0, 1, {1,3,2}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {22.0, 1, {1,2,1}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, 1, {1,2,3}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, 2, {2,2,1}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {4.0, 2, {2,2,3}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, 2, {2,1,0}};
    vex_node_list.push_back(temp_node);
    
    temp_node = {1.0, 3, {0,0,5}};
    vex_node_list.push_back(temp_node);
    
    Graph A(vex_node_list);
    A.printGraph();
    std::map<int, std::vector<int>> routes;
    A.mwis(routes);
}

int test_treeTograph(){

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
    for(int j=0; j < tree_list.size(); j++){
        if(routes.count(tree_list[j].getId())){
            tree_list[j].pruning(routes[tree_list[j].getId()]);
            tree_list[j].printTree(tree_list[j].getRoot());
        }   
    }
    for(int i=0; i < tree_list.size(); i++){
        cv::Rect_<float> result_vector;
        if(tree_list[i].sentResult(result_vector)){
            std::cout<<"ID is "<< tree_list[i].getId() << ", Result is "<< result_vector<<std::endl;
        }
    }
    for(int i=0; i < tree_list.size(); i++){
        cv::Rect_<float> result_vector;
        if(tree_list[i].sentResult(routes[tree_list[i].getId()],result_vector)){
            std::cout<<"ID is "<< tree_list[i].getId() << ", Result is "<< result_vector<<std::endl;
        }
    }
  
}


int test_tree(){

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

    std::cout<<"id:"<<test_tree.getId()<<std::endl;
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
    cv::Rect_<float> box1 = cv::Rect(100,110,120,130);//
    cv::Rect_<float> box2 = cv::Rect(100,110,140,150);//will match
    cv::Rect_<float> box3 = cv::Rect(500,510,160,170);//will not match
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
    std::cout<<std::endl;*/////////////

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
    
    treeNode node_c = {{100,90,100,130},6,3,4,node_a_ptr};//100,90,100,130//283.84, 125.45, 55.569, 168.71
    treeNode node_d = {{100,110,120,130},6,3,2,node_a_ptr};//100,110,140,150//369, 513, 79, 239
    treeNode node_e = {{100,110,120,130},6,3,2,node_b_ptr};//2//100,110,160,170
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
    
    treeNode node_c2 = {{110,90,100,130},6,3,4,node_a_ptr2};//
    treeNode node_d2 = {{100,110,140,150},6,3,2,node_a_ptr2};//110,100,140,150//858.74, 234.93, 63.98, 193.94
    treeNode node_e2 = {{100,110,140,150},6,3,2,node_b_ptr2};//1//110,110,160,170//292.02, 340.52, 59.629, 180.89
    treeNode node_f2 = {{110,90,120,110},6,3,3,node_b_ptr2};//
    
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
    std::cout<<"After Gating :"<<std::endl;
    for(i=0; i<tree_list.size(); i++)
    {
        tree_list[i].printTree(tree_list[i].getRoot());
        std::cout<<std::endl;
    }
    
    std::cout<<""<<std::endl;
    for(i=0; i<tree_list.size(); i++)
    {
        for(auto iter :tree_list[i].getLeafNode()){
            std::cout<< iter->index ;
        }
        std::cout<<std::endl;
    }

    /*std::vector<Tree> tree_list;
    tree_list.push_back(test_tree);
    tree_list.push_back(test_tree_2);
    tree_list.push_back(test_tree_3);*/
    
    //test graph
    std::map<int, std::vector<int>> routes;
    routes.clear();
    Graph graph;
    TreeToGraph(tree_list, graph);
    graph.printGraph();
    graph.mwis(routes);
    for(int j=0; j < tree_list.size(); j++){
        if(routes.count(tree_list[j].getId())){
            tree_list[j].pruning(routes[tree_list[j].getId()]);
            tree_list[j].printTree(tree_list[j].getRoot());
        }   
    }
    for(i=0; i < tree_list.size(); i++){
        cv::Rect_<float> result_vector;
        if(tree_list[i].sentResult(result_vector)){
            std::cout<<"ID is "<< tree_list[i].getId() << ", Result is "<< result_vector<<std::endl;
        }
    }
    for(i=0; i < tree_list.size(); i++){
        cv::Rect_<float> result_vector;
        if(tree_list[i].sentResult(routes[tree_list[i].getId()],result_vector)){
            std::cout<<"ID is "<< tree_list[i].getId() << ", Result is "<< result_vector<<std::endl;
        }
    }

    //test tracker


}

int test_read_txt()
{
    Detector detector;
    detector.read_txt();
    int i;
    std::map<int, std::vector<cv::Rect_<float>>>::iterator it;
    for (it=detector.frame_det_map.begin(); it!=detector.frame_det_map.end();it++)
    {
        auto value1 = it->first;
        auto value2 = it->second;
        std::cout<<value1<<" ";
        for(i=0; i<value2.size(); i++)
        {
            std::cout<<value2[i];
        }
        std::cout<<std::endl;
    }
    

}

int test_detector_inference()
{
    int frame = 1050;
    std::vector<cv::Rect_<float>> destination;//random
    //destination.push_back(cv::Rect(1,2,3,4));
    //destination.clear();

    Detector detector;
    detector.read_txt();
    detector.inference(frame, destination);
    
    //std::cout<<destination.size()<<std::endl;
    std::cout<<frame<<" ";
    int i;
    for(i=0;i<destination.size();i++)
    {
        std::cout<<destination[i];
    }
    std::cout<<std::endl;

}

/*int test_visualize()
{

}*/

int test_all()
{   
    
    Detector detector;
    MHT_tracker tracker;
    detector.read_txt();

    std::vector<cv::Rect_<float>> det_result;
    /*detector.inference(2, det_result);
    for(int i=0;i<det_result.size();i++)
    {
        std::cout<<det_result[i];
    }
    std::cout<<std::endl;*/

    byavs::TrackeObjectCPUs tracking_results;

    std::string imgPath;
    imgPath = "/nfs-data/tracking/MOT16/train/MOT16-04/img1/";
    std::vector<std::string> files;
    listDir(imgPath.c_str(), files, true);
    sort(files.begin(), files.end());

    std::string curr_img;
    cv::Mat img;

    for(int frame =1 ;frame<files.size(); frame++)//files.size()
    {
        detector.inference(frame, det_result);

        std::cout<<"frame:"<<frame<<" det_result size:"<<det_result.size()<<std::endl;
        //for(int j=0;j<det_result.size();j++)
        //{
        //    std::cout<<det_result[j];
        //}
        //std::cout<<std::endl;

        tracker.inference(det_result, tracking_results);

        //std::cout<<"after gating:"<<std::endl;
        //for(int i=0;i<tracker.get_tree_list().size();i++)
        //{
        //   tracker.get_tree_list()[i].printTree(tracker.get_tree_list()[i].getRoot());
        //   std::cout<<std::endl;
        //}

        //std::cout<<"current leaf_node:"<<std::endl;
        //for(int i=0; i<tracker.get_tree_list().size(); i++)
        //{
        //    for(auto iter :tracker.get_tree_list()[i].getLeafNode())
        //    {
        //    std::cout<< iter->index ;
        //    }
        //std::cout<<std::endl;
        //}

        if(frame >= 3)
        {
            //std::cout<<" "<<tracking_results.size()<<std::endl;
            curr_img = files[frame-2-1];
            img = cv::imread(curr_img);
            visualize(frame-3, img, tracking_results);
        }
     
        // curr_img = files[frame-1];
        // //curr_img = files[frame];
        // img = cv::imread(curr_img);
        // visualize(frame, img, tracking_results);
    
       
        det_result.clear();
        tracking_results.clear();

    }
    
    


    // {
    //     test_tracker.get_tree_list()[i].printTree(test_tracker.get_tree_list()[i].getRoot());
    //     std::cout<<std::endl;
    // }
    
    // std::cout<<""<<std::endl;
    // for(i=0; i<test_tracker.get_tree_list().size(); i++)
    // {
    //     for(auto iter :test_tracker.get_tree_list()[i].getLeafNode()){
    //         std::cout<< iter->index ;
    //     }
    //     std::cout<<std::endl;
    //}
    //-----------------------------------------------------
    //visualize the results
    
}
