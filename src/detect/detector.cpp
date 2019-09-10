#include "detector.h"


int Detector::read_txt(std::string det_file){//string seq

    //Tracker
    /*Tracking tracker;
    string model_dir="/data/wuh/project/algorithm_module/data";
    TrackeParas track_param;
    tracker.init(model_dir,track_param,2);*/
//      MultipleTracker  tracker;

        //Detection File

    std::string detFileName;
    //imgPath = "/nfs-data/tracking/MOT16/train/"+seq+"/img1/";
    //detFileName = "/nfs-data/tracking/MOT16/train/"+seq+"/det/det.txt";
    detFileName = det_file;

    // read detection file and put the box into a dictionary
    std::ifstream detectionFile;//read file from th detectionFile;
    detectionFile.open(detFileName);

    if (!detectionFile.is_open())
    {
        std::cerr << "Error: can not find file " << detFileName <<std::endl;
    }

    std::string detLine;
    //istringstream ss;//??
    //std::vector<cv::Rect_<float>> detBox;//detBox?
    std::vector<std::string> split;//firstly save in a string vector split and then change the split's element into float 
    //char ch;
    int frame;
    float tp_tl, tp_tr, w, h;
    float score;
    cv::Rect_<float> box;
    //std::map<int, std::vector<cv::Rect_<float>> frame_det_map;//a dictionary containing the boxes of each frame

    std::cout<<"Get Detection Groundtruth"<<std::endl;
    while (getline(detectionFile, detLine))//end reading when meeting endl,read a line
    {
        //cout << detLine << endl; map<int, std::vector<cv::Rect_<float>>
                //TrackingBox tb;//key is frame number,push_back box 
        split_string(detLine, split, ",");
        frame = std::atof(split[0].c_str());
                //tb.id = 0;
        tp_tl = std::atof(split[2].c_str());//atof:string to float
        tp_tr = std::atof(split[3].c_str());
        w = std::atof(split[4].c_str());
        h = std::atof(split[5].c_str());
        box = cv::Rect_<float>(cv::Point_<float>(tp_tl, tp_tr), cv::Point_<float>(tp_tl+w, tp_tr +h));
        score = std::atof(split[6].c_str());
        if(score > 0.2 ){
            Detector::frame_det_map[frame].push_back(box);
            Detector::frame_det_score[frame].push_back(score);
        }
        
                /*cout<<"Frame:"<<tb.frame << "," << tp_tl << "," << tp_tr << "," << tp_bb
l << "," << h << "," << tb.score << endl;*/
        //detBox.push_back(box);//push a box

        split.clear();
    }
    detectionFile.close();
}

void Detector::split_string(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

int Detector::inference(int frame, std::vector<cv::Rect_<float>>& destination, std::vector<float>& destination_score)
{
    int i;
    cv::Rect_<float> zero_box = cv::Rect(0,0,0,0);
    if(frame<=frame_det_map.size())
    {
     
        for(i=0; i<frame_det_map[frame].size(); i++)
        {
            destination.push_back(frame_det_map[frame][i]);  //box
            destination_score.push_back(frame_det_score[frame][i]);//score
            //std::cout<<destination[i];
        }
    //std::cout<<std::endl;
    //std::cout<<frame_det_map[1050][0];
    }
    // else
    // {
    //     destination.push_back(zero_box);  //box
    //     destination_score.push_back(0);//score
    // }
    
}
