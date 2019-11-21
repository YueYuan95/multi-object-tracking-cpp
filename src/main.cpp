#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "sort_tracker.h"
#include "mht_tracker.h"
#include "multi_tracker.h"
#include "detector.h"
#include "util.h"
#include "test.h"

int main(int argc, char * argv[]) {

  //google::InitGoogleLogging(argv[0]);

  /*TODO: 
  *
  * 1. Because the tracking algorithm with time window need a special way to test,
  *    so we will using a class to do all the algorithm test, called `Test`, using a
  *    flag to distinguish with time window test or not.
  * 
  * 2. We will put all the file operation into `Detector` class and may add CNN detector
  *    in order to deal with video files.
  * 
  * 3. Show tracking result with colored bounding boxes and output the txt files. 
  * 
  * 4. Auto Factory, Abstract class
  * 
  * 5. Reading confige from `ini` file
  * 
  * 6. pass data to python program
  * 
  * 6. Better file struct (main, test)
  * 
  * 7. Run test can output the banchmark result
  * 
  * 8. Owner
  * 
  * 9. Docker
  * 
  */

  int N=10;
  bool visualization = false;
  bool save_txt = false;
  bool test_set = false;

  //detection file
  std::string root;     // "/nfs-data/tracking/MOT16/train/";
  std::vector<std::string> sequence; //{"MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"}

  //use `switch` is better 
  root = test_set ? "/root/tracking/MOT16/test/":"/root/tracking/MOT16/train/";
  if(test_set){
    sequence = {"MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", "MOT16-08", "MOT16-12", "MOT16-14"};
  }else{
    sequence = {"MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"};
  }

  double avg_fps = 0.00;

  sequence = {"MOT16-04"};

  for(int i=0; i < sequence.size(); i++){

    std::string seq = sequence[i];
    std::string seq_root = root + seq + "/";
    std::string imgPath = seq_root + "img1/";
    std::string detPath = seq_root + "det/det.txt";
    std::vector<std::string> files;
    listDir(imgPath.c_str(), files, true);
    std::sort(files.begin(), files.end());
    //result file
    std::string result_dir = "result/";
    std::string result_img_dir = result_dir + seq + "/";
    std::string result_txt_dir = result_dir + "txt/";
    if(test_set){
      result_txt_dir = result_txt_dir + "test/";
    }else{
      result_txt_dir = result_txt_dir + "train/";
    }
    std::string txt_name = seq + ".txt";

    std::string command = "mkdir -p " + result_img_dir + "&&" + "mkdir -p " + result_txt_dir;
    system(command.c_str());

    Detector detector;
    //MHT_tracker tracker;
    //SortTracker tracker;
    MultiTracker tracker;
    byavs::PedFeatureParas ped_feature_paras;
    int gpu_id = 0;
    std::string ped_model_dir = "/root/data";
    tracker.init(ped_feature_paras, ped_model_dir, gpu_id);

    detector.read_txt(detPath);

    std::vector<cv::Rect_<float>> det_result;
    std::vector<float> det_result_score;
    byavs::TrackeObjectCPUs tracking_results;

    std::string curr_img;
    cv::Mat img;

    byavs::TrackeInputGPU  inputs;
    byavs::TrackeObjectGPUs outputs;

    int file_size = files.size();
    file_size = 30;
    double start, end, duration, fps;
    for (int frame = 1; frame < file_size; frame++) {
        detector.inference(frame, det_result, det_result_score);
        convert_to_tracking_input(files[frame], det_result, det_result_score, inputs);
        start = clock();
        tracker.inference(inputs, outputs);
        end = clock();
        duration += (end - start);
        std::cout << "frame:" << frame  << " "
                  << "det_result size:" << " "
                  << det_result.size()    << ", "
                  << "tracking_results size:" << " "
                  << outputs.size() << ", "
                  << "cost time :"
                  << (double)(end - start)/ CLOCKS_PER_SEC * 1000 << " ms "
                  << std::endl;

        //save tracking results
        if (visualization) {
          curr_img = files[frame];
          img = cv::imread(curr_img);
          visualize(frame, img, outputs, result_img_dir);
        }
        if(save_txt){
          writeResult(frame, outputs, result_txt_dir, txt_name);
        }
        det_result.clear();
        det_result_score.clear();
        cudaFree(inputs.gpuImg.data);
        inputs.objs.clear();
        outputs.clear();
    }
    duration = (double)duration/CLOCKS_PER_SEC;
    fps = files.size()/duration;
    avg_fps += fps;
    std::cout<< "dataset is "<<seq<<" , its" << "fps is " << fps<< "frames/s" << std::endl;
  }
  avg_fps = avg_fps/sequence.size();
  std::cout<<"Test "<<sequence.size()<<" dataset, average FPS is "<<avg_fps<<"frames/s"<<std::endl;
  
  //google::ShutdownGoogleLogging();

}