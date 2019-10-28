#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "sort_tracker.h"
#include "mht_tracker.h"
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
  * 4. Auto Factory
  * 
  * 5. Reading confige from `ini` file
  * 
  * 6. Run test can output the banchmark result
  * 
  * 7. Owner
  */

  int N=10;
  bool visualization = false;
  bool save_txt = true;
  //detection file
  std::string root = "/nfs-data/tracking/MOT16/train/";
  std::vector<std::string> sequence{"MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"};
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
    std::string txt_name = seq + ".txt";

    std::string command = "mkdir -p " + result_img_dir + "&&" + "mkdir -p " + result_txt_dir;
    system(command.c_str());

    Detector detector;
    //MHT_tracker tracker;
    SortTracker tracker;
    detector.read_txt(detPath);

    std::vector<cv::Rect_<float>> det_result;
    std::vector<float> det_result_score;
    byavs::TrackeObjectCPUs tracking_results;

    std::string curr_img;
    cv::Mat img;

    double start, end, duration;
    std::cout << "total frame:" << files.size() <<std::endl;
    start = clock();
    for (int frame = 1; frame < files.size(); frame++) {
        detector.inference(frame, det_result, det_result_score);
        
        tracker.inference(det_result, det_result_score, tracking_results);
        std::cout << "frame:" << frame  << " "
                  << " det_result size:" << " "
                  << det_result.size()    << " "
                  << "tracking_results size:" << " "
                  << tracking_results.size() << std::endl;

        //save tracking results
        if (visualization) {
          curr_img = files[frame];
          img = cv::imread(curr_img);
          visualize(frame, img, tracking_results, result_img_dir);
        }
        if (save_txt){
          writeResult(frame, tracking_results, result_txt_dir, txt_name);
        }
        
        det_result.clear();
        tracking_results.clear();
    }
    end = clock();
    duration = (double)(end - start)/CLOCKS_PER_SEC;
    std::cout << "Tracking time cost : " << duration <<" s " << std::endl;
    std::cout << "fps: " << files.size()/duration << "frames/s" << std::endl;
  }
  //google::ShutdownGoogleLogging();

}
