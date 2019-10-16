#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "mht_tree.h"
#include "mht_graph.h"
#include "mht_tracker.h"
#include "detector.h"
#include "util.h"
#include "test.h"

int main() {

  // the fuction in comment line is used for testing

  //test_graph();
    
  //test_tree();
  
  //test_treeTograph();

  //test_gating();

  //test_detector_to_do();

  //test_detector_inference();

  //test_NMS();
  
  //test_writeResult();

  //test_all();

  //test_mwis();

  // run the sequence
  int N=10;
  //detection file
  std::string root = "/nfs-data/tracking/MOT16/train/";
  std::string seq = "MOT16-11";
  root = root + seq + "/";
  std::string imgPath = root + "img1/";
  std::string detPath = root + "det/det.txt";
  std::vector<std::string> files;
  listDir(imgPath.c_str(), files, true);
  sort(files.begin(), files.end());
  //result file
  std::string result_dir = "tracking_result_0925/";
  result_dir = result_dir + seq + "/";
  std::string txt_name = seq + ".txt";

  Detector detector;
  MHT_tracker tracker;
  detector.read_txt(detPath);

  std::vector<cv::Rect_<float>> det_result;
  std::vector<float> det_result_score;
  byavs::TrackeObjectCPUs tracking_results;

  std::string curr_img;
  cv::Mat img;
  int filesize = files.size();
  double start, end, duration;
  std::cout << "total frame:" << filesize<<std::endl;
  start = clock();
  for (int frame = 1; frame < filesize + N; frame++) {
      detector.inference(frame, det_result, det_result_score);
      std::cout << "frame:" << frame << " det_result size:" 
                << det_result.size()  << std::endl;
      
      tracker.inference(det_result, det_result_score, tracking_results);
      std::cout << "after inference, tracking_results size:" 
                << tracking_results.size() << std::endl;

      //save tracking results
      if (frame >= N) {
          curr_img = files[frame-N];
          img = cv::imread(curr_img);
          visualize(frame-N+1, img, tracking_results, result_dir);
          writeResult(frame-N+1,tracking_results, result_dir, txt_name);
      }
      
      det_result.clear();
      tracking_results.clear();
  }
  end = clock();
  duration = (double)(end - start)/CLOCKS_PER_SEC;
  std::cout << "Tracking time cost : " << duration <<" s " << std::endl;
  std::cout << "fps: " << filesize/duration << "frames/s" << std::endl;
}
