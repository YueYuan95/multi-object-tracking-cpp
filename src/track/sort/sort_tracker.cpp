#include "sort_tracker.h"

int SORT_tracker::inference(byavs::GpuImg image){

    std::vector<int> activated_trackers;
    std::vector<int> refind_trackers;
    std::vector<int> lost_trackers;
    std::vector<int> removed_trackers;

    /*
     * Step 1 : join the activate tracker and lost trackers
     *          generate the candidate tracker
     * */
    std::vector<int> unconfirmed_trackers;
    std::vector<int> tracked_trackers;
    unconfirmed_trackers.clear();
    tracked_trackers.clear();
    for(auto i : m_trackers.size()){
        if(m_trackers[i].IsActivate()){
            tracked_trackers.push_back(i);
        }else{
            unconfirmed_trackers.push_back(i);
        }
    }
    
    std::vector<int> candidate_trackers;
    candidate_trackers.clear();

    generate_candidate_trackers(candidate_trackers, tracked_trackers, m_lost_trackers);

    /*
     * Step 2 : tracker predict
     *
     * */
    for(int i=0; i < candidate_trackers.size(); i++){
        m_tracker[candidate[i]].predict();
    }

    std::vector<cv::Rect_<float>> predict_boxs;
    predict_boxs.clear();
    for(int i=0; i<candidate.size();i++){
        predict_boxs.push_back(m_tracker[candidate[i]].getBox());
    }

    /*
     * Step 3 : if using deep feature, compute the feature distacne
     *          otherwise only iou cost will be compute
     * */

    // deep feature match
    std::vector<std::vector<float>> det_feature;
    
    compute_detection_feature(image, det_feature);

    std::vector<int> matches;
    std::vector<int> um_tracker;
    std::vector<int> um_detection;

    matches.clear();
    um_tracker.clear();
    um_detection.clear();
    if(USE_DEEP){
        std::vector<std::vector<float> feature_cost_matrix;
        feature_cost_matrix.clear();

        compute_feature_distance(feature_cost_matrix, candidate, det_feature);
        matching(feature_cost_matrix, matches, um_tracker, um_detection);
        
        for(int i=0; i < matches.size(); i++){
            if(m_tracker[candidate[i]].getState() == STATE_TRACKED){
                m_tracker[candidate[i]].update(detection_boxs[matches[i]]);
                activated_trackers.push_back(candidate[i]);
            }else{
                m_tracker[candidate[i]].re_activate(detection_boxs[matches[i]]);
                refind_trackers.push_back(candidate[i]);
            }
        }

    }
    
    // iou match
    std::vector<cv::Rect_<float>> detection_boxs;
    
    std::vector<std::vector<float>> iou_cost_matrix;
    iou_cost_matrix.clear();
    
    matches.clear();
    um_tracker.clear();
    um_detecion.clear();

    compute_iou_distance(iou_cost_matrix, matches, um_tracker, um_detection);
    matching(iou_cost_matrix, matches, um_tracker, um_detecion);

    for(int i=0; i < matches.size(); i++){
        if(m_tracker[candidate[i]].getState() == STATE_TRACKED){
            m_tracker[candidate[i]].update(detection_boxs[matches[i]]);
            activated_trackers.push_back(candidate[i]);
        }else{
            m_tracker[candidate[i]].re_activate(detection_boxs[matches[i]]);
            refind_trackers.push_back(candidate[i]);
        }
    }

}
