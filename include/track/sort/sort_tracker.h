#ifndef __SORT_TRACKER_H_
#define __SORT_TRACKER_H_

#include <iostream>
#include <math.h>

#include "util.h"
#include "byavs.h"

class SORT_tracker{

    public:
        int inference(const std::string& model_dir, const byavs::TrackeParas& pas,                      const int gpu_id);
        int inference();
    
    private:
        
        std::vector<Tracker> m_tracked_trackers;
        std::vector<Tracker> m_lost_trackers;
        std::vector<Tracker> m_removed_trackers;

        int generate_candidate_trackers();
        int compute_apperance_distance();
        int compute_iou_distance();
        int predict();
        int match();
        int init_new_tracker();
        int update_state();
};

#endif
