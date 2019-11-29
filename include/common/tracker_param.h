#ifndef _TRACKER_PARAM_H_
#define _TRACKER_PARAM_H_

#include <iostream>
#include <cuda_runtime.h>

enum tracker_state {
    /*
    Enumeration type for the single target state.
    Newly created tracks are classified as `tentative` until enough evidence
    has been collected. Then, the track state is changed to `confirmed`. Trackers
    that are no longer alive are classified as `deleted` to mark them for removal
    from the set of active trackers.
    */
   Tentative = 1,
   Confirmed = 2,
   Deleted = 3
};

const int n_init = 3;
const int max_age = 30;
const double max_iou_distance = 0.3;
const int FEATURE_SIZE = 2048;

#endif