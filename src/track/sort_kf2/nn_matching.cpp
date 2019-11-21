#include "nn_matching.h"

int DistanceMetric::partial_fit(std::vector<std::vector<float>> feature_list, 
                std::vector<int> tracker_id){

   assert(feature_list.size() == tracker_id.size());
//    debug<<"fit begin"<<debugend;
   for(int i=0; i<tracker_id.size(); i++){
       if(m_samples.count(tracker_id[i])){
           m_samples[tracker_id[i]].push_back(feature_list[i]);
           if(m_samples[tracker_id[i]].size() > m_budget){
                m_samples[tracker_id[i]].erase(m_samples[tracker_id[i]].begin());
           }
       }else{
           m_samples[tracker_id[i]].clear();
           m_samples[tracker_id[i]].push_back(feature_list[i]);
       }
   }
//    debug<<"fit end"<<debugend;

}

int DistanceMetric::distance(std::vector<std::vector<double>>& cost_matrix, std::vector<std::vector<float>> features, std::vector<int> tracker_ids){

    int row = tracker_ids.size();
    int col = features.size();

    cost_matrix.clear();
    cost_matrix.resize(row, std::vector<double>(col, 10000));

    for(int i=0; i < row; i++){
        for(int j=0; j < col; j++){
            double min_distance = 100000;
            for(int k=0; k < m_samples[tracker_ids[i]].size(); k++){
                //TODO: use enum to find the correct function
                double distance = compute_euclidean_distance(m_samples[tracker_ids[i]][k], features[j]);
                //std::cout<<"nn matching, line 36 : "<<distance<<std::endl;
                min_distance =  distance  < min_distance? distance : min_distance; 
            }
            cost_matrix[i][j] = min_distance;
        }
    }

}

int DistanceMetric::distance(std::vector<std::vector<double>>& cost_matrix, std::vector<cv::Rect_<float>> detect_box, std::vector<BaseTracker> tracker){

    cost_matrix.clear();
    cost_matrix.resize(tracker.size(), std::vector<double>(detect_box.size(), 0.00));
    for(int i=0; i < tracker.size(); i++){
        for(int j=0; j < detect_box.size(); j++){
            cost_matrix[i][j] = 1 - get_iou(tracker[i].to_rect(), detect_box[j]);
        }
    }

}

double DistanceMetric::compute_euclidean_distance(std::vector<float> feature_a, std::vector<float> feature_b){

    assert(feature_a.size() == feature_b.size());
    double distance;
    for(int i=0; i<feature_a.size(); i++){
        distance += pow((feature_a[i]-feature_b[i]), 2);
    }
    //distance = sqrt(distance);
    return (double)distance;

}

double DistanceMetric::compute_cosine_distance(std::vector<float> feature_a, std::vector<float> feature_b){
    
    assert(feature_a.size() == feature_b.size());
    
    double denominator = get_pow_sum(feature_a) * get_pow_sum(feature_b);
    if(denominator == 0) return 1.00;
    double distance = get_vector_time(feature_a, feature_b);
    distance = distance / denominator;

    return distance;
}