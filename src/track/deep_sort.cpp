#include "deep_sort.h"

bool DeepSort::inference(const cv::Mat image, const DetectObjects& detectResults, std::vector<TrackeKeyObject>& keyObjects){

	for(int i=0; i < detectResults.size(); i++){
		float x = detectResults[i].box.topLeftX;
		float y = detectResults[i].box.topLeftY;
		float w = detectResults[i].box.width;
		float h = detectResults[i].box.height;
		cv::Rect_<float> rect = cv::Rect_<float>(x, y, w, h);
        DetObj det_obj = {detectResults[i].label, detectResults[i].score, rect};
		m_detect.push_back(det_obj);
	} 

    if(m_trackers.size() == 0){

        for (int i = 0; i < m_detect.size(); i++)
	    {
		    KalmanTracker trk = KalmanTracker(m_detect[i].bbox, m_detect[i].label);
            //TODO: update descriptor
		    m_trackers.push_back(trk);
	    }

	}else{
        predict(image);
		computeDistance();
		assignMatrix();
		matchResult();
		update(image);
	}   
	
    sendResult(image, keyObjects);

    return true;
};


void DeepSort::predict(cv::Mat image){

    for(int i = 0; i < m_trackers.size(); i++)
	{
		m_trackers[i].predict(image);
	}
};

void DeepSort::computeDistance(){
    
    int trk_num = m_trackers.size();
    int det_num = m_detect.size();

    m_cost_matrix.clear();
    m_cost_matrix.resize(trk_num, std::vector<double>(det_num, 0));

    //Compute cost with label;

    for(int i=0; i < trk_num; i ++){
        for(int j=0; j < det_num; j++){
            m_cost_matrix[i][j] = 1 - getIou(m_trackers[i].getBbox(), m_detect[j].bbox);
        }
    }
};

double DeepSort::getIou(cv::Rect_<float> tracker, cv::Rect_<float> detection){
    
    float in = (tracker & detection).area();
	float un = tracker.area() + detection.area() - in;

    if(un < DBL_EPSILON){
        return 0.0;
    }

    return (double)(in/un);
};

void DeepSort::assignMatrix(){

    //TODO: assign with label

    Hungarian HungAlgo;

    m_assign.clear();
	HungAlgo.Solve(m_cost_matrix, m_assign);

};


void DeepSort::matchResult(){

    int trkNum = m_trackers.size();
    int detNum = m_detect.size();

	m_unmatched_trk.clear();
	m_unmatched_det.clear();
	m_all.clear();
	m_matched.clear();

	if (detNum > trkNum) //	there are unmatched detections
	{
		for (int n = 0; n < detNum; n++)
			m_all.insert(n);

		for (int i = 0; i < trkNum; ++i)
			m_matched.insert(m_assign[i]);

		std::set_difference(m_all.begin(), m_all.end(),
			m_matched.begin(), m_matched.end(),
			std::insert_iterator<std::set<int>>(m_unmatched_det, m_unmatched_det.begin()));
	}
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (int i = 0; i < trkNum; ++i)
			if (m_assign[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				m_unmatched_trk.insert(i);
	}
    
	// filter out matched with low IOU
	m_match_pairs.clear();
	for (int i = 0; i < trkNum; ++i)
	{
		if (m_assign[i] == -1) // pass over invalid values
			continue;
        if (1 - m_cost_matrix[i][m_assign[i]] < 0.25)
		{
			m_unmatched_trk.insert(i);
        	m_unmatched_det.insert(m_assign[i]);
		}
	    else
		    m_match_pairs.push_back(cv::Point(i, m_assign[i]));
	}

};

void DeepSort::update(cv::Mat image){

    int detIdx, trkIdx;
    for (int i = 0; i < m_match_pairs.size(); i++)
    {
        trkIdx = m_match_pairs[i].x;
        detIdx = m_match_pairs[i].y;
        m_trackers[trkIdx].update(image, m_detect[detIdx].bbox);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : m_unmatched_det)
    {
        KalmanTracker tracker = KalmanTracker(m_detect[umd].bbox, m_detect[umd].label);
        m_trackers.push_back(tracker);
    }

    //Update Feature Descriptor

};

void  DeepSort::sendResult(cv::Mat image, std::vector<TrackeKeyObject>& keyObjects){

    keyObjects.clear();
    for (auto it = m_trackers.begin(); it != m_trackers.end();)
    {
        if (((*it).m_time_since_update <= m_max_miss_time))
        {
            cv::Rect_<float> rect_box = (*it).getBbox();
            int id = (*it).getId();
            int label  = (*it).getLabel();
            BboxInfo box = {int(rect_box.x), int(rect_box.y), int(rect_box.width), int(rect_box.height)};
            TrackeKeyObject obj = {id, label, box, image, false}; 
            //cout << id << "," << rect_box.x << "," << rect_box.y << "," << rect_box.width << "," << rect_box.height << endl;
            keyObjects.push_back(obj);
            it++;
        }
        else
        {
            it++;
        }
        // remove dead tracklet
        if (it != m_trackers.end() && (*it).m_time_since_update > m_max_miss_time){
            
            cv::Rect_<float> rect_box = (*it).getBbox();
            int id = (*it).getId();
            int label  = (*it).getLabel();
            BboxInfo box = {int(rect_box.x), int(rect_box.y), int(rect_box.width), int(rect_box.height)};
            TrackeKeyObject obj = {id, label, box, image, true}; 
            //cout << id << "," << rect_box.x << "," << rect_box.y << "," << rect_box.width << "," << rect_box.height << endl;
            keyObjects.push_back(obj);
            
            it = m_trackers.erase(it);
        }
    }
}



bool DeepSort::init(const std::string& model_dir, const TrackeParas& pas, const int gpu_id){
    return true;
}