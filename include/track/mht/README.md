## MHT 
Multiple Hypothesis Tracking Revised


----------
## Method and parameters
#Gating in mht_tracker.cpp
Gating: IOU/(1_distance)>0.4/(1+40)
Scoring: IOU
New tree's root score = ICH score = 0.01 0r 0.0001 (no difference)
maxScaleDiff = 1.4

#in mht_tracker.h
N = 10
miss_time threshold = N+10 

#mwis in mht_tracker.cpp
MWIS: mwis_greedy

# NMS in mht.cpp
score_diff = 2

# computerdistance in mht.cpp
NMS overlap threshold  = 0.35 

#create_ICH in tree.cpp
ICH socre: 0.01 or 0.0001 (no difference)


----------
## TO Do
1.Establish a file to save the tracking results, eg:tracking_result_0925
2.Establish files to save tracking results for a specific sequence in 1, eg:MOT16-13. The tracking result images and .txt file are saved in it.
3.The name of the tracking result txt is the name of the sequence

## Run
Run the main.cpp to test a sequence
