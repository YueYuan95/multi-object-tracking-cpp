#ifndef __KM_H__
#define __KM_H__

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <vector>

class KM{

    private:
        std::vector<int> m_match_x;
        std::vector<int> m_match_y;

        std::vector<float> m_weight_x;
        std::vector<float> m_weight_y;

        std::vector<int> m_vis_x;
        std::vector<int> m_vis_y;

        float min_dis = 100.0;

    public:
        int solve(std::vector<std::vector<float>>, std::vector<int>&);
        bool dfs(int, std::vector<std::vector<float>>);


};

#endif