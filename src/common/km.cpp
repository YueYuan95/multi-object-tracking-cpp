#include "km.h"

int KM::solve(std::vector<std::vector<float>> cost_matrix, std::vector<int>& assign){

    if(cost_matrix.size() == 0) return 0;

    m_match_x.clear();
    m_match_y.clear();
    m_weight_x.clear();
    m_weight_y.clear();
    m_vis_x.clear();
    m_vis_y.clear();

    int num_x = cost_matrix.size();
    int num_y = cost_matrix[0].size();

    m_match_x.resize(num_x, -1);
    m_match_y.resize(num_y, -1);
    m_weight_x.resize(num_x, 0);
    m_weight_y.resize(num_y, 0);
    m_vis_x.resize(num_x, 0);
    m_vis_y.resize(num_y, 0);

    assign.resize(num_x);

    for(int i=0; i < num_x; i++){
        for(int j=0; j < num_y; j++){
            if(cost_matrix[i][j] == 0.0){
                continue;
            }
            m_weight_x[i] = std::max(m_weight_x[i], cost_matrix[i][j]);
        }
    }

    for(int i=0; i < num_x; i++){

        while(true){
           m_vis_x.clear();
           m_vis_y.clear();
           m_vis_x.resize(num_x, 0);
           m_vis_y.resize(num_y, 0);

           if(dfs(i, cost_matrix)) break;

           for(int j=0; j < num_x; j++)
                if(m_vis_x[j]) m_weight_x[j] -= min_dis;
           for(int j=0; j < num_y; j++)
                if(m_vis_y[j]) m_weight_y[j] += min_dis;

        }
    }

    for(int i=0; i < num_x; i++){
        assign[i] = m_match_x[i];
    }

    return 1;
}

bool KM::dfs(int u, std::vector<std::vector<float>> cost_matrix){

    m_vis_x[u] = 1;

    for(int v=0; v<m_vis_y.size();v++){
        if(!m_vis_y[v] && cost_matrix[u][v] != 0.0){
            float temp = m_weight_x[u] + m_weight_y[v] - cost_matrix[u][v];
            
            if(temp == 0){
                m_vis_y[v] = 1;
                if(m_match_y[v] == -1 || dfs(m_match_y[v], cost_matrix)){
                    m_match_x[u] = v;
                    m_match_y[v] = u;
                    return true;
                }
            }else{
                if(temp>0){
                    min_dis = std::min(min_dis, temp);
                }
            }
        }
    }    
    return false;
}