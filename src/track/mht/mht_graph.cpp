#include "mht_graph.h"

Graph::Graph(){

    m_node_list.clear();

    m_max_clique.clear();

    m_stk_list.clear();

    m_adj_mat.clear();
    m_dej_mat.clear();
    
}

Graph::Graph(std::vector<VexNode> vex_node_list){

    m_node_list = vex_node_list;
    
    m_node_num = m_node_list.size();

    m_max_clique.clear();
    m_max_clique.resize(m_node_num);

    m_stk_list.clear();
    m_stk_list.resize(m_node_num, std::vector<int>(m_node_num));


    m_adj_mat.clear();
    m_dej_mat.clear();
    
    m_adj_mat.resize(m_node_num, std::vector<int>(m_node_num, 0));
    m_dej_mat.resize(m_node_num, std::vector<int>(m_node_num, 1));

    m_vetex_list.clear();
    m_max_clique_list.clear();

    for(int i=0; i < m_node_num; i++){
        for(int j=0; j < m_node_num;j++){
            //assert(vex_node_list[i].path.size() == vex_node_list[j].path.size());
            if(i == j){
                m_dej_mat[i][j]=0;
                m_dej_mat[j][i]=0;
                continue;
            }
            for(int k=0; k < vex_node_list[i].path.size(); k++){
                if(vex_node_list[i].path[k] ==  vex_node_list[j].path[k]){
                    m_adj_mat[i][j] = 1;
                    m_adj_mat[j][i] = 1;
                    m_dej_mat[i][j] = 0;
                    m_dej_mat[j][i] = 0;
                    break;
                }

            } 
            
        }
    }

    m_score = 0.0;
}

int Graph::DFS(int n){

    for(int i=n+1; i<=m_node_num; i++){
    //TODO: if the sum of all left node less than current score, save time;
       if(m_dej_mat[n][i]){
            int j;
            for(j=0; j< m_vetex_list.size(); j++){
                if(!m_dej_mat[i][m_vetex_list[j]]){
                    break;                
                }
            }
            if(j == m_vetex_list.size()){
                
               m_vetex_list.push_back(i);
               DFS(i);
               //TODO: if score big than max_score, push back and reset score;
               m_max_clique_list.push_back(m_vetex_list);
               m_vetex_list.pop_back();
            }
                
       }
        
    }

}

int Graph::mwis(std::map<int, std::vector<int>>& routes){

    m_max_clique.clear();
    routes.clear();
    int n = m_node_list.size();

    std::vector<std::vector<int>> save_clique; 
    for(int i=n-1; i>=0; i--){
        m_vetex_list.clear();
        m_vetex_list.push_back(i);
        DFS(i);
        save_clique.push_back(m_vetex_list);
    }

       
    for(int i=0;  i <  m_max_clique_list.size(); i++){
        float sum = 0.0;
        for(auto node : m_max_clique_list[i]){
            sum += m_node_list[node].score;
        }
        if(sum >= m_score){
            m_max_clique = m_max_clique_list[i];
            m_score = sum;
        }
    }

    for(auto path : m_max_clique){
        std::cout<<path<<" ";
    }
    std::cout<<std::endl;

    for(auto node_index : m_max_clique){
        routes[m_node_list[node_index].id] = m_node_list[node_index].path;
    }

}


int Graph::printGraph(){
   
    std::cout<<"Adj Mat is"<<std::endl;
    for(auto row : m_adj_mat){
           for(auto col : row){
             std::cout<<col<<" ";
           }
           std::cout<<std::endl;
    }

    std::cout<<"Dej Mat is"<<std::endl;
     for(auto row : m_dej_mat){
           for(auto col : row){
             std::cout<<col<<" ";
           }
           std::cout<<std::endl;
    }


}
