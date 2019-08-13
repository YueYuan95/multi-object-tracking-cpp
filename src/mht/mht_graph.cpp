#include "mht_graph.h"

int Graph::DFS(int n, int neighbor_idx, int deep){

    if(neighbor_idx == 0){
        float clique_score = 0.00;
        for(auto iter : vetexlist){
            clique_score += node_list[(*iter)].score;
        }
        if(clique_score > score){
            mx = deep;
            std::vector<int>::iterator iter = vetexlist.begin();
            max_clique_list.clear();
            for(;iter!=vetexlist.end();iter++){
                maxcliquelist.push_back(*iter);
            }
        }
        return 1;
    }
    for(int i=0; i < neighbor_idx; i++){
        
        int k = stk[deep][i];
        int cnt = 0;
        //TODO: cut
        for(int j=i+1; j < neighbor_idx; j++){
            int p = stk[deep][j];
            if(dej_mat[k][p]){
                stk[deep+1][cnt++] = p;
            }
        }
        vetexlist.push_back(k);
        DFS(n, cnt, deep+1);
        vetexlist.pop_back();
    }
    return 1;

}

int Graph::mwis(){

    int n = node_list.size();
    for(int i= n-1; i >= 0; i--){
        for(int j=i+1, ns=0; j < n; j++){
            if(dej_mat[i][j]){
                stk[1][ns++] = j;
            }
        }
        vetexlist.push_back(i);
        DFS(n, ns, 1);
        vetexlist.pop_back();
        dp[i] = mx;
    }
}

int Graph::print(){

}