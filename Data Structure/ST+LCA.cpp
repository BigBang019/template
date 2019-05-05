void ST(){
    for (int i = 1; i <= dfsn;i++)
    {
        dp[i][0] = seq[i];                                  //dfs sequence
    }
    for (int j = 1; (1 << j) <= dfsn;j++)
    {
        for (int i = 1; i + (1 << j) - 1 <= dfsn; i++)      //!
        {
            int a = dp[i][j - 1];
            int b = dp[i + (1 << (j - 1))][j - 1];
            dp[i][j] = dep[a] < dep[b] ? a : b;
        }
    }
}
int lca(int x,int y){
    int ix = first[x], iy = first[y];                       //index of its first shown in dfs sequence
    if (ix>iy)
        swap(ix, iy);
    int k = 0;
    while ((1 << (1 + k)) <= iy - ix + 1)
        k++;
    int a = dp[ix][k];
    int b = dp[iy - (1 << k) + 1][k];
    return dep[a] < dep[b] ? a : b;
}
