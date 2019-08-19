void init(){
    LOG[2] = 1;
    for (register int i = 3; i <= int(1e5); ++i)
    {
        LOG[i] = LOG[i / 2] + 1;
    }
}
void dfs(int x, int fa){
    id[dfs_in[x] = ++dfn] = x;
    seq[++dfsn] = x;
    first[x] = dfsn;
    dep[x] = dep[fa] + 1;
    for (int i = he[x]; i; i = ne[i])
    {
        int p = v[i];
        if (p==fa) continue;
        dfs(p, x);
        seq[++dfsn] = x;
    }
    dfs_out[x] = dfn;
}
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
inline int lca(int x,int y){
    int ix = first[x], iy = first[y];                       //index of its first shown in dfs sequence
    if (ix>iy)
        swap(ix, iy);
    int k = LOG[iy - ix + 1];
    int a = dp[ix][k];
    int b = dp[iy - (1 << k) + 1][k];
    return dep[a] < dep[b] ? a : b;
}
