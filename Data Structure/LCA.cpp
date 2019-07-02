/*
  根节点的深度为1
  根节点的父亲为0
*/
void LCA(){
    for (int j = 1; (1 << j) <= n;j++)
    {
        for (int i = 1; i <= n;i++)
        {
            f[i][j] = f[f[i][j - 1]][j - 1];
            g[i][j] = g[f[i][j - 1]][j - 1] + g[i][j - 1];
        }
    }
}
void dfs(int x,int fa,int dep){
    dp[x] = dep;
    f[x][0] = fa;
    for (int i = he[x]; i;i=ne[i])
    {
        int p = v[i];
        if (p==fa)
            continue;
        g[p][0] = w[i];
        dfs(p, x, dep + 1);
    }
}
ll query(int x,int y){
    ll sum = 0;
    if (dp[x]<dp[y])
        swap(x, y);
    for (int j = 18; j >= 0;j--)
    {
        int fx = f[x][j];
        if (dp[fx]>=dp[y]){
            sum += g[x][j];
            x = fx;
        }
    }
    for (int j = 18; j >= 0;j--)
    {
        int fx = f[x][j];
        int fy = f[y][j];
        if (fx==fy)
            continue;
        sum += g[x][j];
        sum += g[y][j];
        x = fx;
        y = fy;
    }
    if (x!=y){
        sum += g[x][0];
        sum += g[y][0];
    }
    return sum;
}
