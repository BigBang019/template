/*
  节点编号从1开始，
  根节点父亲为0
  根节点深度为1
*/
namespace LCA{
    using namespace std;
    typedef long long ll;
    const int N = 4e4 + 5;
    int f[N][30];
    ll g[N][30];

    int cnt, he[N], ne[N * 2], v[N * 2], dp[N];
    ll w[N * 2];
    int n;
    void add(int x,int y,ll z){
        cnt++;
        v[cnt] = y;
        ne[cnt] = he[x];
        he[x] = cnt;
        w[cnt] = z;
    }
    void lca(){
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
    void init(){
        cnt = 0;
        memset(he, 0, sizeof(he));
    }
}
using namespace LCA;
