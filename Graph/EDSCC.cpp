namespace EDSCC{
    using namespace std;
    const int N = 1e5 + 10;
    int n;
    int cnt, he[N], ne[N * 2], v[N * 2];
    void add(int x,int y){
        cnt++;
        ne[cnt] = he[x];
        v[cnt] = y;
        he[x] = cnt;
    }
    int sccn, ind, dfn[N], low[N], blk[N];
    bool bri[N * 2];
    void init(){
        sccn = ind = 0;
        cnt = -1;
        memset(he, -1, sizeof(he));
        memset(dfn, 0, sizeof(dfn));
        memset(blk, 0, sizeof(blk));
    }
    void tanjan(int x,int ie){
        dfn[x] = low[x] = ++ind;
        for (int i = he[x]; ~i;i=ne[i])
        {
            int p = v[i];
            if (!dfn[p]){
                tanjan(p, i);
                low[x] = min(low[x], low[p]);
                if (low[p]>low[x])
                    bri[i] = bri[i ^ 1] = 1;
            }else if (i!=(ie^1)){
                low[x] = min(low[x], dfn[p]);
            }
        }
    }
    
    void dfs(int x,int cl){
        blk[x] = cl;
        for (int i = he[x]; ~i;i=ne[i])
        {
            int p = v[i];
            if (bri[i])
                continue;
            if (!blk[p])
                dfs(p, cl);
        }
    }
    void get_edscc(){
        for (int i = 1; i <= n;i++)
        {
            if (!blk[i])
                dfs(i, ++sccn);
        }
    }
}
using namespace EDSCC;
