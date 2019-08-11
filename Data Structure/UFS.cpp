/*******************************************************************************
  按秩合并并查集，支持修改操作
  join函数返回的是本次join产生的影响数目，意思是需要撤回几次才能恢复本次join造成的影响
********************************************************************************/
namespace UFS{
    using namespace std;
    int rank[N], fa[N];
    int n;
    stack<pair<int*, int> > cur;
    void init()
    {
        memset(rank, 0, sizeof(rank));
        REP(i, 1, n) fa[i] = i;
    }
    inline int find(int x) { return x == fa[x] ? x : find(fa[x]); }
    inline int join(int x,int y){
        x = find(x), y = find(y);
        if (x==y) return 0;
        int tot = 0;
        if (rank[x]<rank[y]){
            cur.push(mp(fa + x, fa[x]));
            fa[x] = y;
            tot = 1;
        }else {
            cur.push(mp(fa + y, fa[y]));
            fa[y] = x;
            tot = 1;
            if (rank[x]==rank[y]){
                cur.push(mp(rank + y, rank[y]));
                ++rank[y];
                tot++;
            }
        }
        return tot;
    }
    inline void undo(int x){
        FOR(i,x){
            *cur.top().fi = cur.top().sc;
            cur.pop();
        }
    }
}
using namespace UFS;
