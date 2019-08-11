/******************************************************
  差分BIT维护差分数组，query查询前缀和
*******************************************************/
namespace BIT{
    int n;
    ll c1[N], c2[N];
    void init(int _n) { n = _n; memset(c1, 0, sizeof(c1)); memset(c2, 0, sizeof(c2)); }
    int lb(int x) { return x & -x; }
    void update(int x, int v)
    {
        for (int i = x; i <= n; i += lb(i))
        {
            c1[i] += v;
            c2[i] += v * x;
        }
    }
    ll query(int x){
        ll sum = 0;
        for (int i = x; i > 0; i -= lb(i)){
            sum += 1ll * (x + 1) * c1[i] - c2[i];
        }
        return sum;
    }
}
