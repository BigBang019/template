/*
	MOD, root is needed
    lazy,son,dep,he,cnt initialization needed
*/
namespace HLD{
    using namespace std;
    typedef long long ll;
    const int N = 1e5 + 5;

    int cnt, he[N], ne[N * 2], v[N * 2];

    int son[N], sz[N], id[N], top[N], dep[N], fa[N], idn;

    ll wt[N], a[N], tree[N << 2], lazy[N << 2];

    ll MOD;
    int n, root;
    void add(int x,int y){
        cnt++;
        ne[cnt] = he[x];
        v[cnt] = y;
        he[x] = cnt;
    }
    void dfs1(int x,int fat,int d){
        dep[x] = d;
        fa[x] = fat;
        sz[x] = 1;
        int heavy = 0;
        for (int i = he[x]; i; i = ne[i])
        {
            int p = v[i];
            if (p==fat)
                continue;
            dfs1(p, x, d+1);
            if (heavy<sz[p]){
                heavy = sz[p];
                son[x] = p;
            }
            sz[x] += sz[p];
        }
    }
    void dfs2(int x,int fa,int topf){
        id[x] = ++idn;
        a[idn] = wt[x];
        top[x] = topf;
        if (!son[x])
            return;
        dfs2(son[x], x, topf);
        for (int i = he[x]; i; i = ne[i])
        {
            int p=v[i];
            if (p==fa || p==son[x])
                continue;
            dfs2(p, x, p);
        }
    }
    void init(){
        cnt = 0;
        memset(he, 0, sizeof(he));
        memset(dep, 0, sizeof(dep));
    }
    void pushdown(int x,int l,int r){
        if (l==r){
            lazy[x] = 0;
            return;
        }
        int mid = l + r >> 1;
        tree[2 * x] += lazy[x] * (mid - l + 1) % MOD; //MOD
        tree[2 * x + 1] += lazy[x] * (r - mid) % MOD; //MOD
        lazy[2 * x] += lazy[x];
        lazy[2 * x + 1] += lazy[x];

        tree[2 * x] %= MOD;						//MOD
        tree[2 * x + 1] %= MOD;
        lazy[2 * x] %= MOD;
        lazy[2 * x + 1] %= MOD;

        lazy[x] = 0;
    }
    void build(int x,int l,int r){
        if (l==r){
            tree[x] = a[l];
            return;
        }
        int mid = l + r >> 1;
        build(2 * x, l, mid);
        build(2 * x + 1, mid + 1, r);
        tree[x] = tree[2 * x] + tree[2 * x + 1];
        tree[x] %= MOD;							//MOD
    }
    void insert(int x,int l,int r,int ql,int qr,ll v){
        if (qr<l || ql>r)
            return;
        if (ql<=l && qr>=r){
            tree[x] += (r - l + 1) * v;
            lazy[x] += v;

            tree[x] %= MOD;						//MOD
            lazy[x] %= MOD;

            return;
        }
        int mid = l + r >> 1;
        pushdown(x, l, r);
        insert(2 * x, l, mid, ql, qr, v);
        insert(2 * x + 1, mid + 1, r, ql, qr, v);
        tree[x] = tree[2 * x] + tree[2 * x + 1];
        tree[x] %= MOD;							//MOD
    }
    ll query(int x,int l,int r,int ql,int qr){
        if (qr<l || ql>r) return 0;
        int mid = l + r >> 1;
        if (ql<=l && qr>=r)
            return tree[x];
        pushdown(x, l, r);
        return (query(2 * x, l, mid, ql, qr) + query(2 * x + 1, mid + 1, r, ql, qr)) % MOD;//MOD
    }
    void dec(){
        idn = 0;
        dfs1(root, 0, 1);
        dfs2(root, 0, root);
        build(1, 1, n);
    }
    ll qSon(int x){
        int l = id[x], r = id[x] + sz[x] - 1;
        return query(1, 1, n, l, r) % MOD;				//MOD
    }
    void aSon(int x,ll v){
        v %= MOD;										//MOD
        int l = id[x], r = id[x] + sz[x] - 1;
        insert(1, 1, n, l, r, v);
    }
    ll qRange(int x,int y){
        ll sum = 0;
        while (top[x]!=top[y]){
            if (dep[top[x]] < dep[top[y]])				//比较top dep
                swap(x, y);
            int topf = top[x];
            sum += query(1, 1, n, id[topf], id[x]);
            sum %= MOD;									//MOD
            x = fa[topf];
        }
        if (dep[x]>dep[y])
            swap(x, y);
        sum += query(1, 1, n, id[x], id[y]);
        return sum % MOD;								//MOD
    }
    void aRange(int x,int y, ll v){
        v %= MOD;										//MOD
        while (top[x]!=top[y]){
            if (dep[top[x]] < dep[top[y]])
                swap(x, y);
            int topf = top[x];
            insert(1, 1, n, id[topf], id[x], v);
            x = fa[topf];
        }
        if (dep[x]>dep[y])
            swap(x, y);
        insert(1, 1, n, id[x], id[y], v);
    }
}
using namespace HLD;
