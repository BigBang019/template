# Template

## 数据结构

### BIT

#### 点修改区间查询

```c++
/*
	initalize: c
	维护a[i][j]数组
	查询结果为(x2,y2)-(x1-1,y2)-(x2,y1-1)+(x1-1,y1-1)
*/
int lowbit(int x){
	return x & (-x);
}
void add(int x,int y,ll v){
	for (int i = x; i <= n;i+=lowbit(i))
	{
		for (int j = y; j <= n;j+=lowbit(j))
		{
			c[i][j] += v;
		}
	}
}
ll query(int x,int y){
	ll sum = 0;
	for (int i = x; i > 0;i-=lowbit(i))
	{
		for (int j = y; j > 0;j-=lowbit(j))
		{
			sum += c[i][j];
		}
	}
	return sum;
}
ll getSum(int x1,int x2,int y1,int y2){
	return query(x2, y2) - query(x2, y1 - 1) - query(x1 - 1, y2) + query(x1 - 1, y1 - 1);
}
```

#### 区间修改单点查询

```c++
/*
	initialize: c,d
	维护差分数组d[i][j]
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1050;
int n, c[N][N];
int lowbit(int x){
	return x & (-x);
}
void add(int x,int y,int v){
	for (int i = x; i <= n;i+=lowbit(i))
	{
		for (int j = y; j <= n;j+=lowbit(j))
		{
			c[i][j] += v;
		}
	}
}
int query(int x,int y){
	int sum = 0;
	for (int i = x; i > 0;i-=lowbit(i))
	{
		for (int j = y; j > 0;j-=lowbit(j)){
			sum += c[i][j];
		}
	}
	return sum;
}
void add(int x1,int x2,int y1,int y2){
	add(x1, y1, 1);
	add(x1, y2 + 1, -1);
	add(x2 + 1, y1, -1);
	add(x2 + 1, y2 + 1, 1);
}
```

#### 区间修改区间查询

```c++
/*
	initialize: c1,c2,c3,c4,d
*/
int lowbit(int x){
    return x & (-x);
}
void add(ll x, ll y, ll z){
    for(int i = x; i <= n; i += lowbit(i))
        for(int j = y; j <= m; j += lowbit(j)){
            c1[i][j] += z;
            c2[i][j] += z * x;
            c3[i][j] += z * y;
            c4[i][j] += z * x * y;
        }
}
void range_add(ll x1, ll x2, ll y1, ll y2, ll z){ //(xa, ya) 到 (xb, yb) 的矩形
    add(x1, y1, z);
    add(x1, y2 + 1, -z);
    add(x2 + 1, y1, -z);
    add(x2 + 1, y2 + 1, z);
}
ll ask(ll x, ll y){
    ll res = 0;
    for(int i = x; i; i -= lowbit(i))
        for(int j = y; j; j -= lowbit(j))
            res += (x + 1) * (y + 1) * c1[i][j]
                - (y + 1) * c2[i][j]
                - (x + 1) * c3[i][j]
                + c4[i][j];
    return res;
}
ll range_ask(ll x1, ll x2, ll y1, ll y2){
    return ask(x2, y2) - ask(x2, y1 - 1) - ask(x1 - 1, y2) + ask(x1 - 1, y1 - 1);
}
```

#### 差分

```c++
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
```

### HLD

```c++
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
	memset(son,0,sizeof(son));
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
```

### LCA

```c++
/*
  节点编号从1开始，
  根节点父亲为0
  根节点深度为1
*/
namespace LCA{
    using namespace std;
    typedef long long ll;
    const int N = 2e3 + 5;
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
        memset(f, 0, sizeof(f));
        memset(g, 0, sizeof(g));
    }
}
using namespace LCA;
```

### Linear Base

```c++
// #pragma GCC optimize(2)
#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define sc second
#define fi first
using namespace std;
typedef long long ll;
typedef unsigned int uint;
typedef pair<int, int> pi;
const int N = 5e4 + 5;
const int B = 31;
struct Base{
    uint b[B + 3];
    Base() { init(); }
    uint &operator [] (uint i) {
        return b[i];
    }
    void init() { memset(b, 0, sizeof(b)); }
    void update(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                b[i] = x;
                break;
            }
            x ^= b[i];
        }
    }
    bool check(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                break;
            }
            x ^= b[i];
        }
        return (x == 0);
    }
    static Base Merge(Base a, Base b){
        Base c, tmp_b, tmp_k;
        for (int i = 0; i <= B; i++)
        {
            tmp_b[i] = tmp_k[i] = 0;
            if (a[i]) tmp_b[i] = a[i];
        }
        for (int i = 0; i <= B; i++) c[i] = 0;
        for (int i = 0; i <= B; i++) if (b[i])
        {
            int ok = 1;
            uint v = b[i], k = 0;
            for (int j = B; ~j; j--)
            {
                if (v & (uint(1) << j)){
                    if (tmp_b[j]){
                        v ^= tmp_b[j];
                        k ^= tmp_k[j];
                    }else{
                        tmp_b[j] = v;
                        tmp_k[j] = (uint(1) << i) ^ k;
                        ok = 0;
                        break;
                    }
                }
            }
            if (ok){
                uint v = b[i];
                for (int j = 0; j <= B; j++)
                {
                    if (k & (uint(1) << j))
                    {
                        v ^= b[j];
                    }
                }
                for (int j = B; ~j; j--)
                {
                    if (v & (uint(1) << j)){
                        if (c[j]){
                            v ^= c[j];
                        }else{
                            c[j] = v;
                            break;
                        }
                    }
                }
            }
        }
        return c;
    }
};
```

### 主席树

#### 静态

```c++
/*
    空间开2+log(N)倍
    静态区间查询第k小
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 5;
int lc[N * 20], rc[N * 20], tree[N], sum[N * 20];
int tot;
void build(int &rt,int l,int r){
    rt = ++tot;
    sum[rt] = 0;
    if (l==r)
        return;
    int mid = l + r >> 1;
    build(lc[rt], l, mid);
    build(rc[rt], mid + 1, r);
}
void insert(int last,int p,int l,int r,int &rt){
    rt = ++tot;
    lc[rt] = lc[last];
    rc[rt] = rc[last];
    sum[rt] = sum[last] + 1;
    if (l==r)
        return;
    int mid = l + r >> 1;
    if (p<=mid)
        insert(lc[last], p, l, mid, lc[rt]);
    else
        insert(rc[last], p, mid + 1, r, rc[rt]);
}
int query(int last,int now,int l,int r,int k){
    if (l==r)
        return l;
    int mid = l + r >> 1;
    int cnt = sum[lc[now]] - sum[lc[last]];
    if (k<=cnt)
        return query(lc[last], lc[now], l, mid, k);
    else
        return query(rc[last], rc[now], mid + 1, r, k - cnt);
}
void init(){
    tot = 0;
}
```

#### 动态

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 6e4 + 5;
const int M = 1e4 + 5;
typedef long long ll;
struct Q{
	int l, r, k;
	Q(){}
	Q(int l, int r, int k) : l(l), r(r), k(k){}
} qu[M];

int a[N], b[N], cnt;
int T[N], S[N], tot;
int sum[N * 40], lc[N * 40], rc[N * 40];
int use[N];
int n, m;

void init(){
	tot = 0;
	cnt = 0;
}
void Init_Hash(){
	sort(b + 1, b + 1 + cnt);
	cnt = unique(b + 1, b + 1 + cnt) - b - 1;
}
int has(int x){
	return (lower_bound(b + 1, b + 1 + cnt, x) - b);
}
int lowbit(int x){
	return x & (-x);
}
void build(int &rt,int l,int r){
	rt = ++tot;
	sum[rt] = 0;
	if (l==r)
		return;
	int mid = l + r >> 1;
	build(lc[rt], l, mid);
	build(rc[rt], mid + 1, r);
}
int update(int last,int l,int r,int p,int v){
	int rt = ++tot;
	int tmp = rt;
	sum[rt] = sum[last] + v;
	while (l<r){
		int mid = l + r >> 1;
		if (p<=mid){
			lc[rt] = ++tot;
			rc[rt] = rc[last];
			rt = lc[rt];
			last = lc[last];
			r = mid;
		}else{
			rc[rt] = ++tot;
			lc[rt] = lc[last];
			rt = rc[rt];
			last = rc[last];
			l = mid + 1;
		}
		sum[rt] = sum[last] + v;
	}
	return tmp;
}
void add(int index,int x,int v){
	for (int i = index; i <= n;i+=lowbit(i))
	{
		S[i] = update(S[i], 1, cnt, x, v);
	}
}
int sumLeft(int x){
	int ans = 0;
	for (int i = x; i;i-=lowbit(i))
	{
		ans += sum[lc[use[i]]];
	}
	return ans;
}
int query(int le,int re,int k){
	int left_root = T[le - 1];
	int right_root = T[re];
	int l = 1, r = cnt;
	for (int i = le - 1; i;i-=lowbit(i))
		use[i] = S[i];
	for (int i = re; i;i-=lowbit(i))
		use[i] = S[i];
	while (l<r){
		int mid = l + r >> 1;
		int tmp = sumLeft(re) - sumLeft(le - 1) + sum[lc[right_root]] - sum[lc[left_root]];
		if (tmp>=k){
			r = mid;
			left_root = lc[left_root];
			right_root = lc[right_root];
			for (int i = le - 1; i;i-=lowbit(i))
				use[i] = lc[use[i]];
			for (int i = re; i;i-=lowbit(i))
				use[i] = lc[use[i]];
		}else{
			l = mid + 1;
			left_root = rc[left_root];
			right_root = rc[right_root];
			for (int i = le - 1; i;i-=lowbit(i))
				use[i] = rc[use[i]];
			for (int i = re; i;i-=lowbit(i))
				use[i] = rc[use[i]];
			k -= tmp;
		}
	}
	return l;
}
void work(){
	init();
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d", &a[i]);
		b[++cnt] = a[i];
	}
	for (int i = 1; i <= m;i++)
	{
		char ch;
		int l, r, k;
		scanf(" %c%d%d", &ch, &l, &r);
		if (ch=='Q'){
			scanf("%d", &k);
			qu[i] = Q(l, r, k);
		}else{
			qu[i] = Q(l, r, -1);
			b[++cnt] = r;
		}
	}
	Init_Hash();
	build(T[0], 1, cnt);
	for (int i = 1; i <= n;i++)
	{
		T[i] = update(T[i - 1], 1, cnt, has(a[i]), 1);
		S[i] = T[0];
	}
	for (int i = 1; i <= m;i++)
	{
		if (~qu[i].k){
			printf("%d\n", b[query(qu[i].l, qu[i].r, qu[i].k)]);
		}else{
			add(qu[i].l, has(a[qu[i].l]), -1);
			add(qu[i].l, has(qu[i].r), 1);
			a[qu[i].l] = qu[i].r;
		}
	}
}
int main(){
	int t;
	scanf("%d", &t);
	while (t--){
		work();
	}
	return 0;
}
```

### ST表

#### 2D

```c++
namespace ST_2D{
    using namespace std;
    typedef long long ll;
    const int N = 5e2 + 5;
    int maze[N][N];
    int LOG[N];
    int dp[N][N][10][10];
    int mp[N][N][10][10];
    inline int Max(int a,int b){
        return a > b ? a : b;
    }
    inline int Min(int a,int b){
        return a < b ? a : b;
    }
    void init(){
        for (register int i = 2; i <= N - 5; i++)
        {
            LOG[i] = LOG[i >> 1] + 1;
        }
    }
    inline void ST(int n,int m){
        for (register int i = 1; i <= n;i++)
        {
            for (register int j = 1; j <= m;j++)
            {
                dp[i][j][0][0] = maze[i][j];
                mp[i][j][0][0] = maze[i][j];
            }
        }
        for (register int k = 0; (1 << k) <= n;k++)
        {
            for (register int l = 0; (1 << l) <= m;l++)
            {
                if (l==0 && k==0)
                    continue;
                for (register int i = 1; i + (1 << k) - 1 <= n;i++)
                {
                    for (register int j = 1; j + (1 << l) - 1 <= m;j++)
                    {
                        if (k==0){
                            dp[i][j][k][l] = Max(dp[i][j][k][l - 1], dp[i][j + (1 << (l - 1))][k][l - 1]);
                            mp[i][j][k][l] = Min(mp[i][j][k][l - 1], mp[i][j + (1 << (l - 1))][k][l - 1]);
                        }
                        else{
                            dp[i][j][k][l] = Max(dp[i][j][k - 1][l], dp[i + (1 << (k - 1))][j][k - 1][l]);
                            mp[i][j][k][l] = Min(mp[i][j][k - 1][l], mp[i + (1 << (k - 1))][j][k - 1][l]);
                        }
                    }
                }
            }
        }
    }
    inline int query(int x1,int x2,int y1,int y2){
        int k = LOG[x2 - x1 + 1], l = LOG[y2 - y1 + 1];
        int ans = dp[x1][y1][k][l];
        ans = Max(ans, dp[x1][y2 - (1 << l) + 1][k][l]);
        ans = Max(ans, dp[x2 - (1 << k) + 1][y1][k][l]);
        ans = Max(ans, dp[x2 - (1 << k) + 1][y2 - (1 << l) + 1][k][l]);
        return ans;
    }
    inline int queryM(int x1,int x2,int y1,int y2){
        int k = LOG[x2 - x1 + 1], l = LOG[y2 - y1 + 1];
        int ans = mp[x1][y1][k][l];
        ans = Min(ans, mp[x1][y2 - (1 << l) + 1][k][l]);
        ans = Min(ans, mp[x2 - (1 << k) + 1][y1][k][l]);
        ans = Min(ans, mp[x2 - (1 << k) + 1][y2 - (1 << l) + 1][k][l]);
        return ans;
    }
}
using namespace ST_2D;
```

#### +LCA

```c++
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
```

### TREAP

```c++
namespace TRP{
    using namespace std;
    typedef long long ll;
    const int N = 2e5 + 5;
    struct TREAP{
        int v, cnt, sz, ch[2], rnd;
    } tr[N * 40];
    int tot;
    int n, m;
    int sum[N];
    void init(){
        tot = 0;
    }
    void up(int x){
        tr[x].sz = tr[tr[x].ch[0]].sz + tr[tr[x].ch[1]].sz + tr[x].cnt;
    }
    void rot(int & x,bool f){
        int t = tr[x].ch[f];
        tr[x].ch[f] = tr[t].ch[!f];
        tr[t].ch[!f] = x;
        up(x);
        up(t);
        x = t;
    }
    void insert(int &x,int v){
        if (!x){
            x = ++tot;
            tr[x].rnd = rand();
            tr[x].v = v;
            tr[x].cnt = tr[x].sz = 1;
            return;
        }
        tr[x].sz++;
        if (tr[x].v==v){
            tr[x].cnt++;
            return;
        }
        bool f = v > tr[x].v;
        insert(tr[x].ch[f], v);
        if (tr[tr[x].ch[f]].rnd<tr[x].rnd)
            rot(x, f);
    }
    int getRank(int x,int v){
        if (!x)
            return 0;
        if (tr[x].v==v)
            return tr[tr[x].ch[0]].sz + tr[x].cnt;
        if (tr[x].v>v)
            return getRank(tr[x].ch[0], v);
        else
            return getRank(tr[x].ch[1], v) + tr[x].cnt + tr[tr[x].ch[0]].sz;
    }
    
}
using namespace TRP;
```

### Trie

```c++
namespace TRIE{
    using namespace std;
    typedef long long ll;
    const int N = 1e6 + 5;
    int tot;
    struct NODE{
        int ne[26], cnt;
        void init(){
            memset(ne, -1, sizeof(ne));
            cnt = 0;
        }
    } tr[N];
    void init(){
        tot = 0;
    }
    void insert(string s){
        int now = 0;
        for (int i = 0; i < s.size();i++)
        {
            int p = s[i] - 'a';
            if (~tr[now].ne[p]){
                now = tr[now].ne[p];
            }else{
                tr[now].ne[p] = ++tot;
                now = tot;
                tr[now].init();
            }
            tr[now].cnt++;
        }
    }
    int query(string s){
        int now = 0;
        for (int i = 0; i < s.size();i++){
            int p = s[i] - 'a';
            if (~tr[now].ne[p]){
                now = tr[now].ne[p];
            }else
                return 0;
        }
        return tr[now].cnt;
    }
}
using namespace TRIE;

/*****************************************************************************
    多个Trie，update要改改
*****************************************************************************/
struct Trie{
	struct NODE{
		int nxt[B];
		int cnt;
		NODE() { memset(nxt, -1, sizeof(nxt)), cnt = 0; }
		int &operator [] (ull i){
			return nxt[i];
		}
	};
	NODE &operator [] (ull i){
		return po[i];
	}
	vector<NODE> po;
	void init(){
		po.clear();
		po.pb(NODE());
	}
	void update(int a){
		int now = 0;
		for (int i = 29; i >= 0; i--)
		{
			bool x = a & (1 << i);
			if (po[now][x]==-1){
				po[now][x] = po.size();
				po.pb(NODE());
			}
			now = po[now][x];
			po[now].cnt++;
		}
	}
};
```

### UFS

```c++
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
```

### 左偏树

```c++
/*
    大根堆
    编号从1开始
    unite返回合并后的跟节点编号
    pop返回弹出堆顶后的根节点编号
 */
#include <bits/stdc++.h>
namespace LHEAP{
    using namespace std;
    const int N = 1e6 + 5;
    struct NODE{
        int v, lc, rc, dis, pa;
        NODE(){}
        NODE(int v, int lc, int rc, int dis, int pa) : v(v), lc(lc), rc(rc), dis(dis), pa(pa){};
    } t[N];
    int n;
    int getf(int x){
        return (x == t[x].pa ? x : t[x].pa = getf(t[x].pa));
    }
    void init(){
        for (int i = 1; i <= n;i++)
        {
            t[i] = NODE(0, 0, 0, 0, i);
        }
    }
    int unite(int x,int y){
        if (!x)
            return y;
        if (!y)
            return x;
        if (t[x].v<t[y].v)
            swap(x, y);
        t[x].rc = unite(t[x].rc, y);
        t[t[x].rc].pa = x;
        if (t[t[x].lc].dis<t[t[x].rc].dis)
            swap(t[x].lc, t[x].rc);
        if (t[x].rc==0)
            t[x].dis = 0;
        else
            t[x].dis = t[t[x].rc].dis + 1;
        return x;
    }
    int pop(int x){
        int l = t[x].lc, r = t[x].rc;
        t[x] = NODE(t[x].v, 0, 0, 0, x);
        t[l].pa = l;
        t[r].pa = r;
        return unite(l, r);
    }
}
using namespace LHEAP;
```

### 线段树

#### 自创

```c++
namespace Seg{
    vector<int> b;
    ll tree[N << 2], lazy[N << 2];

    void init_hash(){
        sort(ALL(b));
        b.erase(unique(ALL(b)), b.end());
    }
    int has(int x){
        return lower_bound(ALL(b), x) - b.begin() + 1;
    }
    void pushdown(int x,int l,int r){
        if (lazy[x]==0)
            return;
        int mid = l + r >> 1;
        tree[x << 1] += lazy[x] * (b[mid - 1] - b[l - 1] + 1);
        tree[x << 1 | 1] += lazy[x] * (b[r - 1] - b[mid] + 1);
        lazy[x << 1] += lazy[x];
        lazy[x << 1 | 1] += lazy[x];
        lazy[x] = 0;
    }
    void update(int x,int l,int r,int ql,int qr){
        if (ql>r || qr<l)
            return;
        if (ql<=l && qr>=r){
            lazy[x]++, tree[x] += (b[r - 1] - b[l - 1] + 1);
            return;
        }
        tree[x] += b[min(qr, r) - 1] - b[max(ql, l) - 1] + 1;
        int mid = l + r >> 1;
        pushdown(x, l, r);
        update(x << 1, l, mid, ql, qr);
        update(x << 1 | 1, mid + 1, r, ql, qr);
    }
    int query(int x,int l,int r,ll k){
        if (l==r) return b[l - 1];
        int mid = l + r >> 1;
        pushdown(x, l, r);
        ll inmid = tree[x] - tree[x << 1] - tree[x << 1 | 1];
        ll avg = inmid == 0 ? 0 : inmid / (b[mid] - b[mid - 1] - 1);
        if (k<=tree[x<<1])
            return query(x << 1, l, mid, k);
        else if (k <= tree[x << 1] + inmid)
            return b[mid - 1] + (k - tree[x << 1]) / avg + ((k - tree[x << 1]) % avg != 0);
        else
            return query(x << 1 | 1, mid + 1, r, k - tree[x << 1] - inmid);
    }
}
```

### KDTREE

```c++
/*
	k, point.l, point.r, point.d is needed
	FUNCTION value is sometimes needed
*/
#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N = 1e5 + 5;
const int K = 6;
const ll inf = 0x7fffffff;
//namespace KDTREE{
int dim;
int k;
bool smaller;
struct NODE{
	ll d[K];
	ll l[K],r[K];					//初始化时要让l=r=d
	int lc, rc;
	bool operator < (const NODE & a) const {
        return d[dim] < a.d[dim];
    }
    bool operator > (const NODE & a) const {
        return a < (*this);
    }
} point[N]; 							//起点从0开始
priority_queue<pair<ll, NODE> > q;
ll square(ll x){
	return x * x;
}
ll distance(NODE a,NODE b){
	ll sum = 0;
	for (int i = 0; i < k;i++)
	{
		sum += square(a.d[i] - b.d[i]);
	}
	return sum;
}
NODE tree[N << 2];
int tot;
void up(int x){
	int lc = tree[x].lc;
	int rc = tree[x].rc;
	for (int i = 0; i < k;i++)
	{
		if (~lc){
			tree[x].l[i] = min(tree[x].d[i], min(tree[x].l[i], tree[lc].l[i]));
			tree[x].r[i] = max(tree[x].d[i], max(tree[x].r[i], tree[lc].r[i]));
		}
		if (~rc){
			tree[x].l[i] = min(tree[x].d[i], min(tree[x].l[i], tree[rc].l[i]));
			tree[x].r[i] = max(tree[x].d[i], max(tree[x].r[i], tree[rc].r[i]));
		}		
	}
}
void build(int l,int r,int dep){	//[l,r]
	tot++;
	int now = tot;
	dim = dep % k;
	int mid = (l + r) / 2;
	nth_element(point + l, point + mid, point + r + 1);
	tree[now] = point[mid];
	tree[now].lc = tree[now].rc = -1;
	if (l<mid){
		tree[now].lc = tot + 1;
		build(l, mid - 1, dep + 1);
	}
	if (mid<r){
		tree[now].rc = tot + 1;
		build(mid + 1, r, dep + 1);
	}
	up(now);
}
void build(int l,int r){
	tot = 0;
	build(l,r,0);
}
ll value(int x,NODE t){			//估价函数：计算到其区域的最小距离
	ll sum = 0;
	if (smaller){
		for (int i = 0; i < k;i++)
		{
			ll d1 = max((ll)0, tree[x].l[i] - t.d[i]);
			ll d2 = max((ll)0, t.d[i] - tree[x].r[i]);
			sum += d1 * d1 + d2 * d2;
		}
	}else{
		for (int i = 0; i < k;i++){
			sum += max(square(tree[x].l[i] - t.d[i]), square(tree[x].r[i] - t.d[i]));
		}
	}
	return sum;
}
void query(int x,NODE t,int m){
	pair<ll, NODE> now((smaller?1:-1)*distance(tree[x],t),tree[x]);
	if (q.size()<m)
		q.push(now);
	else if (now.first<q.top().first)
		q.pop(),q.push(now);
	ll dl = inf, dr = inf;
	int xx = tree[x].lc, yy = tree[x].rc;
	if (~xx)
		dl = (smaller?1:-1)*value(xx, t);
	if (~yy)
		dr = (smaller?1:-1)*value(yy, t);
	if ( dl>dr )
	{
		swap(xx, yy);
		swap(dl, dr);
	}
	if (q.size() < m && ~xx || dl < q.top().first)
		query(xx, t, m);
	if (~yy && q.size() < m || dr < q.top().first)
		query(yy, t, m);
}
vector<pair<ll,NODE> > query(NODE t,int m, bool flag){
	smaller = flag;
	query(1, t, m);
	vector<pair<ll, NODE> > b;
	while (!q.empty())
		b.push_back(q.top()), q.pop();
	return b;
}
//}
```

### AVL

```c++
#include<cstdio>
#include<iostream>
#include<cstring>
#include<string>
#include<cstdlib>
#define N 200005

using namespace std;
struct Node{
    Node *l,*r;
    int h;
    long long key;
    int num;
    int v;
    bool lazy;
};
int n,ans;
long long m,base;
class AVL{
    Node* root;
    int ex;
    Node* findmax(Node *k){
        if (k==NULL) return NULL;
        Node* k1=k;
        while (k1->r!=NULL){
            k1=k1->r;
        }
        return k1;
    }
    Node* findmin(Node *k){
        if (k==NULL) return NULL;
        Node* k1=k;
        while (k1->l!=NULL){
            k1=k1->l;
        }
        return k1;
    }
    void destroy(Node*k){
        if (k==NULL) return;
        if (k->l!=NULL) destroy(k->l);
        if (k->r!=NULL) destroy(k->r);
        delete(k);
    }
    static Node* creat(long long key,Node* l,Node* r){
        Node* t=new Node();
        t->key=key;
        t->l=l;
        t->h=0;
        t->num=1;
        t->v=1;
        t->lazy=0;
        return t;
    }
    static int h(Node* t){
        return (t?t->h:0);
    }
    static int g(Node* t){
        return (t?t->num:0);
    }
    static int me(Node* t){
        return ((t&&t->lazy==0)?t->v:0);
    }
    static Node* llr(Node* k2){
        Node* k1;
        k1=k2->l;
        k2->l=k1->r;
        k1->r=k2;

        k2->h=max(h(k2->l),h(k2->r))+1;
        k1->h=max(h(k1->l),k2->h)+1;

        k2->num=g(k2->l)+g(k2->r)+me(k2);
        k1->num=g(k1->l)+g(k2)+me(k1);
        return k1;
    }
    static Node* rrr(Node* k1){
        Node* k2;
        k2=k1->r;
        k1->r=k2->l;
        k2->l=k1;

        k1->h=max( h(k1->l), h(k1->r) )+1;
        k2->h=max( h(k2->r), k1->h )+1;

        k1->num=g(k1->l)+g(k1->r)+me(k1);
        k2->num=g(k2->r)+g(k1)+me(k2);
        return k2;
    }
    static Node* lrr(Node* k1){
        k1->l=rrr(k1->l);
        return llr(k1);
    }
    static Node* rlr(Node* k1){
        k1->r=llr(k1->r);
        return rrr(k1);
    }
    Node* nsert(Node* k, long long key){

        if (k==NULL){

            k=creat(key,NULL,NULL);

        }else if (key<k->key){

            k->l=nsert(k->l,key);
            if (h(k->l) - h(k->r) == 2){
                if (key< k->l->key)
                    k=llr(k);
                else
                    k=lrr(k);
            }

        }else if (key>k->key){

            k->r=nsert(k->r,key);
            if (h(k->r) - h(k->l) == 2){
                if (key < k->r->key)
                    k=rlr(k);
                else
                    k=rrr(k);
            }

        }else{//same!
            if (k->lazy){
                k->v=1;
                k->lazy=0;
            }else k->v++;

        }
        k->h = max( h(k->l), h(k->r))+1;
        k->num = g(k->l)+g(k->r)+me(k);
        return k;
    }
    void dfs(Node* k,long long x){//返回本树的size？
        if (k==NULL) return;
        if (k->key < x){//删这个点

            if (!k->lazy){
                ans+=k->v;
                k->lazy=1;
            }

            dfs(k->l,x);//左子树size
            dfs(k->r,x);//右子树size
        }else{
            dfs(k->l,x);
        }
        k->num=g(k->r)+g(k->l)+me(k);
    }
public:

    void init(){
        destroy(root);
        root=NULL;
    }

    void insert(long long key){
        root=nsert(root,key);
    }
    void lazy_del(long long x){
        dfs(root,x);
    }
    Node* searchRank(int x){
        if (x>size()) return NULL;
        int sum=x;
        Node* k=root;
        while (k!=NULL){
            if (sum>g(k->l) && sum<=g(k->l)+(k->lazy?0:k->v)) break;
            if (sum<=g(k->l))
                k=k->l;
            else{
                sum-=g(k->l)+(k->lazy?0:k->v);
                k=k->r;
            }
        }
        return k;
    }
    int size(){
        return (root? root->num:0);
    }

}tr;

void work(){
    base=0;
    ans=0;
    tr.init();
    scanf("%d%lld",&n,&m);
    for (int i=0;i<n;i++){
        char c;
        long long x;
        scanf(" %c %lld",&c,&x);
        if (c=='I'){

            if (x>=m)
                tr.insert(x-base);

        }else if (c=='A'){

            base+=x;

        }else if (c=='S'){

            base-=x;
            tr.lazy_del(m-base);

        }else if (c=='Q'){

            Node* p=tr.searchRank(tr.size()-x+1);
            printf("%lld\n",p?p->key + base:-1);

        }
    }
    printf("%d\n",ans);
}
int main(){
    int t;
    scanf("%d",&t);
    while (t--){
        work();
    }
    return 0;
}
```

### LCT

```c++
/*
	节点编号从1开始，
	不存在标志为0
	一棵splay相当于是一条链，因为一个节点的实边只有一条
	splay的中序遍历的节点编号在原树的深度是递增的，因此深度相同的两个节点不可能在一个splay中
 */
namespace LCT{
	using namespace std;
	const int N = 3e5 + 5;
	struct NODE{
		int fa, ch[2], v, sz, mx, tag;
		bool rev;
		NODE(){
			fa = ch[0] = ch[1] = tag = mx = 0;
			rev = 0;
			sz = 1;
		}
	} tr[N];
	bool isroot(int x){
		return tr[tr[x].fa].ch[0] != x && tr[tr[x].fa].ch[1] != x;
	}
	bool isleft(int x){
		return tr[tr[x].fa].ch[0] == x;
	}
	void reverse(int x){
		swap(tr[x].ch[0], tr[x].ch[1]);
		tr[x].rev ^= 1;
	}
	void pushdown(int x){
		int l = tr[x].ch[0], r = tr[x].ch[1];
		if (tr[x].rev){
			if (l)
				reverse(l);
			if (r)
				reverse(r);
			tr[x].rev ^= 1;
		}
		if (tr[x].tag){
			if (l) {
				tr[l].tag += tr[x].tag;
				tr[l].mx += tr[x].tag;
				tr[l].v += tr[x].tag;
			}
			if (r){
				tr[r].tag += tr[x].tag;
				tr[r].mx += tr[x].tag;
				tr[r].v += tr[x].tag;
			}
			tr[x].tag = 0;
		}
	}
	//*********************************************************
	void pushup(int x){
		int l = tr[x].ch[0], r = tr[x].ch[1];
		tr[x].sz = tr[l].sz + tr[r].sz + 1;
		tr[x].mx = max(tr[l].mx, max(tr[r].mx, tr[x].v));
	}
	//*********************************************************
	int st[N];
	void pushto(int x){
		int top = 0;
		while (!isroot(x)){
			st[top++] = x;
			x = tr[x].fa;
		}
		st[top++] = x;
		while (top){
			pushdown(st[--top]);
		}
	}
	
	void rotate(int x){
		bool t = !isleft(x);
		int fa = tr[x].fa, ffa = tr[fa].fa;

		tr[x].fa = ffa;
		if (!isroot(fa)) tr[ffa].ch[!isleft(fa)] = x;

		tr[fa].ch[t] = tr[x].ch[!t];
		tr[tr[fa].ch[t]].fa = fa;

		tr[x].ch[!t] = fa;
		tr[fa].fa = x;
		pushup(fa);
	}
	void splay(int x){
		pushto(x);
		for (int fa = tr[x].fa; !isroot(x);rotate(x),fa=tr[x].fa)
		{
			if (!isroot(fa))
				rotate(isleft(fa) == isleft(x) ? fa : x);
		}
		pushup(x);
	}
	void access(int x){
		for (int p = 0; x; x = tr[p = x].fa)
		{
			splay(x);
			tr[x].ch[1] = p;
			pushup(x);
		}
	}
	void makert(int x){
		access(x);
		splay(x);
		reverse(x);
	}
	int findrt(int x){
		access(x);
		splay(x);
		while (tr[x].ch[0])
			x = tr[x].ch[0];
		return x;
	}
	void split(int x,int y){
		makert(x);
		access(y);
		splay(y);
	}
	void link(int x,int y){
		makert(x);
		tr[x].fa = y;
	}
	void cut(int x,int y){
		split(x, y);
		tr[tr[y].ch[0]].fa = 0;
		tr[y].ch[0] = 0;
		pushup(y);
	}

	//*********************************************************
	void modify(int x,int y,int v){
		split(x, y);
		tr[y].tag += v;
		tr[y].mx += v;
		tr[y].v += v;
	}
	int query(int x,int y){
		split(x, y);
		return tr[y].mx;
	}
	void init(int n){
		for (int i = 0; i <= n;i++)
		{
			tr[i] = NODE();
		}
	}
	//*********************************************************
}
using namespace LCT;
```



## 字符串

### ACAM

```c++
/*
    fail指针指向与当前有最长公共前缀的字符位置，相当于常数优化前的kmp的next
    nxt[x][i]指针指向x节点匹配，但是它的下一个字符i失配时下次直接匹配的位置，它的意义有两个：
        当x的nxt[i]存在时，它的意义仍是Trie的next
        当x的nxt[i]不存在时，它的意义类似优化常数后的kmp的next数组（nxt[i]失配时应该直接匹配的位置）
    因为最长公共前缀end的节点深度一定小于我，实现时使用bfs
*/
struct ACAM{
    int nxt[N][B], fail[N], cnt[N];
    int tot, root;
    int newNODE(){
        tot++;
        for (int i = 0; i < B;i++)
            nxt[tot][i] = -1;
        cnt[tot] = 0;
        return tot;
    }
    void init(){
        tot = 0;
        root = newNODE();
    }
    void update(char *s){
        int len = strlen(s);
        int now = root;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            if (nxt[now][p]==-1)
                nxt[now][p] = newNODE();
            now = nxt[now][p];
        }
        cnt[now]++;
    }
    void build(){
        queue<int> q;
        fail[root] = root;
        for (int i = 0; i < B;i++)
        {
            if (nxt[root][i]==-1)
                nxt[root][i] = root;
            else{
                int x = nxt[root][i];
                fail[x] = root;
                q.push(x);
            }
        }
        while (!q.empty()){
            int now = q.front(); 
            q.pop();
            for (int i = 0; i < B;i++)
            {
                if (nxt[now][i]==-1)
                    nxt[now][i] = nxt[fail[now]][i];
                else{
                    int x = nxt[now][i];
                    fail[x] = nxt[fail[now]][i];
                    q.push(x);
                }
            }
        }
    }
    int query(char *s){
        int len = strlen(s);
        int now = root;
        int res = 0;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            now = nxt[now][p];
            int tmp = now;
            while (tmp!=root){
                res += cnt[tmp];
                cnt[tmp] = 0;
                tmp = fail[tmp];
            }
        }
        return res;
    }
} ac;
```

### PAM

```c++
/***********************************************************************************
    点：除奇/偶根节点外的每一个节点代表一种回文串
    边：nxt[x][c]=y表示节点x表示的回文串左右加上字符c形成的字符串节点为y
    ------------------------------------------------------------------------
    fail[x]：x失配后跳转到的不等于自身的最长后缀回文子串
        若fail[x]=y，y节点串一定是x节点串的后缀
    len[x]：以x为结尾的最长回文子串的长度
    cnt[x]：与以x结尾的最长回文子串相同的子串的个数
    nxt[x][c]：编号为x的节点表示的回文串在两边添加字符c以后变成的回文串的编号
    s[x]：第x次添加的字符（一开始设S[0] = -1，也可以是任意一个在串S中不会出现的字符）
    https://www.cnblogs.com/nbwzyzngyl/p/8260921.html
    https://blog.csdn.net/stevensonson/article/details/81748093
***********************************************************************************/
struct PAM{
    int nxt[N][27]; //next指针，next指针和字典树类似，指向的串为当前串两端加上同一个字符构成
    int fail[N];    //fail指针，失配后跳转到fail指针指向的节点
    int cnt[N];
    int num[N];
    int len[N]; //len[i]表示节点i表示的回文串的长度
    int S[N];   //存放添加的字符
    int last;   //指向上一个字符所在的节点，方便下一次add
    int n;      //字符数组指针
    int p;      //节点指针

    int newnode(int l){ //新建节点
        for (int i = 0; i < 27; ++i)
            nxt[p][i] = 0;
        cnt[p] = 0;
        num[p] = 0;
        len[p] = l;
        return p++;
    }

    void init(){ //初始化
        p = 0;
        newnode(0);
        newnode(-1);
        last = 0;
        n = 0;
        S[n] = -1; //开头放一个字符集中没有的字符，减少特判
        fail[0] = 1;
    }

    int get_fail(int x){ //和KMP一样，失配后找一个尽量最长的
        while (S[n - len[x] - 1] != S[n])
            x = fail[x];
        return x;
    }

    void extend(int c){
        S[++n] = c;
        int cur = get_fail(last); //通过上一个回文串找这个回文串的匹配位置
        if (!nxt[cur][c])
        {                                            //如果这个回文串没有出现过，说明出现了一个新的本质不同的回文串
            int now = newnode(len[cur] + 2);         //新建节点
            fail[now] = nxt[get_fail(fail[cur])][c]; //和AC自动机一样建立fail指针，以便失配后跳转
            nxt[cur][c] = now;
            num[now] = num[fail[now]] + 1;
        }
        last = nxt[cur][c];
        cnt[last]++;
    }

    void count(){
        for (int i = p - 1; i >= 0; --i)
            cnt[fail[i]] += cnt[i];
        //父亲累加儿子的cnt，因为如果fail[v]=u，则u一定是v的子回文串！
    }
};
```

### SAM

```c++
/****************************************************************************
  https://blog.csdn.net/qq_35649707/article/details/66473069
  https://www.cnblogs.com/mangoyang/p/9760416.html
  https://www.cnblogs.com/candy99/p/6374177.html
  
  每个节点表示某一endpos等价类（标记为S）
  连边nxt[x][c]=y表示Sx后面加上字符c就变成了Sy
  fa[x]=y表示Sx是Sy的真子集，x节点的孩子失配时利用fa[x]跳转，查找时相当于放宽了限制
****************************************************************************/
struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }

    int id[N];
    int v[N];
    int sum[N];
    
    void getSum(){
        int mx = 0;
        for (int i = 1; i <= tot; i++)
            mx = max(mx, len[i]), ++v[len[i]];
        for (int i = 1; i <= mx; i++)
            v[i] += v[i - 1];
        for (int i = 1; i <= tot; i++){
            id[v[len[i]]--] = i;
        }
        for (int i = tot; i; i--)
        {
            int t = id[i];
            if (T)
                sz[fa[t]] += sz[t];
            else
                sz[t] = 1;
        }
        // sz[0] = 0;//空串不能算在内
        for (int i = tot; ~i;i--)
        {
            int t = id[i];
            sum[t] = sz[t];
            for (int j = 0; j < 26;j++) if (~nxt[t][j])
            {
                 sum[t] += sum[nxt[t][j]];
            }
        }
    }
} sam;
```

### manacher

```c++
/*
  给定字符串求出s中所有回文串的数目
*/
int countSubstrings(string s) {
    //预处理
    string t = "#";
    for (int i = 0; i < s.size(); ++i) {
        t += s[i];
        t += "#";
    }

    vector<ll> RL(t.size(), 0);
    ll MaxRight = 0, pos = 0;
    ll res = 0;
    for (int i = 0; i < t.size(); ++i) {
        RL[i] = MaxRight > i ? min(RL[2 * pos - i], MaxRight - i) : 1;

        while (i-RL[i] >=0 && i+RL[i] < t.size() && t[i + RL[i]] == t[i - RL[i]])//扩展，注意边界
            ++RL[i];
        //更新最右端及其中心
        if (MaxRight < i + RL[i] -1) {
            MaxRight = i + RL[i] -1;
            pos = i;
        }

        res += RL[i]/2;
    }
    return res;
}
```

### 最大最小表示法

```c++
/*把一个长为len的字符串围成一个圈，然后以任意一个字符作为起点，都会产生一个新的长为len的字符串，
字符串的最小表示就是所有新字符串中字典序最小的那个。下面这个函数就是解决这个问题的，返回值为字典序最小的串的在原串中的起始位置。
基本想法就是两个位置的字符比较，如果s[i+k] > s[j+k]那么i到i+k位置都不是最小表示的位置，所以i直接跳k+1步，反之j直接跳k+1步。*/
int get_min(char *s)
{
    int len = strlen(s) / 2;            //当前s为扩展过的s
    int i = 0, j = 1;
    while (i < len && j < len)
    {
        int k = 0;
        while (s[i + k] == s[j + k] && k < len)
            k++;
        if (k == len)
            break;
        if (s[i + k] < s[j + k])
        {
            if (j + k > i)
                j += k + 1;
            else
                j = i + 1;
        }
        else
        {
            if (i + k > j)
                i += k + 1;
            else
                i = j + 1;
        }
    }
    return min(i, j);
}
int get_max(char *s)
{
    int len = strlen(s) / 2;            //当前s为扩展过的s
    int i = 0, j = 1;
    while (i < len && j < len)
    {
        int k = 0;
        while (s[i + k] == s[j + k])
            k++;
        if (k == len)
            break;
        if (s[i + k] > s[j + k])
        {
            if (j + k > i)
                j += k + 1;
            else
                j = i + 1;
        }
        else
        {
            if (i + k > j)
                i += k + 1;
            else
                i = j + 1;
        }
    }
    return min(i, j);
}
```

### EXKMP

```c++
namespace EXKMP{
	using namespace std;
	typedef long long ll;
	const int N = 1e6 + 5;
	ll extend[N];
	ll nxt[N];
	ll min(ll x, ll y)
	{
		if (x > y)
			return y;
		return x;
	}
	void getNext(string t)
	{
		memset(nxt, 0, sizeof(nxt));
		ll len = t.length();
		nxt[0] = len;
		ll a, p;
		a = 1;
		while (a < len && t[a] == t[a - 1])
			a++; // 求出长度为1的时候 解为多少
		nxt[1] = a - 1;
		a = 1;
		for (ll i = 2; i < len; i++) // 后续的按照算法来就好
		{
			p = a + nxt[a] - 1;
			if ((i - 1) + nxt[i - a] < p)
				nxt[i] = nxt[i - a]; // 第一种情况 没有超过等于的部分
			else					 // 超过的话就不好直接用next的定义 需要后续的遍历
			{
				ll j = (p - i + 1) > 0 ? (p - i + 1) : 0;
				while (i + j < len && t[i + j] == t[j])
					j++;
				nxt[i] = j;
				a = i;
			}
		}
	}
	void exkmp(string s, string t) // s->extend  t->next
	{
		getNext(t);
		ll a, p; //
		ll slen = s.length();
		ll tlen = t.length();
		a = p = 0;
		ll len = min(s.length(), t.length());
		while (p < len && t[p] == s[p])
			p++; // after
		extend[0] = p;
		for (ll i = 1; i < slen; i++)
		{
			p = a + extend[a] - 1; // update
			if ((i - 1) + nxt[i - a] < p)
				extend[i] = nxt[i - a];
			else
			{
				ll j = (p - i + 1) > 0 ? (p - i + 1) : 0;
				while (j < tlen && i + j < slen && s[i + j] == t[j])
					j++;
				extend[i] = j;
				a = i;
			}
		}
	}
}
using namespace EXKMP;
```

## 图论

### Dijkstra_分层

```c++
// Dijkstra 分层
// 可以将k条路权重置为0
 
#include<cstdio>
#include<iostream>
#include<queue>

using namespace std;
const int maxn=1005;
const int maxm=2010;
const int INF = 0x3f3f3f3f;

struct Node{
	int dist;
	int index;
	int l;
	bool operator < (const Node &x) const {
		return dist>x.dist; 
	}
	Node(int x,int y, int z) {
		index=x;
		dist=y;
		l = z;
	}
};
int n,m,k;

struct Edge{
	int x,y;
	int next;
	int w;
} edge[maxm];
int head[maxn];
int tot;

inline void init() {
	for (int i=1;i<=n;i++) {
		head[i]=0;
	}
	tot=0;
}

inline void addedge(int x, int y,int w) {
	tot++;
	edge[tot].next=head[x];
	edge[tot].x=x;
	edge[tot].y=y;
	edge[tot].w=w;
	head[x]=tot;
}

int dis[maxn][maxn];

priority_queue<Node> que;

void Dijkstra(int x) {
	for (int i=1;i<=n;i++) {
		for (int j=0;j<=k;j++) {
			dis[i][j]=INF;
		}
	}
	while (!que.empty()) {
		que.pop();
	}
	dis[x][0]=0;
	que.push(Node(x,0,0));
	while (!que.empty()) {
		Node t = que.top();
		que.pop();
		int u=t.index;
		int dist = t.dist;
		int l = t.l;
		if (dis[u][l]<dist) {
			continue;
		}
		
		for (int i=head[u];i!=0;i=edge[i].next) {
			int v = edge[i].y;
			if (dis[u][l]+edge[i].w<dis[v][l]) {
				dis[v][l] = dis[u][l]+edge[i].w;
				que.push(Node(v,dis[v][l],l));
			}
			if (l<k && dis[u][l]<dis[v][l+1]) {
				dis[v][l+1] = dis[u][l];
				que.push(Node(v,dis[v][l+1],l+1));
			}
		}
//		printf("%d\n", u);
//		for (int i=0;i<=k;i++) {
//			printf("%d ", dis[u][i]);
//		}printf("\n");
	}
	
}


int main() {
	int s, t;
    scanf("%d%d%d%d%d", &n, &m, &s, &t, &k);
    init();
	for (int i = 1; i <= m;i++){
        int x, y, z;
        scanf("%d%d%d", &x, &y, &z);
        addedge(x, y, z);
        addedge(y, x, z);
    }
    Dijkstra(s);
    int ans = INF;
    for (int i = 0; i <= k;i++) {
    	ans = min(dis[t][i], ans);
	}
    printf("%d\n", (s == t ? 0 : ans));
    return 0;
} 
```

### Dinic

```c++
/*
    e为图中编号的最大点
    s=e-1
 */
namespace DINIC{
    using namespace std;
    typedef long long ll;
    const int N = 1e5 + 5;

    int s, e, n, m;
    queue<int> q;
    int dep[N], cur[N];

    int cnt, he[N], v[N], w[N], ne[N];
    void add(int x,int y,int z){
        cnt++;
        ne[cnt] = he[x];
        v[cnt] = y;
        w[cnt] = z;
        he[x] = cnt;
    }
    void init(){
        cnt = -1;
        memset(he, -1, sizeof(he));
        memset(ne, -1, sizeof(ne));
    }

    bool bfs(){
        memset(dep,0,sizeof(dep));
        while (!q.empty()) q.pop();
        dep[s]=1;
        q.push(s);
        while (!q.empty()){
            int j=q.front();
            q.pop();
            for (int i=he[j];i>-1;i=ne[i]){
                int p=v[i];
                if (w[i] && !dep[p]){
                    dep[p]=dep[j]+1;
                    q.push(p);
                }
            }
        }
        if (dep[e]==0) return 0;
        return 1;
    }
    int dfs(int u,int dist){
        if (u==e) return dist;
        else {
            for (int& i=cur[u];i>-1;i=ne[i]){
                int p=v[i];
                if (!w[i] || dep[p]<=dep[u]) continue;
                int di=dfs(p,min(dist,w[i]));
                if (di){
                    w[i]-=di;
                    w[i^1]+=di;
                    return di;
                }
            }
        }
        return 0;
    }
    int Dinic(){
        int ans=0;
        while (bfs()){
            for (int i = 1; i <= e; i++)
            {
                cur[i]=he[i];
            }
            while (int di=dfs(s,INT_MAX)){
                ans+=di;
            }
        }
        return ans;
    }
}
using namespace DINIC;
```

### 二分图

```c++

// poj 1469
#include<cstdio>
#include<cstring>
using namespace std;
const int N=505;
typedef long long ll;
bool maze[N][N],vis[N];
int mat[N];
int p,n;
bool match(int x){
    for (int i=p+1;i<=n+p;i++){
        if (!vis[i] && maze[x][i]){
            vis[i]=1;
            if (mat[i]==-1 || match(mat[i])){
                mat[i]=x;
                mat[x]=i;
                return 1;
            }
        }
    }
    return 0;
}
int getSum(){
    int ans=0;
    for (int i=1;i<=p;i++){
        memset(vis,0,sizeof(vis));
        if (match(i)) ans++;
    }
    return ans;
}
int main(){
    int t;
    scanf("%d",&t);
    while (t--){
        memset(mat,-1,sizeof(mat));
        memset(maze,0,sizeof(maze));
        scanf("%d%d",&p,&n);
        for (int i=1;i<=p;i++){
            int num;
            scanf("%d",&num);
            for (int j=1;j<=num;j++){
                int x;
                scanf("%d",&x);
                maze[i][x+p]=1;
                maze[x+p][i]=1;
            }
        }
        int ans=getSum();
        printf("%s\n",ans==p?"YES":"NO");
    }
    return 0;
}
```

### 边双

```c++
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
```

### 费用流

#### Dijkstra

```c++
// 最小费用最大流 Dijkstra
#include<cstdio>
#include<iostream>
#include<queue>
using namespace std;
const int maxn=10000;
const int maxm=200000;
const int INF = 0x3f3f3f3f;

struct Edge{
	int from;
	int to;
	int next;
	int cap;
	int flow;
	int cost;
} edge[maxm];

int head[maxn];
int tot;
int n,m;
int s,t;

int pre[maxn]; //记录路径 
int dist[maxn]; //到每个点可行增广路的最小费用和 



struct Node{
	int x;
	int dist;
	bool operator < (const Node &p) const {
		return dist>p.dist;
	}
};

priority_queue<Node> que;
int h[maxn];  // 用势能 代替边权 e' = e + h[u] - h[v]


// starting from 0
inline void init() {
	tot=-1;
	for (int i=0;i<=n;i++) {
		head[i]=-1;
		h[i]=0;
	}
}

inline void addedge(int x,int y,int c,int w) {
	tot++;
	edge[tot].next=head[x];
	edge[tot].from=x;
	edge[tot].to=y;
	edge[tot].flow=0;
	edge[tot].cap=c;
	edge[tot].cost=w;
	head[x]=tot;
	tot++;
	edge[tot].next=head[y];
	edge[tot].from=y;
	edge[tot].to=x;
	edge[tot].flow=0;
	edge[tot].cap=0;
	edge[tot].cost=-w;
	head[y]=tot;
}



// dijkstra魔改需要解决反向边负环问题 
// 可以证明不可能有负环，所以只要魔改负权边 

bool Dijkstra() {
	for (int i=0;i<=n;i++) {
		dist[i]=INF;
	}
	while (!que.empty()) {que.pop();}
	Node tmp;
	tmp.dist=0;
	tmp.x=s;
	que.push(tmp);
	dist[s]=0;
	
	while (!que.empty()) {
		tmp = que.top();
		int u=tmp.x;
		
		int dis=tmp.dist;
		que.pop();
		if (dist[u]<dis) {continue;}
		
		for (int i=head[u];~i; i=edge[i].next) {
			int v = edge[i].to;
			if (edge[i].cap>edge[i].flow && dist[v] > dist[u]+ edge[i].cost + h[u]-h[v]){
				dist[v] = dist[u] + edge[i].cost + h[u] - h[v];
				pre[v]=i;
				Node tmp1;
				tmp1.dist = dist[v];
				tmp1.x = v;
				que.push(tmp1);
			}
		}
	}
	for (int i = 0;i <= n;i++) {
		h[i] += dist[i];
	}
	if(dist[t]!=INF) {
		return true;
	} else {
    	return false;
    }
}

int CostFlow(int &flow) { // EK算法 
	int mincost = 0;
	while (Dijkstra()) { // 能找到增广路
	
		int Min = INF;
		for (int i=t;i!=s;i=edge[pre[i]].from) { // 寻找最小流
			Min = min(Min,edge[pre[i]].cap - edge[pre[i]].flow);
		}
		for (int i=t;i!=s; i=edge[pre[i]].from) { //处理所有边 
			edge[pre[i]].flow+=Min;
            edge[pre[i]^1].flow-=Min;
		}
		flow += Min;
		mincost+=(h[t]*Min);
	}
	return mincost;
}

int main() {
	while (~scanf("%d%d%d%d", &n,&m,&s,&t)) {
		
		init();
		for (int i=0;i<m;i++) {
			int x,y,c,w;
			scanf("%d%d%d%d", &x,&y,&c,&w);
			addedge(x,y,c,w);
		}
		int maxFlow = 0;
		int minCost = CostFlow(maxFlow);
		printf("%d %d\n", maxFlow, minCost);
	}
}
```

#### Spfa

```c++
// 最小费用最大流SPFA

#include<cstdio>
#include<iostream>
#include<queue>
using namespace std;
const int maxn=10000;
const int maxm=200000;
const int INF = 0x3f3f3f3f;

struct Edge{
	int from;
	int to;
	int next;
	int cap;
	int flow;
	int cost;
} edge[maxm];

int head[maxn];
int tot;
int n,m;
int s,t;

int pre[maxn]; //记录路径 
int dist[maxn]; //到每个点可行增广路的最小费用和 
int vis[maxn];

// starting from 0
inline void init() {
	tot=-1;
	for (int i=0;i<=n;i++) {
		head[i]=-1;
	}
}

inline void addedge(int x,int y,int c,int w) {
	tot++;
	edge[tot].next=head[x];
	edge[tot].from=x;
	edge[tot].to=y;
	edge[tot].flow=0;
	edge[tot].cap=c;
	edge[tot].cost=w;
	head[x]=tot;
	tot++;
	edge[tot].next=head[y];
	edge[tot].from=y;
	edge[tot].to=x;
	edge[tot].flow=0;
	edge[tot].cap=0;
	edge[tot].cost=-w;
	head[y]=tot;
}

queue<int> que;

// dijkstra魔改需要解决反向边负环问题 
// 可以证明不可能有负环，所以只要魔改负权边 
 
bool SPFA() {
	for (int i=0;i<=n;i++) {
		vis[i]=0;
		dist[i]=INF;
	}
	while (!que.empty()) {que.pop();}
	que.push(s);
	dist[s]=0;
	vis[s]=1;
	while (!que.empty()) {
		int u=que.front();
		que.pop();
		vis[u]=0;
		for (int i=head[u];~i; i=edge[i].next) {
			int v = edge[i].to;
			if (edge[i].cap>edge[i].flow && dist[v]>dist[u] + edge[i].cost) {
				dist[v] = dist[u] + edge[i].cost;
				pre[v]=i;
				if (!vis[v]) {
					vis[v]=1;
					que.push(v);
				}
			}
		}
	}
	if(dist[t]!=INF) {
		return true;
	} else {
    	return false;
    }
}

int CostFlow(int &flow) { // EK算法 
	int mincost = 0;
	while (SPFA()) { // 能找到增广路
		int Min = INF;
		for (int i=t;i!=s;i=edge[pre[i]].from) { // 寻找最小流
			Min = min(Min,edge[pre[i]].cap - edge[pre[i]].flow);
		}
		for (int i=t;i!=s; i=edge[pre[i]].from) { //处理所有边 
			edge[pre[i]].flow+=Min;
            edge[pre[i]^1].flow-=Min;
		}
		flow += Min;
		mincost+=(dist[t]*Min);
	}
	return mincost;
}

int main() {
	while (~scanf("%d%d%d%d", &n,&m,&s,&t)) {
		
		init();
		for (int i=0;i<m;i++) {
			int x,y,c,w;
			scanf("%d%d%d%d", &x,&y,&c,&w);
			addedge(x,y,c,w);
		}
		int maxFlow = 0;
		int minCost = CostFlow(maxFlow);
		printf("%d %d\n", maxFlow, minCost);
	}
}
```

### KM

```c++
// KM 算法
// HDU 2255

// 如果只是想求最大权值匹配而不要求是完全匹配的话，请把各个不相连的边的权值设置为0。
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;

int love[305][305];    // 每个妹子对每个男生的好感度 
int ex_girl[305];      // 每个妹子的期望值
int ex_boy[305];       // 每个男生的期望值
bool vis_girl[305];    // 每一轮匹配匹配过的女生
bool vis_boy[305];     // 每一轮匹配匹配过的男生
int match[305];        // 每个男生匹配到的妹子 如果没有则为-1
int slack[305];        // 每个汉子如果能被妹子倾心最少还需要多少期望值
int n;

bool dfs(int girl){
    vis_girl[girl] = true;
    for (int boy = 0; boy < n; boy++) {
        if (vis_boy[boy]) continue; // 每一轮匹配 每个男生只尝试一次
        int gap = ex_girl[girl] + ex_boy[boy] - love[girl][boy];
        if (gap == 0) {  // 如果符合要求
            vis_boy[boy] = true;
            if (match[boy] == -1 || dfs( match[boy] )) {    // 找到一个没有匹配的男生 或者该男生的妹子可以找到其他人
                match[boy] = girl;
                return true;
            }
        }else{
            slack[boy] = min(slack[boy], gap);  // slack 可以理解为该男生要得到女生的倾心 还需多少期望值 取最小值 备胎的样子
        }
    }
    return false;
}
int KM(){
    memset(match, -1, sizeof match);    // 初始每个男生都没有匹配的女生
    memset(ex_boy, 0, sizeof ex_boy);   // 初始每个男生的期望值为0
    // 每个女生的初始期望值是与她相连的男生最大的好感度
    for (int i = 0; i < n; i++) {
        ex_girl[i] = love[i][0];
        for (int j = 1; j < n; j++) {
            ex_girl[i] = max(ex_girl[i], love[i][j]);
        }
    }
    // 尝试为每一个女生解决归宿问题
    for (int i = 0; i < n; i++) {
        fill(slack, slack + n, INF);    // 因为要取最小值 初始化为无穷大
        while(1){
            // 为每个女生解决归宿问题的方法是 ：如果找不到就降低期望值，直到找到为止
            // 记录每轮匹配中男生女生是否被尝试匹配过
            memset(vis_girl, false, sizeof vis_girl);
            memset(vis_boy, false, sizeof vis_boy);
            if(dfs(i)) break;  // 找到归宿 退出
            // 如果不能找到 就降低期望值
            // 最小可降低的期望值
            int d = INF;
            for (int j = 0; j < n; j++)
                if (!vis_boy[j])    d = min(d, slack[j]);
            for (int j = 0; j < n; j++) {
                // 所有访问过的女生降低期望值
                if (vis_girl[j]) ex_girl[j] -= d;
                // 所有访问过的男生增加期望值
                if (vis_boy[j]) ex_boy[j] += d;
                // 没有访问过的boy 因为girl们的期望值降低，距离得到女生倾心又进了一步！
                else slack[j] -= d;
            }
        }
    }
    // 匹配完成 求出所有配对的好感度的和
    int res = 0;
    for (int i = 0; i < n; i++)
        res += love[match[i]][i];
    return res;
}
int main(){
    while (~scanf("%d", &n)) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                cin>>love[i][j];
        cout<<KM()<<endl;
    }
    return 0;
}
```

### 最小树形图

```c++
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>
typedef long long LL;
using namespace std;
const int MAXV = 100;
const int MAXE = 10000;
const int INF = 0x3f3f3f3f;

//求具有V个点,以root为根节点的图map的最小树形图
int zhuliu(int root, int V, int map[MAXV + 7][MAXV + 7]){
    bool visited[MAXV + 7];
    bool flag[MAXV + 7];//缩点标记为true,否则仍然存在
    int pre[MAXV + 7];//点i的父节点为pre[i]
    int sum = 0;//最小树形图的权值
    int i, j, k;
    for(i = 0; i <= V; i++) flag[i] = false, map[i][i] = INF;
    pre[root] = root;
    while(true){
        for(i = 1; i <= V; i++){//求最短弧集合E0
            if(flag[i] || i == root) continue;
            pre[i] = i;
            for(j = 1; j <= V; j++)
                if(!flag[j] && map[j][i] < map[pre[i]][i])
                    pre[i] = j;
            if(pre[i] == i) return -1;
        }
        for(i = 1; i <= V; i++){//检查E0
            if(flag[i] || i == root) continue;
            for(j = 1; j <= V; j++) visited[j] = false;
            visited[root] = true;
            j = i;//从当前点开始找环
            do{
                visited[j] = true;
                j = pre[j];
            }while(!visited[j]);
            if(j == root)continue;//没找到环
            i = j;//收缩G中的有向环
            do{//将整个环的取值保存,累计计入原图的最小树形图
                sum += map[pre[j]][j];
                j = pre[j];
            }while(j != i);
            j = i;
            do{//对于环上的点有关的边,修改其权值
                for(k = 1; k <= V; k++)
                    if(!flag[k] && map[k][j] < INF && k != pre[j])
                        map[k][j] -= map[pre[j]][j];
                j = pre[j];
            }while(j != i);
            for(j = 1; j <= V; j++){//缩点,将整个环缩成i号点,所有与环上的点有关的边转移到点i
                if(j == i) continue;
                for(k = pre[i]; k != i; k = pre[k]){
                    if(map[k][j] < map[i][j]) map[i][j] = map[k][j];
                    if(map[j][k] < map[j][i]) map[j][i] = map[j][k];
                }
            }
            for(j = pre[i]; j != i; j = pre[j]) flag[j] = true;//标记环上其他点为被缩掉
            break;//当前环缩点结束,形成新的图G',跳出继续求G'的最小树形图
        }
        if(i > V){//如果所有的点都被检查且没有环存在,现在的最短弧集合E0就是最小树形图.累计计入sum,算法结束
            for(i = 1; i <= V; i++)
                if(!flag[i] && i != root) sum += map[pre[i]][i];
            break;
        }
    }
    return sum;
}
```

# Example

## 数据结构

### 启发式合并

#### 重链剖分

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; I; ++I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 3e5 + 5;

struct Q{
    int p, l, r;
    Q(int p, int l, int r) : p(p), l(l), r(r){};
    Q(){}
};
vector<Q> qu[N];

pi V[N];

int cnt, he[N], ne[N], v[N];

int id[N], dfn, sz[N], dfs_in[N], dfs_out[N], son[N];

ll ans[N];

int n;

ll c1[N], c2[N];
int lb(int x) { return x & -x; }
void update(int x,int v){
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

void add(int x,int y){
    cnt++;
    ne[cnt] = he[x];
    v[cnt] = y;
    he[x] = cnt;
}
void dfs1(int x){
    id[dfs_in[x] = ++dfn] = x;
    sz[x] = 1;
    int mx = 0;
    for (int i = he[x]; i;i=ne[i]){
        int p = v[i];
        dfs1(p);
        if (sz[p]>mx){ mx = sz[p], son[x] = p; }
        sz[x] += sz[p];
    }
    dfs_out[x] = dfn;
}

void dfs2(int x){
    for (int i = he[x]; i;i=ne[i]){
        int p = v[i];
        if (p==son[x]) continue;
        dfs2(p);
        for (int j = dfs_in[p]; j <= dfs_out[p];j++)
        {
            int t = id[j];
            update(V[t].fi, -1), update(V[t].sc + 1, 1);
        }
    }
    if (son[x]) dfs2(son[x]);
    for (int i = he[x]; i;i=ne[i])
    {
        int p = v[i];
        if (p==son[x]) continue;
        for (int j = dfs_in[p]; j <= dfs_out[p];j++)
        {
            int t = id[j];
            update(V[t].fi, 1), update(V[t].sc + 1, -1);
        }
    }
    update(V[x].fi, 1), update(V[x].sc + 1, -1);
    for (auto q : qu[x]){
        ans[q.p] = query(q.r) - query(q.l - 1);
    }
}

int main(){
    int m;
    scanf("%d%d", &n, &m);
    V[1] = mp(1, n);
    FOR (i,m){
        int x, y, l, r;
        scanf("%d%d%d%d", &x, &y, &l, &r);
        add(x, y);
        V[y] = mp(l, r);
    }
    dfs1(1);
    int q;
    scanf("%d", &q);
    FOR(i,q){
        int x, l, r;
        scanf("%d%d%d", &x, &l, &r);
        qu[x].pb(Q(i, l, r));
    }
    dfs2(1);
    FOR(i, q) printf("%lld\n", ans[i]);
    return 0;
}
```

#### SAM+重链剖分

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e6 + 5;
const int MOD = 1e9 + 7;

int n, q;

struct Q{
    int p, l, r, k;
    Q(){}
    Q(int p, int l, int r, int k) : p(p), l(l), r(r), k(k){}
};

vector<Q> qu[N];

int fn[N];

int cnt, he[N], ne[N], v[N];

void add(int x,int y){
    cnt++;
    ne[cnt] = he[x];
    v[cnt] = y;
    he[x] = cnt;
}

int tree[N << 2];

int ans[N];

void update(int x,int l,int r,int p,int v){
    if (l==r){
        tree[x] += v;
        tree[x] = min(tree[x], 1);
        return;
    }
    int mid = l + r >> 1;
    if (p<=mid) update(x << 1, l, mid, p, v);
    else update(x << 1 | 1, mid + 1, r, p, v);
    tree[x] = tree[x << 1] + tree[x << 1 | 1];
}
int query(int x,int l,int r,int k){
    if (l==r){
        return l;
    }
    int mid = l + r >> 1;
    if (k <= tree[x << 1]) return query(x << 1, l, mid, k);
    else return query(x << 1 | 1, mid + 1, r, k - tree[x << 1]);
}


struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    int endpos[N];
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        memset(endpos, 0, sizeof(endpos));
        dfn = 0;
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x,int pos)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
        endpos[cur] = pos;
        fn[pos] = cur;
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }
    int dfs_in[N], dfs_out[N], dfn, id[N];
    int son[N];
    int hev[N];
    void dfs1(int x){
        id[dfs_in[x] = ++dfn] = x;
        hev[x] = 1;
        son[x] = 0;
        int mx = 0;
        for (int i = he[x]; i;i=ne[i])
        {
            int p = v[i];
            dfs1(p);
            if (mx<hev[p]){
                son[x] = p;
                mx = hev[p];
            }
            hev[x] += hev[p];
        }
        dfs_out[x] = dfn;
    }
    void dfs2(int x){
        for (int i = he[x]; i; i = ne[i])
        {
            int p = v[i];
            if (p==son[x]) continue;
            dfs2(p);
            for (int i = dfs_in[p]; i <= dfs_out[p];i++)
            {
                int t = id[i];
                if (endpos[t]) update(1, 1, n, endpos[t], -1);
            }
        }
        if (son[x])
            dfs2(son[x]);
        for (int i = he[x]; i;i=ne[i])
        {
            int p = v[i];
            if (p==son[x]) continue;
            for (int i = dfs_in[p];i<=dfs_out[p];i++)
            {
                int t = id[i];
                if (endpos[t]) update(1, 1, n, endpos[t], 1);
            }
        }
        if (endpos[x])
            update(1, 1, n, endpos[x], 1);
        for (auto q: qu[x])
        {
            ans[q.p] = tree[1] >= q.k ? query(1, 1, n, q.k) - (q.r - q.l) : -1;
        }
    }
    void work(){
        REP(i,1,q){
            int l, r, k;
            scanf("%d%d%d", &l, &r, &k);
            int x = fn[r];
            while (len[fa[x]] >= (r - l + 1)){
                x = fa[x];
            }
            qu[x].pb(Q(i, l, r, k));
        }

        for (register int i = 1; i <= tot;++i)
        {
            add(fa[i], i);
        }
        dfs1(0);
        dfs2(0);
    }

} sam;

char s[N];

inline void init(){
    sam.init();
    scanf(" %s", s + 1);
    for (register int i = 1; i <= n;++i)
    {
        sam.extend(s[i] - 'a', i);
    }
    for (register int i = 0; i <= sam.tot;++i)
    {
        qu[i].clear();
    }
    memset(tree, 0, sizeof(tree));
    cnt = 0;
    memset(he, 0, sizeof(he));
}

int main(){
    // clock_t st = clock();
    int _;
    for (scanf("%d", &_); _;--_)
    {
        scanf("%d%d", &n, &q);
        init();
        sam.work();
        REP(i,1,q){
            printf("%d\n", ans[i]);
        }
    }
    // clock_t et = clock();
    // cout << (double)(et - st) / CLOCKS_PER_SEC << endl;
    return 0;
}
```

#### 长链剖分

```c++
#include<bits/stdc++.h>
#define mp make_pair
using namespace std;
typedef long long ll;
const int N = 1e6+5;
int n;
int cnt, he[N], ne[N << 1], v[N << 1];

int len[N], son[N];

int *id, tmp[N << 2], *f[N], ans[N];

void add(int x, int y)
{
    ++cnt;
    ne[cnt] = he[x];
    v[cnt] = y;
    he[x] = cnt;
}
void dfs1(int x,int fa){
    len[x] = 1;
    son[x] = 0;
    for (int i = he[x]; i; i = ne[i])
    {
        int p = v[i];
        if (p==fa)
            continue;
        dfs1(p, x);
        if (len[x] < len[p] + 1)
        {
            len[x] = len[p] + 1;
            son[x] = p;
        }
    }
}

void dfs2(int x,int fa){
    f[x][0] = 1;
    if (son[x])
    {
        f[son[x]] = f[x] + 1;
        dfs2(son[x], x);
        ans[x] = ans[son[x]] + 1;
    }
    for (int i = he[x]; i;i=ne[i])
    {
        int p = v[i];
        if (p==fa || p==son[x])
            continue;
        f[p] = id;
        id += len[p];
        dfs2(p, x);
        for (int j = 0; j < len[p];++j)
        {
            f[x][j + 1] += f[p][j];
            if (j+1<ans[x] && f[x][j+1]>=f[x][ans[x]] || j+1>ans[x] && f[x][j+1]>f[x][ans[x]]){
                ans[x] = j + 1;
            }
        }
        id -= len[p];
    }
    if (f[x][ans[x]]==1)
        ans[x] = 0;
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i < n;++i)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        add(x, y);
        add(y, x);
    }
    dfs1(1, 0);
    f[1] = id = tmp;
    id += len[1];
    dfs2(1, 0);
    for (int i = 1; i <= n;++i)
    {
        printf("%d\n", ans[i]);
    }
    return 0;
}
```



### 线性基

#### +线段树

```
// #pragma GCC optimize(2)
#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define sc second
#define fi first
using namespace std;
typedef long long ll;
typedef unsigned int uint;
typedef pair<int, int> pi;
const int N = 5e4 + 5;
const int B = 31;
struct Base{
    uint b[B + 3];
    Base() { init(); }
    uint &operator [] (uint i) {
        return b[i];
    }
    void init() { memset(b, 0, sizeof(b)); }
    void update(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                b[i] = x;
                break;
            }
            x ^= b[i];
        }
    }
    bool check(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                break;
            }
            x ^= b[i];
        }
        return (x == 0);
    }
    static Base Merge(Base a, Base b){
        Base c, tmp_b, tmp_k;
        for (int i = 0; i <= B; i++)
        {
            tmp_b[i] = tmp_k[i] = 0;
            if (a[i]) tmp_b[i] = a[i];
        }
        for (int i = 0; i <= B; i++) c[i] = 0;
        for (int i = 0; i <= B; i++) if (b[i])
        {
            int ok = 1;
            uint v = b[i], k = 0;
            for (int j = B; ~j; j--)
            {
                if (v & (uint(1) << j)){
                    if (tmp_b[j]){
                        v ^= tmp_b[j];
                        k ^= tmp_k[j];
                    }else{
                        tmp_b[j] = v;
                        tmp_k[j] = (uint(1) << i) ^ k;
                        ok = 0;
                        break;
                    }
                }
            }
            if (ok){
                uint v = b[i];
                for (int j = 0; j <= B; j++)
                {
                    if (k & (uint(1) << j))
                    {
                        v ^= b[j];
                    }
                }
                for (int j = B; ~j; j--)
                {
                    if (v & (uint(1) << j)){
                        if (c[j]){
                            v ^= c[j];
                        }else{
                            c[j] = v;
                            break;
                        }
                    }
                }
            }
        }
        return c;
    }
} tree[N << 2];
void up(int x){
    tree[x] = Base::Merge(tree[x << 1], tree[x << 1 | 1]);
}
void build(int x,int l,int r){
    if (l==r){
        int sz;
        scanf("%d", &sz);
        while (sz--){
            uint v; scanf("%u", &v);
            tree[x].update(v);
        }
        return;
    }
    int mid = l + r >> 1;
    build(2 * x, l, mid);
    build(2 * x + 1, mid + 1, r);
    up(x);
}
bool query(int x,int l,int r,int ql,int qr,int v){
    if (ql>r || qr<l) return 1;
    if (ql<=l && qr>=r){
        return tree[x].check(v);
    }
    int mid = l + r >> 1;
    return query(2 * x, l, mid, ql, qr, v) && query(2 * x + 1, mid + 1, r, ql, qr, v);
}
int main(){
    int n, m;
    scanf("%d%d", &n, &m);
    build(1, 1, n);
    while (m--){
        int l, r, v;
        scanf("%d%d%d", &l, &r, &v);
        printf("%s\n", query(1, 1, n, l, r, v) ? "YES" : "NO");
    }
    return 0;
}
```

#### +贪心

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e6 + 5;
const int B = 30;
int f[N][B + 3], pos[N][B + 3];
void init(){
    memset(f, 0, sizeof(f));
    memset(pos, 0, sizeof(pos));
}
void add(int i,int x){
    int k = i, tmp;
    for (int j = B; ~j;j--) f[i][j] = f[i - 1][j], pos[i][j] = pos[i - 1][j];
    for (int j = B; ~j;j--) if (x>>j)
    {
        if (!f[i][j]){
            f[i][j] = x;
            pos[i][j] = k;
            break;
        }else{
            if (k>pos[i][j]){
                tmp = k, k = pos[i][j], pos[i][j] = tmp;
                tmp = x, x = f[i][j], f[i][j] = tmp;
            }
            x ^= f[i][j];
        }
    }
}
int main(){
    // freopen("E:\\vscode\\c++\\in.txt", "r", stdin);
    // freopen("E:\\vscode\\c++\\out.txt", "w", stdout);
    int t, n, m;
    scanf("%d", &t);
    while (t--){
        init();
        scanf("%d%d", &n, &m);
        for (int i = 1; i <= n;i++)
        {
            int x;
            scanf("%d", &x);
            add(i, x);
        }
        int ans = 0;
        while(m--){
            int op, l, r;
            scanf("%d%d", &op, &l);
            if (op==0){
                scanf("%d", &r);
                l = (ans ^ l) % n + 1;
                r = (ans ^ r) % n + 1;
                if (l>r)
                    swap(l, r);
                ans = 0;
                for (int i = B; ~i;i--) if ((ans^f[r][i])>ans && pos[r][i]>=l) ans ^= f[r][i];
                printf("%d\n", ans);
            }else{
                l = (l ^ ans);
                add(++n, l);
            }
        }
    }
    // fclose(stdin);
    // fclose(stdout);
    return 0;
}
```

### 树套树

#### BIT+Treap

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 2e5 + 5;
struct TREAP{
	int v, cnt, sz, ch[2], rnd;
} tr[N * 40];
int root[N], tot;
struct NODE{
	int x, y, z;
	bool operator < (const NODE & b) const {
		return (x == b.x ? (y == b.y ? (z < b.z) : (y < b.y)) : (x < b.x));
	}
	bool operator == (const NODE & b) const {
		return (x == b.x && y == b.y && z == b.z);
	}
} a[N];
int n, m;
int sum[N], ans[N];
void init(){
	tot = 0;
}
void up(int x){
	tr[x].sz = tr[tr[x].ch[0]].sz + tr[tr[x].ch[1]].sz + tr[x].cnt;
}
void rot(int & x,bool f){
	int t = tr[x].ch[f];
	tr[x].ch[f] = tr[t].ch[!f];
	tr[t].ch[!f] = x;
	up(x);
	up(t);
	x = t;
}
void insert(int &x,int v){
	if (!x){
		x = ++tot;
		tr[x].rnd = rand();
		tr[x].v = v;
		tr[x].cnt = tr[x].sz = 1;
		return;
	}
	tr[x].sz++;
	if (tr[x].v==v){
		tr[x].cnt++;
		return;
	}
	bool f = v > tr[x].v;
	insert(tr[x].ch[f], v);
	if (tr[tr[x].ch[f]].rnd<tr[x].rnd)
		rot(x, f);
}
int getRank(int x,int v){
	if (!x)
		return 0;
	if (tr[x].v==v)
		return tr[tr[x].ch[0]].sz + tr[x].cnt;
	if (tr[x].v>v)
		return getRank(tr[x].ch[0], v);
	else
		return getRank(tr[x].ch[1], v) + tr[x].cnt + tr[tr[x].ch[0]].sz;
}
int lowbit(int x){
	return x & (-x);
}
void update(int x,int p){
	for (int i = x; i <= m; i += lowbit(i))
		insert(root[i], p);
}
int query(int x,int p){
	int sum = 0;
	for (int i = x; i; i -= lowbit(i))
		sum += getRank(root[i], p);
	return sum;
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d%d%d", &a[i].x, &a[i].y, &a[i].z);
	}
	sort(a + 1, a + 1 + n);
	for (int i = 1; i <= n;i++)
	{
		if (a[i]==a[i+1])
			sum[i + 1] = sum[i] + 1;
		else{
			int tmp = query(a[i].y, a[i].z);
			ans[tmp] += sum[i] + 1;
		}
		update(a[i].y, a[i].z);
	}
	for (int i = 0; i < n;i++)
	{
		printf("%d\n", ans[i]);
	}
	return 0;
}
```

#### BIT维护最近点对

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 2e5 + 5;
 
struct BIT{
    vector<vector<vector<int>>> c;
    int n, m, h;
    void init(int _n, int _m, int _h){
        n = _n, m = _m, h = _h;
        c = vector<vector<vector<int>>>(n + 3, vector<vector<int>>(m + 3, vector<int>(h + 3, -1)));
    }
    void add(int x, int y, int z, int v){
        for (int i = x; i <= n; i += i & -i)
            for (int j = y; j <= m; j += j & -j)
                for (int k = z; k <= h; k += k & -k)
                    c[i][j][k] = max(c[i][j][k], v);
    }
    int query(int x,int y,int z){
        int res = -1;
        for (int i = x; i > 0; i -= i & -i)
            for (int j = y; j > 0; j -= j & -j)
                for (int k = z; k > 0; k -= k & -k)
                    res = max(c[i][j][k], res);
        return res;
    }
} t[8];

 
int n, m, h, q;
int a[3], b[3], c[3];
void read(int &x){
	char ch; x = 0;
	ch = getchar();
	while (ch>'9' || ch<'0') ch = getchar();
	while (ch<='9' && ch>='0')
		x = x * 10 + ch - '0', ch = getchar();
}

int main(){
    scanf("%d%d%d%d", &n, &m, &h, &q);
    c[0] = n, c[1] = m, c[2] = h;
	FOR(i, 8) t[i].init(n, m, h);
	while (q--){
        int op;
		read(op); read(a[0]); read(a[1]); read(a[2]);
		int ans = 1e9;
        FOR(cas,8){
            FOR(i, 3) b[i] = a[i] * (cas >> i & 1 ? -1 : 1);
            FOR(i, 3) if (b[i] < 0) b[i] += c[i] + 1;
            if (op==1){
                t[cas].add(b[0], b[1], b[2], b[0] + b[1] + b[2]);
                // printf("%d: %d\n", cas, t[cas].query(n, m, h));
            }else{
                int res = t[cas].query(b[0], b[1], b[2]);
                if (~res)
                    ans = min(ans, b[0] + b[1] + b[2] - res);
            }
        }
        if (op==2) printf("%d\n", ans);
    }
    return 0;
}
```

#### 树上主席树

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
int n, m;
int a[N * 2];
int len;
pair<pair<int,int>,int> qu[N];

int cnt, he[N], w[N*2], v[N*2], ne[N*2];

int dp[N * 4][26], seq[N * 4], first[N], dfsn, dep[N];

int tree[N], sum[N * 40], lc[N * 40], rc[N * 40], tot;

void add(int x,int y,int z){
    cnt++;
    v[cnt] = y;
    w[cnt] = z;
    ne[cnt] = he[x];
    he[x] = cnt;
}
void build(int &rt,int l,int r){
    rt = ++tot;
    sum[rt] = 0;
    lc[rt] = rc[rt] = -1;
    int mid = l + r >> 1;
    if (l==r){
        return;
    }
    build(lc[rt], l, mid);
    build(rc[rt], mid + 1, r);
}
void insert(int last,int l,int r,int v,int &rt){
    rt = ++tot;
    lc[rt] = lc[last];
    rc[rt] = rc[last];
    sum[rt] = sum[last] + 1;
    int mid = l + r >> 1;
    if (l==r)
        return;
    if (v<=mid)
        insert(lc[last], l, mid, v, lc[rt]);
    else
        insert(rc[last], mid + 1, r, v, rc[rt]);
}
int query(int last,int l,int r,int k,int now){
    int mid = l + r >> 1;
    if (l>k)
        return 0;
    if (r<=k)
        return sum[now] - sum[last];
    return query(lc[last], l, mid, k, lc[now]) + query(rc[last], mid + 1, r, k, rc[now]);
}
void dfs(int x,int fat,int d){
    dep[x] = d;
    seq[++dfsn] = x;
    first[x] = dfsn;
    for (int i = he[x]; i; i = ne[i])
    {
        int p = v[i];
        if (p==fat)
            continue;
        int k = lower_bound(a + 1, a + 1 + len, w[i]) - a;
        insert(tree[x], 1, len, k, tree[p]);
        dfs(p, x, d + 1);
        seq[++dfsn] = x;
    }
}
void ST(){
    for (int i = 1; i <= dfsn;i++)
    {
        dp[i][0] = seq[i];
    }
    for (int j = 1; (1 << j) <= dfsn;j++)
    {
        for (int i = 1; i + (1 << j) - 1 <= dfsn; i++)
        {
            int a = dp[i][j - 1];
            int b = dp[i + (1 << (j - 1))][j - 1];
            dp[i][j] = dep[a] < dep[b] ? a : b;
        }
    }
}
int lca(int x,int y){
    int ix = first[x], iy = first[y];
    if (ix>iy)
        swap(ix, iy);
    int k = 0;
    while((1<<(k+1))<=iy-ix+1)
        k++;
    int a = dp[ix][k];
    int b = dp[iy - (1 << k) + 1][k];
    return dep[a] < dep[b] ? a : b;
}
int main(){
    len = 0;
    scanf("%d%d", &n, &m);
    for (int i = 1; i < n;i++)
    {
        int x, y, v;
        scanf("%d%d%d", &x, &y, &v);
        add(x, y, v);
        add(y, x, v);
        a[++len] = v;
    }
    for (int i = 1; i <= m;i++)
    {
        scanf("%d%d%d", &qu[i].first.first, &qu[i].first.second, &qu[i].second);
        a[++len] = qu[i].second;
    }
    sort(a + 1, a + 1 + len);
    len = unique(a + 1, a + 1 + len) - a - 1;
    tot = 0;
    build(tree[1], 1, len);
    dfs(1, 0, 0);
    ST();
    for (int i = 1; i <= m;i++)
    {
        int x, y, k;
        x = qu[i].first.first;
        y = qu[i].first.second;
        k = qu[i].second;
        k = lower_bound(a + 1, a + len + 1, k) - a;
        int ac = lca(x, y);
        int tmp1 = query(tree[ac], 1, len, k, tree[x]);
        int tmp2 = query(tree[ac], 1, len, k, tree[y]);
        printf("%d\n", tmp1 + tmp2);
    }

    return 0;
}
```

#### 树上Trie

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;

int n;

int T[N], tot, nex[N * 10][30], cot[N * 10];

int cnt, he[N], ne[N * 2], v[N * 2];
string w[N * 2];

int dep[N], f[N][30];

void init(){
    cnt = 0;
    T[0] = 0;
    tot = 0;
    memset(he, 0, sizeof(he));
}
void add(int x,int y,string z){
    cnt++;
    v[cnt] = y;
    ne[cnt] = he[x];
    w[cnt] = z;
    he[x] = cnt;
}
void insert(int &rt,int la,string s,int p){
    rt = ++tot;
    int now = tot;
    cot[now] = cot[la] + 1;
    for (int i = 0; i < 26;i++)
    {
        nex[now][i] = nex[la][i];
    }
    if (p+1 < s.size())
        insert(nex[now][s[p + 1] - 'a'], nex[la][s[p + 1] - 'a'], s, p + 1);
}
void dfs(int x,int fa,int dp){
    f[x][0] = fa;
    dep[x] = dp;
    for (int i = he[x]; i;i=ne[i])
    {
        int p = v[i];
        if (p==fa)
            continue;
        insert(T[p], T[x], w[i], -1);
        dfs(p, x, dp + 1);
    }
}
int query(int now,string s){
    int sum = 0;
    for (int i = 0; i < s.size();i++)
    {
        int p = s[i] - 'a';
        if (nex[now][p]){
            now = nex[now][p];
        }else
            return 0;
    }
    return cot[now];
}
void LCA(){
    for (int j = 1; (1 << j) <= n;j++)
    {
        for (int i = 1; i <= n;i++)
        {
            f[i][j] = f[f[i][j - 1]][j - 1];
        }
    }
}
int getlca(int x,int y){
    if (dep[x]<dep[y])
        swap(x, y);
    for (int j = 24; j >= 0;j--)
    {
        int fx = f[x][j];
        if (dep[fx]>=dep[y]){
            x = fx;
        }
    }
    for (int j = 24; j >= 0;j--)
    {
        int fx = f[x][j];
        int fy = f[y][j];
        if (fx==fy)
            continue;
        x = fx;
        y = fy;
    }
    if (x!=y){
        x = f[x][0];
    }
    return x;
}

char s[100];
int main(){
    scanf("%d", &n);
    init();
    for (int i = 1; i < n;i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        scanf(" %s", s);
        add(x, y, s);
    }
    dfs(1, 0, 1);
    LCA();
    int m;
    scanf("%d", &m);
    for (int i = 1; i <= m;i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        scanf(" %s", s);
        int lca = getlca(x, y);
        int a1 = query(T[x], s);
        int a2 = query(T[y], s);
        int a3 = query(T[lca], s);
        printf("%d\n", a1 + a2 - 2 * a3);
    }
    return 0;
}
```

#### 偏序问题

##### 三维

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 2e5 + 5;
struct TREAP{
	int v, cnt, sz, ch[2], rnd;
} tr[N * 40];
int root[N], tot;
struct NODE{
	int x, y, z;
	bool operator < (const NODE & b) const {
		return (x == b.x ? (y == b.y ? (z < b.z) : (y < b.y)) : (x < b.x));
	}
	bool operator == (const NODE & b) const {
		return (x == b.x && y == b.y && z == b.z);
	}
} a[N];
int n, m;
int sum[N], ans[N];
void init(){
	tot = 0;
}
void up(int x){
	tr[x].sz = tr[tr[x].ch[0]].sz + tr[tr[x].ch[1]].sz + tr[x].cnt;
}
void rot(int & x,bool f){
	int t = tr[x].ch[f];
	tr[x].ch[f] = tr[t].ch[!f];
	tr[t].ch[!f] = x;
	up(x);
	up(t);
	x = t;
}
void insert(int &x,int v){
	if (!x){
		x = ++tot;
		tr[x].rnd = rand();
		tr[x].v = v;
		tr[x].cnt = tr[x].sz = 1;
		return;
	}
	tr[x].sz++;
	if (tr[x].v==v){
		tr[x].cnt++;
		return;
	}
	bool f = v > tr[x].v;
	insert(tr[x].ch[f], v);
	if (tr[tr[x].ch[f]].rnd<tr[x].rnd)
		rot(x, f);
}
int getRank(int x,int v){
	if (!x)
		return 0;
	if (tr[x].v==v)
		return tr[tr[x].ch[0]].sz + tr[x].cnt;
	if (tr[x].v>v)
		return getRank(tr[x].ch[0], v);
	else
		return getRank(tr[x].ch[1], v) + tr[x].cnt + tr[tr[x].ch[0]].sz;
}
int lowbit(int x){
	return x & (-x);
}
void update(int x,int p){
	for (int i = x; i <= m; i += lowbit(i))
		insert(root[i], p);
}
int query(int x,int p){
	int sum = 0;
	for (int i = x; i; i -= lowbit(i))
		sum += getRank(root[i], p);
	return sum;
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d%d%d", &a[i].x, &a[i].y, &a[i].z);
	}
	sort(a + 1, a + 1 + n);
	for (int i = 1; i <= n;i++)
	{
		if (a[i]==a[i+1])
			sum[i + 1] = sum[i] + 1;
		else{
			int tmp = query(a[i].y, a[i].z);
			ans[tmp] += sum[i] + 1;
		}
		update(a[i].y, a[i].z);
	}
	for (int i = 0; i < n;i++)
	{
		printf("%d\n", ans[i]);
	}
	return 0;
}
```

##### 四维

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 5e5 + 5;

struct NODE
{
    int x, y, z, id, kind;
    NODE() {}
    NODE(int x, int y, int z, int k, int t) : x(x), y(y), z(z), kind(k), id(t){};
};
vector<NODE> q, q1, q2;
vector<int> vc;
int n;
int c[N], ans[N];

bool cmp(NODE a, NODE b)
{
    if (a.x == b.x)
        return a.id < b.id;
    return a.x < b.x;
}

bool cmp1(NODE a, NODE b)
{
    if (a.y == b.y)
        return a.id < b.id;
    return a.y < b.y;
}
int lowbit(int x)
{
    return x & (-x);
}
void add(int x, int v)
{
    for (int i = x; i <= vc.size(); i += lowbit(i))
    {
        c[i] += v;
    }
}
int query(int x)
{
    int sum = 0;
    for (int i = x; i > 0; i -= lowbit(i))
    {
        sum += c[i];
    }
    return sum;
}
void countstar()
{
    for (int i = 0; i < q2.size(); i++)
    {
        if (q2[i].kind == 0)
            add(q2[i].z, 1);
        else
            ans[q2[i].id] += q2[i].kind * query(q2[i].z);
    }
    for (int i = 0; i < q2.size(); i++)
        if (q2[i].kind == 0)
            add(q2[i].z, -1);
}

void cdq1(int l, int r)
{
    if (l >= r)
        return;
    int mid = (l + r) >> 1;
    cdq1(l, mid);
    cdq1(mid + 1, r);
    q2.clear();
    for (int i = l; i <= mid; i++)
        if (q1[i].kind == 0)
            q2.push_back(q1[i]);
    for (int i = mid + 1; i <= r; i++)
        if (q1[i].kind)
            q2.push_back(q1[i]);
    sort(q2.begin(), q2.end(), cmp1);
    countstar();
}

void cdq(int l, int r)
{
    if (l >= r)
        return;
    int mid = (l + r) >> 1;
    cdq(l, mid);
    cdq(mid + 1, r);
    q1.clear();
    for (int i = l; i <= mid; i++)
        if (q[i].kind == 0)
            q1.push_back(q[i]);
    for (int i = mid + 1; i <= r; i++)
        if (q[i].kind)
            q1.push_back(q[i]);
    sort(q1.begin(), q1.end(), cmp);
    cdq1(0, q1.size() - 1);
}
void init()
{
    q.clear();
    vc.clear();
    memset(c, 0, sizeof(c));
}
void work()
{
    init();
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        int fl, x1, y1, z1;
        scanf("%d%d%d%d", &fl, &x1, &y1, &z1);
        if (fl == 1)
        {
            q.push_back(NODE(x1, y1, z1, 0, i));
            vc.push_back(z1);
            ans[i] = -1;
        }
        else
        {
            int x2, y2, z2;
            scanf("%d%d%d", &x2, &y2, &z2);
            x1--;
            y1--;
            z1--;
            q.push_back(NODE(x1, y1, z1, -1, i));
            q.push_back(NODE(x2, y2, z2, 1, i));
            q.push_back(NODE(x1, y2, z2, -1, i));
            q.push_back(NODE(x2, y1, z2, -1, i));
            q.push_back(NODE(x2, y2, z1, -1, i));
            q.push_back(NODE(x2, y1, z1, 1, i));
            q.push_back(NODE(x1, y2, z1, 1, i));
            q.push_back(NODE(x1, y1, z2, 1, i));
            vc.push_back(z1);
            vc.push_back(z2);
            ans[i] = 0;
        }
    }
    sort(vc.begin(), vc.end());
    vc.erase(unique(vc.begin(), vc.end()), vc.end());
    for (int i = 0; i < q.size(); i++)
    {
        q[i].z = lower_bound(vc.begin(), vc.end(), q[i].z) - vc.begin() + 1;
    }
    cdq(0, q.size() - 1);
    for (int i = 0; i < n; i++)
    {
        if (~ans[i])
            printf("%d\n", ans[i]);
    }
}
int main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        work();
    }
    return 0;
}
```



### 莫队

#### 静态

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 5e4 + 5;

struct Q{
	ll l, r, p;
} query[N], ans[N];
int a[N], cnt[N];
int S, n, m;
ll now;
bool cmp(Q a, Q b){
	return (a.l / S == b.l / S ? a.r < b.r : a.l / S < b.l / S);
}
ll gcd(ll a,ll b){
	return b ? gcd(b, a % b) : a;
}

void move(int pos,int v){
	ll tmp = cnt[a[pos]];
	now -= tmp * (tmp - 1) / 2;

	cnt[a[pos]] += v;

	tmp = cnt[a[pos]];
	now += tmp * (tmp - 1) / 2;
}
int main(){
	scanf("%d%d", &n, &m);
	S = sqrt(n);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d",&a[i]);
	}
	for (int i = 1; i <= m;i++)
	{
		scanf("%d%d", &query[i].l, &query[i].r);
		query[i].p = i;
	}
	sort(query + 1, query + 1 + m, cmp);
	int l = 1, r = 1;
	now = 0;
	cnt[a[1]]++;
	for (int i = 1; i <= m;i++)
	{
		while(l<query[i].l)
			move(l++, -1);
		while(l>query[i].l)
			move(--l, 1);
		while(r<query[i].r)
			move(++r, 1);
		while(r>query[i].r)
			move(r--, -1);
		ll len = r - l + 1;
		ll d = gcd(now, len * (len - 1) / 2);
		ans[query[i].p].l = now / d;
		ans[query[i].p].r = len * (len - 1) / 2 / d;
	}
	for (int i = 1; i <= m;i++)
	{
		printf("%lld/%lld\n", ans[i].l, ans[i].r);
	}
	return 0;
}
```

#### 带修改莫队

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e4 + 5;

struct Q{
	int l, r, t, p;
	Q(){}
	Q(int l,int r,int t,int p):l(l),r(r),t(t),p(p){}
} query[N];
struct C{
	int p, v, last;
	C(){}
	C(int p,int v,int last):p(p),v(v),last(last){}
} change[N];
int cntc, cntq;
int ans[N], now[N], vis[N * 100], S, n, m;
int l, r, cur, ti;
int a[N];
bool cmp(Q a, Q b){
	if (a.l/S==b.l/S)
		if (a.r/S==b.r/S)
			return a.t < b.t;
		else
			return a.r / S < b.r / S;
	else
		return a.l / S < b.l / S;
}
void moveTime(int id,int v){
	int pos = change[id].p;
	if (~v){//v=1
		if (l<=pos && pos<=r){
			if (vis[a[pos]]==1)
				cur--;
			if (!vis[change[id].v])
				cur++;
			vis[a[pos]]--;
			vis[change[id].v]++;
		}
			
		a[pos] = change[id].v;
	}else{
		if (l<=pos && pos<=r){
			if (vis[a[pos]]==1)
				cur--;
			if (!vis[change[id].last])
				cur++;
			vis[a[pos]]--;
			vis[change[id].last]++;
		}
		a[pos] = change[id].last;
	}
}
void move(int pos,int v){
	if (~v && !vis[a[pos]]){
		cur++;
	}
	if (v==-1 && vis[a[pos]]==1){
		cur--;
	}
	vis[a[pos]] += v;
}
int main(){
	scanf("%d%d", &n, &m);
	S = sqrt(n);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d", &a[i]);
		now[i] = a[i];
	}
	for (int i = 0; i < m;i++)
	{
		char ch;
		int x, y;
		scanf(" %c%d%d", &ch, &x, &y);
		if (ch=='Q'){
			query[cntq++] = Q(x, y, cntc, cntq);
		}else{
			change[++cntc] = C(x, y, now[x]);
			now[x] = y;
		}
	}
	sort(query, query + cntq, cmp);
	l = 1, r = 0;
	cur = 0;
	ti = 0;
	for (int i = 0; i < cntq;i++)
	{
		while(ti<query[i].t)
			moveTime(++ti, 1);
		while (ti>query[i].t)
			moveTime(ti--, -1);
		while (l<query[i].l)
			move(l++, -1);
		while (l>query[i].l)
			move(--l, 1);
		while (r<query[i].r)
			move(++r, 1);
		while (r>query[i].r)
			move(r--, -1);
		ans[query[i].p] = cur;
	}
	for (int i = 0; i < cntq;i++)
	{
		printf("%d\n", ans[i]);
	}
	return 0;
}
```



### 线段树

#### 线段树+凸包

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e5 + 5;
const ll R = 1e17;

struct Point{
	ll x, y;
	Point(){}
	Point(ll x, ll y) : x(x), y(y){}
	Point operator + (const Point& b) const {
		return Point(x + b.x, y + b.y);
	}
	Point operator - (const Point& b) const {
		return Point(x - b.x, y - b.y);
	}
	ll operator * (const Point& b) const {
		return x * b.y - b.x * y;
	}
};
struct Convex{
	vector<Point> st;
	Convex() { st.clear(); }
	static ll cross_product(const Point& p2,const Point& p0,const Point& p1){
		return (p2 - p0) * (p1 - p0);
	}
	void insert(const Point& a){
		int sz = st.size();
		while (sz > 1 && cross_product(a, st[sz - 2], st[sz - 1]) <= 0)
		{
			st.pop_back();
			--sz;
		}
		st.push_back(a);
	}
	bool query(const Point& a,const Point& b){
		int l = 0, r = int(st.size()) - 2;
		while (l<r){
			int mid = l + r >> 1;
			if (cross_product(st[mid], a, b) < cross_product(st[mid+1], a, b)){
				r = mid;
			}else{
				l = mid + 1;
			}
		}
		return cross_product(st[l], a, b) < 0 || cross_product(st[l + 1], a, b) < 0;
	}
};
Convex tree[N << 2];
Point p[N];
void build(int x,int l,int r){
	tree[x] = Convex();
	REP(i, l, r + 1) tree[x].insert(p[i]);
	if (l==r) return;
	int mid = l + r >> 1;
	build(x << 1, l, mid);
	build(x << 1 | 1, mid + 1, r);
}
int query(int x,int l,int r,int ql,int qr,const Point& a,const Point& b){
	if (l>qr || r<ql ||l>r)
		return 0;
	if (ql<=l && qr>=r){
		if (!tree[x].query(a,b))
			return 0;
		if (l==r)
			return l;
	}
	int mid = l + r >> 1;
	int res = query(x << 1, l, mid, ql, qr, a, b);
	return res ? res : query(x << 1 | 1, mid + 1, r, ql, qr, a, b);
}
int main(){
	int _;
	for (scanf("%d", &_); _;_--)
	{
		int n;
		scanf("%d", &n);
		REP(i,1,n) scanf("%lld%lld", &p[i].x, &p[i].y);
		build(1, 1, n - 1);
		REP(i, 1, n - 2){
			printf("%d ", query(1, 1, n - 1, i + 1, n - 1, p[i], p[i + 1]));
		}
		puts("0");
	}
	return 0;
}
```

#### 区间离散化线段树

```c++
// #pragma GCC optimize(3)
#include <bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; I; ++I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 2e5 + 5;
const int INF = 1e9;
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

struct EDGE{
    int x, y, l, r;
    EDGE(){}
    EDGE(int x, int y, int l, int r) : x(x), y(y), l(l), r(r){}
};
vector<int> tree[N << 2];
vector<EDGE> E;
vector<int> b;
int ans = 0;
int has(int x){
    return lower_bound(ALL(b), x) - b.begin() + 1;
}
void update(int x,int l,int r,int ql,int qr,int v){
    if (ql>r || qr<l) return;
    if (ql<=l && qr>=r){
        tree[x].pb(v);
        return;
    }
    int mid = l + r >> 1;
    update(x << 1, l, mid, ql, qr, v);
    update(x << 1 | 1, mid + 1, r, ql, qr, v);
}
void dfs(int x,int l,int r){
    int now = 0;
    for (int id : tree[x]) now += join(E[id].x, E[id].y);
    if (l==r || find(1)==find(n)){
        if (find(1)==find(n)){
            ans += b[r] - b[l - 1];
        }
    }else{
        int mid = l + r >> 1;
        dfs(x << 1, l, mid);
        dfs(x << 1 | 1, mid + 1, r);
    }
    undo(now);
}
int main(){
    int m;
    scanf("%d%d",&n,&m);
    init();
    FOR (i,m){
        int x, y, l, r;
        scanf("%d%d%d%d", &x, &y, &l, &r);
        E.pb(EDGE(x, y, l, ++r));
        b.pb(l), b.pb(r);
    }
    sort(ALL(b));
    b.erase(unique(ALL(b)), b.end());
    int len = b.size();
    FOR(i, SZ(E)) update(1, 1, len, has(E[i].l), has(E[i].r) - 1, i);
    dfs(1, 1, len);
    printf("%d\n", ans);
    return 0;
}
```

#### 线段树维护下溢

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 5;
int n, m;
ll tree[N << 2];
ll mn[N << 2];
ll lazy[N << 2];
ll num_of_not0[N << 2];

void up(int x)
{
    tree[x] = tree[x << 1] + tree[x << 1 | 1];
    mn[x] = min(mn[x << 1], mn[x << 1 | 1]);
    num_of_not0[x] = num_of_not0[x << 1] + num_of_not0[x << 1 | 1];
}
void pushdown(int x,int l,int r){
    if (lazy[x]==0)
        return;
    int mid = l + r >> 1;
    mn[x << 1] -= lazy[x];
    mn[x << 1 | 1] -= lazy[x];

    tree[x << 1] -= lazy[x] * num_of_not0[x << 1];
    tree[x << 1 | 1] -= lazy[x] * num_of_not0[x << 1 | 1];

    lazy[x << 1] += lazy[x];
    lazy[x << 1 | 1] += lazy[x];
    lazy[x] = 0;
}
void update(int x,int l,int r,int ql,int qr,ll v){
    if (l>qr || r<ql || ql>qr)
        return;
    if (num_of_not0[x]==0) return;
    if (l==r){
        tree[x] -= v;
        mn[x] = tree[x] = max(0ll, tree[x]);
        if (tree[x]==0){
            num_of_not0[x] = 0;
            mn[x] = 0x7fffffff;
        }
        return;
    }
    if (ql<=l && qr>=r && mn[x]>v){
        mn[x] -= v;
        tree[x] -= v * num_of_not0[x];
        lazy[x] += v;
        return;
    }
    int mid = l + r >> 1;
    pushdown(x, l, r);
    update(x << 1, l, mid, ql, qr, v);
    update(x << 1 | 1, mid + 1, r, ql, qr, v);
    up(x);
}
ll query(int x,int l,int r,int ql,int qr){
    if (l>qr ||r<ql || qr<ql) return 0;
    if (ql <= l && qr >= r){
        return tree[x];
    }
    int mid = l + r >> 1;
    pushdown(x, l, r);
    return query(x << 1, l, mid, ql, qr) + query(x << 1 | 1, mid + 1, r, ql, qr);
}
void build(int x, int l, int r)
{
    if (l==r){
        scanf("%lld", &tree[x]);
        mn[x] = tree[x];
        num_of_not0[x] = (tree[x] != 0);
        return;
    }
    int mid = l + r >> 1;
    build(x << 1, l, mid);
    build(x << 1 | 1, mid + 1, r);
    up(x);
}
int main()
{
    // clock_t st = clock();
    scanf("%d%d", &n, &m);
    build(1, 1, n);
    while (m--){
        int op, l, r;
        ll s;
        scanf("%d%d%d", &op, &l, &r);
        if (op==1){
            if (l<=r)
                printf("%lld\n", query(1, 1, n, l, r));
            else
                printf("%lld\n", query(1, 1, n, l, n) + query(1, 1, n, 1, r));
        }
        else
        {
            scanf("%lld", &s);
            if (l<=r)
                update(1, 1, n, l, r, s);
            else
                update(1, 1, n, 1, r, s), update(1, 1, n, l, n, s);
        }
    }
    // clock_t et = clock();
    // cout << (double)(et - st) / CLOCKS_PER_SEC;
    return 0;
}
```

#### 扫描线统计单个贡献

```c++
// #pragma GCC optimize(2)
#include<bits/stdc++.h>
using namespace std;
#define mp make_pair
#define fi first
#define sc second
#define pb push_back
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
const int N = 1e5 + 5;
const int MOD = 998244353;
struct P{
    int x, y, v;
    P(){}
    P(int x, int y, int v) : x(x), y(y), v(v){}
};
int tree[N << 3], lazy[N << 3];
vector<P> a;
vector<int> b;
void Init_Hash(){
    sort(b.begin(), b.end());
    b.erase(unique(b.begin(), b.end()), b.end());
}
int has(int x){
    return lower_bound(b.begin(), b.end(), x) - b.begin() + 1;
}
void build(int x,int l,int r){
    tree[x] = lazy[x] = 0;
    if (l==r){
        return;
    }
    int mid = l + r >> 1;
    build(2 * x, l, mid);
    build(2 * x + 1, mid + 1, r);
}
void pushdown(int x){
    if (!lazy[x])
        return;
    tree[2*x] += lazy[x];
    tree[2 * x + 1] += lazy[x];
    lazy[2 * x] += lazy[x];
    lazy[2 * x + 1] += lazy[x];
    lazy[x] = 0;
}
void up(int x){
    tree[x] = max(tree[2 * x], tree[2 * x + 1]);
}
void update(int x,int l,int r,int ql,int qr,int v){
    if (ql>r || qr<l)
        return;
    if (ql<=l && qr>=r){
        tree[x] += v;
        lazy[x] += v;
        return;
    }
    pushdown(x);
    int mid = l + r >> 1;
    update(2 * x, l, mid, ql, qr, v);
    update(2 * x + 1, mid + 1, r, ql, qr, v);
    up(x);
}
int main(){
    int n, k;
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n;i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        b.pb(y);
        b.pb(y + k);
        a.pb(P(x, y, 1));
        a.pb(P(x + k + 1, y, -1));
    }
    Init_Hash();
    int len = b.size();
    build(1, 1, len);
    sort(a.begin(), a.end(), [](P a, P b) {
        return a.x < b.x;
    });
    int ans = 0;
    for (int i = 0; i < a.size();i++)
    {
        int x = a[i].x, y = a[i].y, v = a[i].v;
        int l = has(y), r = has(y + k);
        update(1, 1, len, l, r, v);
        ans = max(ans, tree[1]);
    }
    printf("%d\n", ans);
    return 0;
}
```

#### 扫描线统计多个贡献

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e5 + 10;

struct Q{
    int p, x, l, r, f;
    Q(){}
    Q(int p,int x,int l,int r,int f):p(p),x(x),l(l),r(r),f(f){}
    bool operator < (const Q &b) const{
        return x < b.x;
    }
} q[N << 1];

int n, m;
int pos[N];
ll ans[N];
vector<int> X[N];
ll tree[N << 2];
void update(int x,int l,int r,int p){
    if (l==r){
        tree[x]++;
        return;
    }
    int mid = l + r >> 1;
    if (p<=mid)
        update(x << 1, l, mid, p);
    else
        update(x << 1 | 1, mid + 1, r, p);
    tree[x] = tree[x << 1] + tree[x << 1 | 1];
}
ll query(int x,int l,int r,int ql,int qr){
    if (l>qr || r<ql || ql>qr)
        return 0;
    if (ql<=l && qr>=r){
        return tree[x];
    }
    int mid = l + r >> 1;
    return query(x << 1, l, mid, ql, qr) + query(x << 1 | 1, mid + 1, r, ql, qr);
}

int main(){
    // clock_t st = clock();
    scanf("%d%d", &n, &m);
    REP(i,1,n){
        int x;
        scanf("%d", &x);
        pos[x] = i;
    }
    for (int i = 1; i <= n;++i)
    {
        int l = pos[i];
        for (int j = 2 * i; j <= n; j += i)
        {
            int r = pos[j];
            X[l].pb(r);
        }
    }
    int cnt = 0;
    REP(i, 1, m)
    {
        int l, r;
        scanf("%d%d", &l, &r);
        q[++cnt] = Q(i, l - 1, l, r, -1);
        q[++cnt] = Q(i, r, l, r, 1);
    }
    sort(q + 1, q + 1 + cnt);
    int ind = 0;
    for (int i = 0; i <= n;++i)
    {
        for (auto y: X[i]){
            update(1, 1, n, y);
        }
        while (ind<=cnt && q[ind].x==i){
            ans[q[ind].p] += query(1, 1, n, q[ind].l, q[ind].r) * q[ind].f;
            ++ind;
        }
    }
    for (int i = 1; i <= m; ++i)
    {
        printf("%lld\n", ans[i]);
    }
    // clock_t et = clock();
    // cout << (double)(et - st) / CLOCKS_PER_SEC << endl;
    return 0;
}
```

### 差分

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 2e5 + 5;
const int R = 2e5;

set<int> S;

vector<int> add[N], del[N];

int C[N], E[N], L[N], n, m;

int c1[N], c2[N];
int lb(int x) { return x & -x; }
void update(int x,int v){
	for (int i = x; i <= R; i += lb(i))
	{
		c1[i] += v;
		c2[i] += x * v;
	}
}
pair<int,int> query(int x){
	pair<int, int> res = mp(0, 0);
	for (int i = x; i > 0; i -= lb(i))
	{
		res.fi += c1[i];
		res.sc += c2[i];
	}
	return res;
}

void s_ins(int x){
	auto it = S.insert(x).fi, l = it, r = it;
	if (S.size()==1) return;
	if (l!=S.begin() && r!=--S.end()){
		update(*(++r) - x, 1);
		update(x - *(--l), 1);
		update(*r - *l, -1);
	}else if (l!=S.begin()){
		update(x - *(--l), 1);
	}else {
		update(*(++r) - x, 1);
	}
}
void s_del(int x){
	auto it = S.find(x), l = it, r = it;
	if (S.size()==1) return;
	if (l!=S.begin() && r!=--S.end()){
		update(*(++r) - x, -1);
		update(x - *(--l), -1);
		update(*r - *l, 1);
	}else if (l!=S.begin()){
		update(x - *(--l), -1);
	}else{
		update(*(++r) - x, -1);
	}
	S.erase(it);
}

void init(){
	S.clear();
	memset(c1, 0, sizeof(c1));
	memset(c2, 0, sizeof(c2));
	REP(i,1,n){
		add[i].clear();
		del[i].clear();
	}
}

int main(){
	int _, cas = 0;
	for (scanf("%d", &_); _;--_)
	{
		scanf("%d", &n);
		init();
		REP(i,1,n){
			scanf("%d%d%d", E + i, L + i, C + i);
		}
		scanf("%d", &m);
		REP(i,1,m){
			int t, l, r;
			scanf("%d%d%d", &t, &l, &r);
			add[l].pb(t);
			del[r].pb(t);
		}
		ll ans = 0;
		S.insert(0);
		REP(i,1,n){
			for (auto x : add[i]){
				s_ins(x);
			}

			if (S.size()<=1) continue;
			
			if (L[i]==0){
				ans += E[i];
			}else{
				int t = C[i] / L[i];
				pair<int, int> res = t ? query(t) : mp(0, 0);
				ans += 1ll * res.sc * L[i] + 1ll * (S.size() - res.fi - 1) * C[i];

				auto it=S.lower_bound(1);
				if (1ll * *it * L[i] < C[i])
				{
					ans += min(C[i] - *it * L[i], E[i]);
				}
			}

			for (auto x : del[i]){
				s_del(x);
			}
		}
		printf("Case #%d: %lld\n", ++cas, ans);
	}
	return 0;
}
```



## 字符串

### hash

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
typedef long long ll;

int n, k;
struct NODE{
    int x, y;
    ll v;
    NODE(){}
    NODE(int x, int y, ll v) : x(x), y(y), v(v){}
    bool operator<(const NODE &b) const{
        return v < b.v;
    }
};
vector<NODE> vc;

ll dfs(int x,int y,int k){
    if (!k){
        return 0;
    }
    int mid = (1 << (k - 1));
    if (x<=mid){
        if (y<=mid){
            return dfs(y, x, k - 1);
        }else{
            y -= mid;
            return dfs(mid - y + 1, mid - x + 1, k - 1) + 3ll * (1 << (2 * k - 2));
        }
    }
    else
    {
        x -= mid;
        if (y <= mid){
            return dfs(x, y, k - 1) + (1 << (2 * k - 2));
        }
        else
        {
            y -= mid;
            return dfs(x, y, k - 1) + 2ll * (1 << (2 * k - 2));
        }
    }
}

int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n;++i)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        vc.push_back(NODE(x, y, dfs(x, y, k)));
    }
    sort(vc.begin(), vc.end());
    for (auto it: vc){
        printf("%d %d\n", it.x, it.y);
    }
    return 0;
}
```

### ACAM

#### Trie图找环

```c++
// #pragma GCC optimize(2)
#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define sc second
#define fi first
// using namespace std;
// typedef long long ll;
// typedef unsigned int uint;
// typedef pair<int, int> pi;
// const int N = 2e5 + 5;
// const int B = 31;
namespace AC{
    typedef long long ll;
    using namespace std;
    const int N = 1e6 + 5;
    int nxt[N][30], fail[N];
    bool en[N];
    int tot, root;
    int newNODE(){
        tot++;
        for (int i = 0; i < 2;i++)
            nxt[tot][i] = -1;
        en[tot] = 0;
        return tot;
    }
    void init(){
        tot = 0;
        root = newNODE();
    }
    void update(char *s){
        int len = strlen(s);
        int now = root;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - '0';
            if (nxt[now][p]==-1)
                nxt[now][p] = newNODE();
            now = nxt[now][p];
        }
        en[now] = 1;
    }
    void build(){
        queue<int> q;
        fail[root] = root;
        for (int i = 0; i < 2;i++)
        {
            if (nxt[root][i]==-1)
                nxt[root][i] = root;
            else{
                int x = nxt[root][i];
                fail[x] = root;
                q.push(x);
            }
        }
        while (!q.empty()){
            int now = q.front(); 
            q.pop();
            for (int i = 0; i < 2;i++)
            {
                if (nxt[now][i]==-1)
                    nxt[now][i] = nxt[fail[now]][i];
                else{
                    int x = nxt[now][i];
                    en[x] |= en[nxt[fail[now]][i]];
                    fail[x] = nxt[fail[now]][i];
                    q.push(x);
                }
            }
        }
    }
}
using namespace AC;
bool vis[N], ins[N];
bool dfs(int x){
    ins[x] = 1;
    for (int i = 0; i < 2;i++)
    {
        int p = nxt[x][i];
        if (ins[p])
            return 1;
        if (en[p] || vis[p])
            continue;
        if (vis[p])
            continue;
        vis[p] = 1;
        if (dfs(p))
            return 1;
    }
    ins[x] = 0;
    return 0;
}
char s[N];
int main(){
    int n;
    scanf("%d", &n);
    init();
    for (int i = 0; i < n;i++)
    {
        scanf(" %s", s);
        update(s);
    }
    build();
    printf("%s", dfs(root) ? "TAK" : "NIE");
    return 0;
}
```

#### 判断存在

```c++
 
#include <bits/stdc++.h>
namespace AC{
    typedef long long ll;
    using namespace std;
    const int N = 1e6 + 5;
    int nxt[N][30], fail[N], cnt[N];
    int tot, root;
    int newNODE(){
        tot++;
        for (int i = 0; i < 26;i++)
            nxt[tot][i] = -1;
        cnt[tot] = 0;
        return tot;
    }
    void init(){
        tot = 0;
        root = newNODE();
    }
    void update(char *s){
        int len = strlen(s);
        int now = root;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            if (nxt[now][p]==-1)
                nxt[now][p] = newNODE();
            now = nxt[now][p];
        }
        cnt[now]++;
    }
    void build(){
        queue<int> q;
        fail[root] = root;
        for (int i = 0; i < 26;i++)
        {
            if (nxt[root][i]==-1)
                nxt[root][i] = root;
            else{
                int x = nxt[root][i];
                fail[x] = root;
                q.push(x);
            }
        }
        while (!q.empty()){
            int now = q.front();
            q.pop();
            for (int i = 0; i < 26;i++)
            {
                if (nxt[now][i]==-1)
                    nxt[now][i] = nxt[fail[now]][i];
                else{
                    int x = nxt[now][i];
                    fail[x] = nxt[fail[now]][i];
                    q.push(x);
                }
            }
        }
    }
    int query(char *s){
        int len = strlen(s);
        int now = root;
        int res = 0;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            now = nxt[now][p];
            int tmp = now;
            while (tmp!=root){
                res += cnt[tmp];
                cnt[tmp] = 0;           //已经被统计过，所以置0
                tmp = fail[tmp];
            }
        }
        return res;
    }
}
using namespace AC;

char s[N];
int n;
void work(){
    init();
    scanf("%d", &n);
    for (int i = 1; i <= n;i++)
    {
        scanf(" %s", s);
        update(s);
    }
    build();
    scanf(" %s", s);
    printf("%d\n", query(s));
}
int main(){
    int t;
    scanf("%d", &t);
    while (t--){
        work();
    }
    return 0;
}
```

#### +矩阵快速幂

```c++
#include<cstdio>
#include<map>
#include<iostream>
#include<cstring>
#include<queue>
#define mp make_pair
#define pb push_back
#define fi first
#define sc second
typedef long long ll;
using namespace std;
const int N = 1e3 + 5;
const int B = 4;
const int mod = 100000;

map<char, int> fn;

struct ACAM{
    int nxt[N][B], fail[N], cnt[N];
    bool end[N];
    int tot, root;
    ll mat[N][N];
    int newNODE(){
        tot++;
        for (int i = 0; i < B;i++)
            nxt[tot][i] = -1;
        cnt[tot] = 0;
        end[tot] = 0;
        return tot;
    }
    void init(){
        tot = 0;
        root = newNODE();
        memset(mat, 0, sizeof(mat));
    }
    void update(char *s){
        int len = strlen(s);
        int now = root;
        for (int i = 0; i < len;i++)
        {
            int p = fn[s[i]];
            if (nxt[now][p]==-1)
                nxt[now][p] = newNODE();
            now = nxt[now][p];
        }
        cnt[now]++;
        end[now] = true;
    }
    void build(){
        queue<int> q;
        fail[root] = root;
        for (int i = 0; i < B;i++)
        {
            if (nxt[root][i]==-1)
                nxt[root][i] = root;
            else{
                int x = nxt[root][i];
                fail[x] = root;
                q.push(x);
            }
        }
        while (!q.empty()){
            int now = q.front(); 
            q.pop();
            if (end[fail[now]])
                end[now] = true;
            for (int i = 0; i < B; i++)
            {
                if (nxt[now][i]==-1)
                    nxt[now][i] = nxt[fail[now]][i];
                else{
                    int x = nxt[now][i];
                    fail[x] = nxt[fail[now]][i];
                    q.push(x);
                }
            }
        }
    }
    
    void setmat()
    {
        for ( int i=1;i<=tot;i++ )
        {
            for ( int j=0;j<B;j++ )
            {
                if ( !end[i] && !end[nxt[i][j]] ) mat[i][nxt[i][j]]++;
            }
        }
    }
    ll res[N][N],tmp[N][N];
    void mul(ll a[][N],ll b[][N])
    {
        for ( int i=1;i<=tot;i++ )
        {
            for ( int j=1;j<=tot;j++ )
            {
                tmp[i][j]=0;
                for ( int k=1;k<=tot;k++ ) tmp[i][j]=(tmp[i][j]+a[i][k]*b[k][j])%mod;
            }
        }
        for ( int i=1;i<=tot;i++ )
        {
            for ( int j=1;j<=tot;j++ ) a[i][j]=tmp[i][j]%mod;
        }
    }
    void pow(ll k)
    {
        memset(res,0,sizeof(res));
        for ( int i=1;i<=tot;i++ ) res[i][i]=1;
        while ( k )
        {
            if ( k&1 ) mul(res,mat);
            mul(mat,mat);
            k/=2;
        }
    }
} ac;
char s[N];
int n, m;
int main()
{
    fn['A'] = 0;
    fn['C'] = 1;
    fn['G'] = 2;
    fn['T'] = 3;
    while (~scanf("%d%d", &n, &m))
    {
        ac.init();
        for (int i = 1; i <= n; ++i)
        {
            scanf("%s", s);
            ac.update(s);
        }
        ac.build();
        ac.setmat();
        ac.pow(m);
        ll ans = 0;
        for ( int i=1;i<=ac.tot;i++ ) 
        {                                                                                                                                                                                  
            ans=(ans+ac.res[1][i])%mod;
        }
        printf("%lld\n", ans);
    }
    return 0;
}
```



### SAM

#### +DP1

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((ll)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (ll I = 0; I < (N); ++I)
#define FORD(I, N) for (ll I = N; ~I; --I)
#define REP(I, A, B) for (ll I = A; I <= (B); ++I)
#define REPD(I, B, A) for (ll I = B; I >= A; --I)
#define FORS(I, S) for (ll I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e6 + 5;

int n;
char s[N];

struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }

    int id[N];
    int v[N];
    int sum[N];
    
    ll getSum(){
        int mx = 0;
        for (int i = 1; i <= tot; i++)
            mx = max(mx, len[i]), ++v[len[i]];
        for (int i = 1; i <= mx; i++)
            v[i] += v[i - 1];
        for (int i = 1; i <= tot; i++){
            id[v[len[i]]--] = i;
        }
        ll ans = 0;
        for (int i = tot; i; i--)
        {
            int t = id[i];
            sz[fa[t]] += sz[t];
            ans += 1ll*(len[t] - len[fa[t]]) * sz[t] * (n - sz[t]);
        }
        return ans;
    }
} sam;


int main(){
    sam.init();
    scanf(" %s", s);
    n = strlen(s);
    FORD(i,n-1){
        sam.extend(s[i] - 'a');
    }
    printf("%lld\n", sam.getSum());
    return 0;
}
```

#### +DP2

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e6 + 5;

ll q, p;
char s[N];
ll f[N];
int n;

struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }
    void work(){
        init();
        f[0] = 0;
        f[1] = p;
        extend(s[1] - 'a');
        int j = 1;
        int now = 0;
        for (int i = 2; i <= n;i++)
        {
            int x = s[i] - 'a';
            if (~nxt[now][x]){
                now = nxt[now][x];
            }else{
                while (~now && nxt[now][x]==-1){
                    now = fa[now];
                    while (j < i - len[now] - 1){
                        extend(s[++j] - 'a');   //***get a instant update to SAM***
                    }
                }
                if (now==-1){
                    now = 0;
                    extend(s[++j] - 'a');       //if not, j will be i-1, which is not what we want
                }else{
                    now = nxt[now][x];          //transfer to next state
                }
            }
            f[i] = f[i - 1] + p;
            if (j<i) f[i] = min(f[i], f[j] + q);//if j>=i, then no match
        }
    }
    
} sam;


int main(){
    while (~scanf("%s", s + 1))
    {
        n = strlen(s + 1);
        scanf("%lld%lld", &p, &q);
        sam.work();
        printf("%lld\n", f[n]);
    }
    return 0;
}
```

#### SAM求sz

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((ll)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (ll I = 0; I < (N); ++I)
#define FORD(I, N) for (ll I = N; ~I; --I)
#define REP(I, A, B) for (ll I = A; I <= (B); ++I)
#define REPD(I, B, A) for (ll I = B; I >= A; --I)
#define FORS(I, S) for (ll I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<ll, ll> pi;
typedef pair<ll, ll> pl;
const ll N = 1e6 + 5;
int n;
char s[N];
char vis[N];

struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x,int pos)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = (vis[pos] == '0');
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }

    int id[N];
    int v[N];
    
    ll getSum(){
        int mx = 0;
        for (int i = 1; i <= tot; i++)
            mx = max(mx, len[i]), ++v[len[i]];
        for (int i = 1; i <= mx; i++)
            v[i] += v[i - 1];
        for (int i = 1; i <= tot; i++){
            id[v[len[i]]--] = i;
        }
        for (int i = tot; i; i--)
        {
            int t = id[i];
            sz[fa[t]] += sz[t];
        }
        // sz[0] = 0;//空串不能算在内
        ll ans = 0;
        for (int i = 1; i <= tot;i++)
        {
            ans = max(ans, 1ll * sz[i] * len[i]);
        }
        return ans;
    }

} sam;

int main(){
    sam.init();
    scanf("%d", &n);
    scanf(" %s %s", s, vis);
    for (int i = 0; i < n;i++)
    {
        sam.extend(s[i] - 'a', i);
    }
    printf("%lld\n", sam.getSum());
    return 0;
}
```

#### SAM求LCS

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e6 + 5;

char s[N];
int cur[N], mn[N], id[N];

struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int root, tot, last;
    int newnode(int l)
    {
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot++] = l;
        return tot - 1;
    }
    void init()
    {
        tot = 0;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }
    int getAns(){
        while (~scanf("%s",s)){
            memset(cur, 0, sizeof(cur));
            int tmp = 0;
            int now = 0;
            FORS(i,s){
                int x = s[i] - 'a';
                if (~nxt[now][x]){
                    now = nxt[now][x];
                    tmp++;
                }else{
                    while (~now && nxt[now][x]==-1) now = fa[now];
                    if (now==-1){
                        tmp = now = 0;
                    }else{
                        tmp = len[now] + 1;
                        now = nxt[now][x];
                    }
                }
                cur[now] = max(cur[now], tmp);
            }
            REPD(i, tot - 1, 0){
                int who = id[i];
                mn[who] = min(cur[who], mn[who]);
                if (cur[who] && ~fa[who]) cur[fa[who]] = len[fa[who]];
            }
        }
        int ans = 0;
        FOR(i,tot){
            ans = max(ans, mn[i]);
        }
        return ans;
    }
} sam;

int v[N];

int main(){
    sam.init();
    scanf("%s", s);
    FORS(i, s) sam.extend(s[i] - 'a');
    /**
     * 我们需要用孩子的信息更新father的信息
     * 但是id大的点不一定是id小的点的father
     * 需要按照len从小到大重新编号
     * 使得能在线性时间内更新完所有son对father的信息
    */
    REP(i,1,sam.tot-1) v[sam.len[i]]++;
    int n = strlen(s); 
    REP(i, 1, n) { v[i] += v[i - 1]; }
    REP(i, 1, sam.tot){
        id[v[sam.len[i]]--] = i;
    }
    /**
     * END
    */
    FOR(i, sam.tot) mn[i] = sam.len[i];

    printf("%d\n", sam.getAns());
    return 0;
}
```

#### TJOI 2015

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e6 + 5;
const int MOD = 1e9 + 7;

bool T;

struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }

    int id[N];
    int v[N];
    int sum[N];
    
    void getSum(){
        int mx = 0;
        for (int i = 1; i <= tot; i++)
            mx = max(mx, len[i]), ++v[len[i]];
        for (int i = 1; i <= mx; i++)
            v[i] += v[i - 1];
        for (int i = 1; i <= tot; i++){
            id[v[len[i]]--] = i;
        }
        for (int i = tot; i; i--)
        {
            int t = id[i];
            if (T)
                sz[fa[t]] += sz[t];
            else
                sz[t] = 1;
            // cout << t << " " << sz[t] << endl;
        }
        // sz[0] = 0;//空串不能算在内
        for (int i = tot; ~i;i--)
        {
            int t = id[i];
            sum[t] = sz[t];
            for (int j = 0; j < 26;j++) if (~nxt[t][j])
            {
                 sum[t] += sum[nxt[t][j]];
            }
        }
    }
    void solve(){
        int x, now = 0; scanf("%d", &x);
        if (sum[0]<x) { puts("-1"); return; }
        while (x){
            for (int i = 0; i < 26;i++)
            {
                if (~nxt[now][i]){
                    if (sum[nxt[now][i]]>=x){
                        putchar('a' + i);
                        now = nxt[now][i];
                        x -= sz[now];
                        break;
                    }else
                        x -= sum[nxt[now][i]];
                }
            }
        }
        puts("");
    }
} sam;

char s[N];
int main(){
    sam.init();
    scanf(" %s", s);
    FORS(i,s){
        sam.extend(s[i] - 'a');
    }
    scanf("%d", &T);
    sam.getSum();
    sam.solve();
    return 0;
}
```



## 图论

### 图分块

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((int)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (int I = 0; I < (N); ++I)
#define FORD(I, N) for (int I = N; ~I; --I)
#define REP(I, A, B) for (int I = A; I <= (B); ++I)
#define REPD(I, B, A) for (int I = B; I >= A; --I)
#define FORS(I, S) for (int I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 2e5 + 5;
const int BLC = 1e3;

int n, m, S, q;

pi edge[N];
int cnt, he[N], ne[N << 1], v[N << 1], w[N << 1];
int deg[N];
ll V[N];

int Q[N], id[N], tot;
int bel[N], st[BLC], ed[BLC];

int lazy[BLC], cur[N];

ll F[BLC][BLC], G[BLC][BLC];

void add(int x,int y,int i){
	ne[++cnt] = he[x];
	he[x] = cnt;
	v[cnt] = y;
	w[cnt] = i;
}

void Force(int l,int r){
	int k = bel[l];
	REP(i,l,r){
		cur[i] ^= 1;
		int x = edge[i].fi, y = edge[i].sc;
		if (deg[x]>S){
			F[k][id[x]] ^= V[y];
			G[k][id[x]] ^= V[y];
		}
		if (deg[y]>S){
			F[k][id[y]] ^= V[x];
			G[k][id[y]] ^= V[x];
		}
	}
}

void update(int l,int r){
	int u = bel[l], v = bel[r];
	if (u==v) return Force(l, r);
	REP(i,u+1,v-1)
		lazy[i] ^= 1;
	Force(l, ed[u]);
	Force(st[v], r);
}

ll query(int x){
	ll ans = 0;
	if (deg[x]>S){
		REP(i,1,tot){
			ans ^= (lazy[i] ? G[i][id[x]] : F[i][id[x]]);
		}
	}else{
		for (int i = he[x]; i; i = ne[i]){
			int k = w[i];
			int u = v[i];
			ans ^= (lazy[bel[k]] ^ cur[k] ? V[u] : 0);
		}
	}
	return ans;
}

void init(){
	*Q = tot = cnt = 0;
	REP(i,1,n){
		he[i] = 0;
		deg[i] = id[i] = 0;
	}
}
ll Rand(){
	return ((ll)rand() << 30ll) + rand();
}
int main(){
	int _;
	for (scanf("%d", &_); _;--_)
	{
		scanf("%d%d", &n, &m);
		init();
		REP(i, 1, n) V[i] = Rand();
		REP(i,1,m){
			int x, y;
			scanf("%d%d", &x, &y);
			edge[i] = mp(x, y);
			add(x, y, i); add(y, x, i);
			++deg[x]; ++deg[y];
			cur[i] = 1;
		}
		for (S = 1; S * S <= m; S++); --S;
		for (int i = 1; i <= m; i += S){
			lazy[++tot] = 0;
			st[tot] = i;
			ed[tot] = min(m, i + S - 1);
			REP(j, st[tot], ed[tot]) bel[j] = tot;
		}
		REP(i,1,n){
			if (deg[i]>S){
				Q[++*Q] = i;
				id[i] = *Q;
			}
		}
		REP(i,1,*Q){
			REP(j,1,tot)
				F[j][i] = G[j][i] = 0;
			for (int t = he[Q[i]]; t; t = ne[t]){
				F[bel[w[t]]][i] ^= V[v[t]];
			}
		}
		scanf("%d", &q);
		FOR(i,q){
			int op, l, r;
			scanf("%d%d%d", &op, &l, &r);
			if (op==1) update(l, r);
			else putchar((query(l) == query(r)) + '0');
		}
		puts("");
	}
	return 0;
}
```

### spfa多层图

```c++
// #pragma GCC optimize(2)
#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define sc second
#define fi first
using namespace std;
typedef long long ll;
typedef pair<int, int> pi;
const int N = 1e3 + 5;
priority_queue<pi, vector<pi>, greater<pi> > q;
int cnt, he[N], ne[N * 2], w[N * 2], v[N * 2];
int d[N][N];
bool vis[N][N];
int n, m, k;
void add(int x,int y,int z){
    if (x==y)
        return;
    for (int i = he[x]; i;i=ne[i])
    {
        int p = v[i];
        if (p==y){
            w[i] = min(w[i], z);
            return;
        }
    }
    cnt++;
    ne[cnt] = he[x];
    v[cnt] = y;
    w[cnt] = z;
    he[x] = cnt;
}
void spfa(int s){
    queue<pi> q;
    memset(d, -1, sizeof(d));
    memset(vis, 0, sizeof(vis));
 
    q.push(mp(s, 0));
    d[s][0] = 0;
    vis[s][0] = 1;
    while (!q.empty()){
        pi j = q.front();
        q.pop();
        int x = j.fi;
        int t = j.sc;
        for (int i = he[x]; i; i = ne[i])
        {
            int p = v[i];
            for (int offset = 0; offset < 2;offset++)
            {
                int nt = t + offset;
                if (nt>k) continue;
                if (d[p][nt] == -1 || d[p][nt] > d[x][t] + (offset == 1 ? 0 : w[i])){
                    d[p][nt] = d[x][t] + (offset == 1 ? 0 : w[i]);
                    if (!vis[p][nt]){
                        q.push(mp(p, nt));
                        vis[p][nt] = 1;
                    }
                }
            }
        }
 
        vis[x][j.sc] = 0;
    }
}
int main(){
    int s, t;
    scanf("%d%d%d%d%d", &n, &m, &s, &t, &k);
    for (int i = 1; i <= m;i++)
    {
        int x, y, z;
        scanf("%d%d%d", &x, &y, &z);
        add(x, y, z);
        add(y, x, z);
    }
    spfa(s);
    int ans = -1;
    for (int i = 0; i <= k;i++)
    {
        if (~d[t][i])
            ans = ans == -1 ? d[t][i] : min(ans, d[t][i]);
    }
    printf("%d\n", (s == t ? 0 : ans));
    return 0;
}
```

### 最大团问题

```c++
/**********************************************************************************************
  最大团问题相当于补图的最大独立集问题
  最大独立集=点数-最小点覆盖
  最大独立集的点集可以从残余图上寻找——我们可以选择U中汇点可达的点（记为T）和V中汇点不可达的点（记为P）
  T任何一点之间一定与P中点共边，否则P中的点不可能属于P（因为要么可达要么还有增广路）
***********************************************************************************************/
// #pragma GCC optimize(2)
#include<bits/stdc++.h>
namespace DINIC{
    using namespace std;
    typedef long long ll;
    const int N = 1e5 + 5;

    int s, e, n, m;
    queue<int> q;
    int dep[N], cur[N];

    int cnt, he[N], v[N], w[N], ne[N];
    void add(int x,int y,int z){
        cnt++;
        ne[cnt] = he[x];
        v[cnt] = y;
        w[cnt] = z;
        he[x] = cnt;
    }
    void init(){
        cnt = -1;
        memset(he, -1, sizeof(he));
        memset(ne, -1, sizeof(ne));
    }

    bool bfs(){
        memset(dep,0,sizeof(dep));
        while (!q.empty()) q.pop();
        dep[s]=1;
        q.push(s);
        while (!q.empty()){
            int j=q.front();
            q.pop();
            for (int i=he[j];i>-1;i=ne[i]){
                int p=v[i];
                if (w[i] && !dep[p]){
                    dep[p]=dep[j]+1;
                    q.push(p);
                }
            }
        }
        if (dep[e]==0) return 0;
        return 1;
    }
    int dfs(int u,int dist){
        if (u==e) return dist;
        else {
            for (int& i=cur[u];i>-1;i=ne[i]){
                int p=v[i];
                if (!w[i] || dep[p]<=dep[u]) continue;
                int di=dfs(p,min(dist,w[i]));
                if (di){
                    w[i]-=di;
                    w[i^1]+=di;
                    return di;
                }
            }
        }
        return 0;
    }
    int Dinic(){
        int ans=0;
        while (bfs()){
            for (int i = 1; i <= e; i++)
            {
                cur[i]=he[i];
            }
            while (int di=dfs(s,INT_MAX)){
                ans+=di;
            }
        }
        return ans;
    }
}
using namespace DINIC;
int a[N];
vector<int> ans;
bool is_source[N];
int main(){
    init();
    scanf("%d", &n);
    s = n + 1;
    e = n + 2;
    for (int i = 1; i <= n;i++)
    {
        scanf("%d", &a[i]);
        if (__builtin_popcount(a[i])&1)
            is_source[i] = 1;
        if (is_source[i]){
            add(s, i, 1);
            add(i, s, 0);
        }
        else{
            add(i, e, 1);
            add(e, i, 0);
        }
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (__builtin_popcount(a[i]^a[j])==1){
                if (is_source[i]){
                    add(i, j, 1);
                    add(j, i, 0);
                }
            }
        }
    }
    printf("%d\n", n - Dinic());
    bfs();
    for (int i = 1; i <= n;i++)
    {
        if (dep[i]!=0){
            if (is_source[i])
                ans.push_back(a[i]);
        }else{
            if (!is_source[i])
                ans.push_back(a[i]);
        }
    }
    for (int i = 0; i < ans.size();i++)
    {
        printf("%d%c", ans[i], i == ans.size() - 1 ? '\n' : ' ');
    }
    return 0;
}
```

### floyd最小环

```c++
// #pragma GCC optimize(3)
#include<bits/stdc++.h>
using namespace std;
#define fi first
#define sc second
#define pb push_back
#define mp make_pair
#define LEN(X) strlen(X)
#define SZ(X) ((ll)(X).size())
#define ALL(X) (X).begin(), (X).end()
#define FOR(I, N) for (ll I = 0; I < (N); ++I)
#define FORD(I, N) for (ll I = N; ~I; --I)
#define REP(I, A, B) for (ll I = A; I <= (B); ++I)
#define REPD(I, B, A) for (ll I = B; I >= A; --I)
#define FORS(I, S) for (ll I = 0; S[I]; ++I)
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
const int N = 1e3 + 5;
const ll INF = 1e9 + 7;
ll f[N][N];
ll d[N][N];
ll a[1000010];
int main(){
    std::ios::sync_with_stdio(false);
    int n;
    cin >> n;
    int tot = 0;
    for (int i = 1; i <= n;i++)
    {
        ll x;
        cin >> x;
        if (x)
            a[++tot] = x;
    }
    if (tot>=500){
        cout << "3" << endl;
        return 0;
    }
    for (int i = 1; i <= tot;i++) for (int j = 1; j <= tot;j++){
        f[i][j] = d[i][j] = INF;
    }
    for (int i = 1; i <= tot;i++) for (int j = i + 1; j <= tot;j++) if (a[i]&a[j])
    {
        f[i][j] = f[j][i] = d[i][j] = d[j][i] = 1;
    }
    ll ans = INF;
    for (int k = 1; k <= tot;k++)
    {
        for (int i = 1; i < k;i++)
        {
            for (int j = i+1; j <= k;j++)
            {
                ll tmp = f[i][j] + d[j][k] + d[k][i];
                if (tmp<ans){
                    ans = tmp;
                }
            }
        }
        for (int i = 1; i <= tot;i++)
        {
            for (int j = 1; j <= tot;j++)
            {
                ll tmp = f[j][k] + f[k][i];
                if (tmp<f[i][j]){
                    f[i][j] = f[j][i] = tmp;
                }
            }
        }
    }
    printf("%lld\n", (ans == INF ? -1 : ans));
    return 0;
}
```

