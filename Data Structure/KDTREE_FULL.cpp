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
