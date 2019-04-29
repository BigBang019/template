#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N = 1e5 + 5;
const int K = 6;
namespace KDTREE{
int dim;
int k;
struct NODE{
	ll d[K];
	// ll l[K],r[K];					//初始化时要让l=r=d
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
// void up(int x){
// 	int lc = tree[x].lc;
// 	int rc = tree[x].rc;
// 	for (int i = 0; i < k;i++)
// 	{
// 		if (lc){
// 			tree[x].l[i] = min(tree[x].d[i], min(tree[x].l[i], tree[lc].l[i]));
// 			tree[x].r[i] = max(tree[x].d[i], max(tree[x].r[i], tree[lc].r[i]));
// 		}
// 		if (rc){
// 			tree[x].l[i] = min(tree[x].d[i], min(tree[x].l[i], tree[rc].l[i]));
// 			tree[x].r[i] = max(tree[x].d[i], max(tree[x].r[i], tree[rc].r[i]));
// 		}		
// 	}
// }
void build(int l,int r,int dep){	//[l,r]
	tot++;
	int now = tot;
	dim = dep % k;
	int mid = l + r >> 1;
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
	// up(now);
}
void build(int l,int r){
	tot = 0;
	build(l,r,0);
}
// ll value(int x,NODE t){			//估价函数
// 	ll sum = 0;
// 	for (int i = 0; i < k;i++)	//曼哈顿距离
// 	{
// 		sum += max((ll)0,tree[x].l[i]-t.d[i]);
// 		sum += max((ll)0,t.d[i]-tree[x].r[i]);
// 	}
// 	return sum;
// }
void query(int x,NODE t,int m,int dep){//q.top()记录当前第size()近的点
	int dim = dep % k;
	pair<ll, NODE> now(distance(t,tree[x]),tree[x]);
	int xx = tree[x].lc, yy = tree[x].rc;
	bool flag = 0;
	// if (xx)						//评估两边的价值
	// 	v1 = value(xx,t);
	// if (yy)
	// 	v2 = value(yy,t);
	if (t.d[dim]>tree[x].d[dim])		//距离小的优先递归
		swap(xx, yy);
	if (~xx)					//如果这个人存在
		query(xx, t, m,dep+1);
	if (q.size()<m)				//如果最优box不够m个，把当前的加进去，次优box仍有更优可能
		q.push(now),flag=1;
	else {						//够了m个，当前节点和次优box仍有更优可能
		if (now.first<q.top().first)//判断当前节点是否能替换
			q.pop(), q.push(now);
		if (square(tree[x].d[dim]-t.d[dim])<q.top().first)
			flag = 1;			//剪枝，如果当前维的距离仍可以接受，那么次优box仍有更优可能
								//否则当前次优box不可能
	}
	if (~yy && flag)
		query(yy,t,m,dep+1);
}
vector<pair<ll,NODE>> query(NODE t,int m){
	query(1, t, m, 0);
	vector<pair<ll, NODE> > b;
	while (!q.empty())
		b.push_back(q.top()), q.pop();
	return b;
}
}
