#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N = 1e5 + 5;
const int K = 6;

int di;
struct NODE{
    ll d[K];
    bool operator < (const NODE & a) const {
        return d[di] < a.d[di];
    }
    bool operator > (const NODE & a) const {
        return a < (*this);
    }
} point[N]; //起点从0开始
priority_queue<pair<ll, NODE> > q;

namespace KDTREE{
NODE tree[N << 2];
int son[N << 2];
int k;									//需要赋值k！！！！
ll square(ll x)
{
    return x * x;
}
void build(int x,int l,int r,int dim=0){//[l,r]
    di = dim % k;
    if (l>r)
        return;
    son[x] = r - l;
    son[2 * x] = son[2 * x + 1] = -1;
    int mid = l + r >> 1;
    nth_element(point + l, point + mid, point + r + 1);
    tree[x] = point[mid];
    build(x * 2, l, mid - 1, dim + 1);
    build(x * 2 + 1, mid + 1, r, dim + 1);
}
void build(int l,int r){
    build(1, l, r, 0);
}
void queryK(int x,NODE o,int m,int dim=0){
    if (son[x]==-1)
        return;
    int di = dim % k;
    pair<ll, NODE> now(0,tree[x]);
    for (int i = 0; i < k;i++)
        now.first += square(tree[x].d[i] - o.d[i]);    //计算当前节点的距离
    int xx = 2 * x, yy = 2 * x + 1;
    bool flag = 0;
    if (o.d[di]>=tree[x].d[di])                        //先递归潜力大的分支
        swap(xx, yy);
    if (~son[xx])                                    //如果xx存在
        queryK(xx, o, m, dim + 1);
    if (q.size()<m)
        q.push(now), flag = 1;
    else{
        if (now.first<q.top().first)
            q.pop(), q.push(now);
        if (square(o.d[di]-tree[x].d[di]) < q.top().first)//当前维数来看有潜力比队列中的小
            flag = 1;
    }
    if (~son[yy] && flag)                            //如果yy存在并且有潜力
        queryK(yy, o, m, dim + 1);
}
void queryK(NODE o,int m){
    queryK(1, o, m, 0);
}
}
