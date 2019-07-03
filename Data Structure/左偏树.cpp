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
