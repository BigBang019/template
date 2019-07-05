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
