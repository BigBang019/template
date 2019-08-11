/******************************************************************
    离散化后线段树求中位数
    写起来有一定难度，能用脚写之前先存个板子
******************************************************************/
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
/**************************************************************************
    区间离散化线段树
**************************************************************************/
/*
 *  仍用<l,r>标记一个节点
 *  但是节点代表一个区间[l,r+1)
*/
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

/*
 * 离散化时应当离散化区间端点l与r+1
 */
    b.pb(l), b.pb(r+1);
    sort(ALL(b));
b.erase(unique(ALL(b)), b.end());
/*
 * 修改时修改[l,r+1)
 * 但是节点是由<l,r>标记的因此右端点-1
 */
update(1, 1, len, has(E[i].l), has(E[i].r) - 1, i);
