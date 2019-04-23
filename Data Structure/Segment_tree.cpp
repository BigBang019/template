#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 2e5+5;
struct NODE{
    ll sum;
    int l, r;
    int da;
    NODE(){}
    NODE(int l,int r,int da,ll sum):l(l),r(r),da(da),sum(sum){}
} tree[N * 4];
ll lazy[N * 4];
ll a[N];
int n, q;
void pushdown(int x,ll l,ll r){
    if (l==r){
        lazy[x] = 0;
        return;
    }
    ll mid = l + r >> 1;
    lazy[2 * x] += lazy[x];
    lazy[2 * x + 1] += lazy[x];
    tree[2 * x].sum += lazy[x] * (mid - l + 1);
    tree[2 * x + 1].sum += lazy[x] * (r - mid);
    tree[2 * x].da += lazy[x];
    tree[2 * x + 1].da += lazy[x];
    lazy[x] = 0;
}
void build(int x,ll l,ll r){
    if (l==r){
        tree[x] = NODE(l,r,a[l],a[l]);
        return;
    }
    tree[x] = NODE(l,r,a[l],a[l]);
    int mid = l + r >> 1;
    build(2*x,l,mid);
    build(2*x+1,mid+1,r);
    tree[x].sum = tree[2 * x].sum + tree[2 * x + 1].sum;
    tree[x].da = max(tree[2 * x].da, tree[2 * x + 1].da);
}
void insert(int x,ll l,ll r,ll ql,ll qr,ll v){
    if (l>qr || r<ql)
        return;
    ll mid = l + r >> 1;
    if (l>=ql && r<=qr){
        tree[x].sum += v * (r - l + 1);
        tree[x].da += v;
        lazy[x] += v;
        return;
    }
    pushdown(x, l, r);
    insert(2*x,l,mid,ql,qr,v);
    insert(2*x+1,mid+1,r,ql,qr,v);
    tree[x].sum = tree[2 * x].sum + tree[2 * x + 1].sum;
    tree[x].da = max(tree[2 * x].da, tree[2 * x + 1].da);
}
ll query(int x,ll l,ll r,ll ql,ll qr){
    if (qr<l || ql>r)
        return 0;
    ll mid = l + r >> 1;
    if (l>=ql && qr>=r)
        return tree[x].sum;
    pushdown(x, l, r);
    return query(2 * x, l, mid, ql, qr) + query(2 * x + 1, mid + 1, r, ql, qr);
}
ll getMax(int x,ll l,ll r,ll ql, ll qr){
    if (qr<l || ql>r)
        return -1;
    ll mid = l + r >> 1;
    if (l>=ql && qr>=r)
        return tree[x].da;
    pushdown(x, l, r);
    return max(getMax(2 * x, l, mid, ql, qr), getMax(x * 2 + 1, mid + 1, r, ql, qr));
}
void change(int x,ll l, ll r, ll p, ll v){
    if (p<l || p>r)
        return;
    if (l==r){
        tree[x].sum = v;
        tree[x].da = v;
        return;
    }
    ll mid = l + r >> 1;
    change(2 * x, l, mid, p, v);
    change(2 * x + 1, mid + 1, r, p, v);
    tree[x].sum = tree[2 * x].sum + tree[2 * x + 1].sum;
    tree[x].da = max(tree[2 * x].da, tree[2 * x + 1].da);
}
int main(){
    // freopen("E:/vscode/main/in.txt","r",stdin);
    // freopen("E:/vscode/main/out.txt","w",stdout);
    while (scanf("%d%d",&n,&q)!=EOF){
        for (int i = 1; i <= n;i++)
        {
            scanf("%lld",&a[i]);
        }
        build(1,1,n);
        while (q--){
            char ch;
            int a, b;
            ll v;
            scanf(" %c%d%d",&ch,&a,&b);
            if (ch=='U'){
                change(1,1,n,a,b);
            }else{
                printf("%lld\n",getMax(1,1,n,a,b));
            }
        }
    }
    
    // fclose(stdin);
    // fclose(stdout);
    return 0;
}
