//poj 3468
#include<cstdio>
using namespace std;
typedef long long ll;
const int N = 1e5+5;
ll tree[N * 4];
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
    tree[2 * x] += lazy[x] * (mid - l + 1);
    tree[2 * x + 1] += lazy[x] * (r - mid);
    lazy[x] = 0;
}
void build(int x,ll l,ll r){
    if (l==r){
        tree[x] = a[l];
        return;
    }
    int mid = l + r >> 1;
    build(2*x,l,mid);
    build(2*x+1,mid+1,r);
    tree[x] = tree[2 * x] + tree[2 * x + 1];
}
void insert(int x,ll l,ll r,ll ql,ll qr,ll v){
    if (l>qr || r<ql)
        return;
    ll mid = l + r >> 1;
    if (l>=ql && r<=qr){
        tree[x] += v * (r - l + 1);
        lazy[x] += v;
        return;
    }
    pushdown(x, l, r);
    insert(2*x,l,mid,ql,qr,v);
    insert(2*x+1,mid+1,r,ql,qr,v);
    tree[x] = tree[2 * x] + tree[2 * x + 1];
}
ll query(int x,ll l,ll r,ll ql,ll qr){
    if (qr<l || ql>r)
        return 0;
    ll mid = l + r >> 1;
    if (l>=ql && qr>=r)
        return tree[x];
    pushdown(x, l, r);
    return query(2 * x, l, mid, ql, qr) + query(2 * x + 1, mid + 1, r, ql, qr);
}
int main(){
    // freopen("E:/vscode/main/in.txt","r",stdin);
    // freopen("E:/vscode/main/out.txt","w",stdout);
    scanf("%d%d",&n,&q);
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
        if (ch=='C'){
            scanf("%lld",&v);
            insert(1,1,n,a,b,v);
        }else{
            printf("%lld\n",query(1,1,n,a,b));
        }
    }
    // fclose(stdin);
    // fclose(stdout);
    return 0;
}
