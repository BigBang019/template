#include<bits/stdc++.h>
#define N 100005
using namespace std;
typedef long long ll;
ll tree[8*N];
ll tree2[8*N];
bool no[8*N];
ll same[8*N],lazy[8*N];
int n;
ll a[N];
void build(int x,int l,int r){
    if (l==r){
        tree[x]=a[l];
        tree2[x]=a[l]*a[l];
        return;
    }
    int mid=l+r>>1;
    build(x*2,l,mid);
    build(x*2+1,mid+1,r);
    tree[x]=tree[2*x]+tree[2*x+1];
    tree2[x]=tree2[2*x]+tree2[2*x+1];
    lazy[x]=0;
    no[x]=0;
    same[x]=0;
}

void add(int x,int l,int r,int ql,int qr,int key){
    int mid=l+r>>1;
    if (no[x]){
        same[x]=lazy[x]=0;
        no[2*x]=1;
        no[2*x+1]=1;
        no[x]=0;
    }
    if (same[x]){
        same[x*2]=same[x];
        same[2*x+1]=same[x];

        tree[2*x]=same[x]*(mid-l+1);
        tree[2*x+1]=same[x]*(r-mid);

        tree2[2*x]=same[x]*same[x]*(mid-l+1);
        tree2[2*x+1]=same[x]*same[x]*(r-mid);
        same[x]=0;
    }
    if (lazy[x]){
        lazy[x*2]=lazy[x];
        lazy[2*x+1]=lazy[x];

        tree2[2*x]+=2*lazy[x]*tree[2*x]+(mid-l+1)*lazy[x]*lazy[x];
        tree2[2*x+1]+=2*lazy[x]*tree[2*x+1]+(r-mid)*lazy[x]*lazy[x];

        tree[2*x]+=lazy[x]*(mid-l+1);
        tree[2*x+1]+=lazy[x]*(r-mid);

        lazy[x]=0;
    }
    if (ql > r || qr < l || l>r) return;

    if (ql<=l && qr>=r){
        tree2[x]+=2*key*tree[x]+(r-l+1)*key*key;

        tree[x]+=key*(r-l+1);

        lazy[x]+=key;
        return;
    }

    add(2*x,l,mid,ql,qr,key);
    add(2*x+1,mid+1,r,ql,qr,key);
    tree[x]=tree[2*x]+tree[2*x+1];
    tree2[x]=tree2[2*x]+tree2[2*x+1];
}
void make_same(int x,int l,int r,int ql,int qr,int key){
    int mid=l+r>>1;
    if (same[x]){
        same[x*2]=same[x];
        same[2*x+1]=same[x];

        tree[2*x]=same[x]*(mid-l+1);
        tree[2*x+1]=same[x]*(r-mid);

        tree2[2*x]=same[x]*same[x]*(mid-l+1);
        tree2[2*x+1]=same[x]*same[x]*(r-mid);
        same[x]=0;
    }
    if (ql > r || qr < l || l>r) return;
    if (ql<=l && qr>=r){
        tree2[x]=key*key*(r-l+1);
        tree[x]=key*(r-l+1);
        same[x]=key;
        lazy[x]=0;
        no[2*x]=1;
        no[2*x+1]=1;
        return;
    }
    make_same(2*x,l,mid,ql,qr,key);
    make_same(2*x+1,mid+1,r,ql,qr,key);
    tree[x]=tree[2*x]+tree[2*x+1];
    tree2[x]=tree2[2*x]+tree2[2*x+1];
}
ll query(int x,int l,int r,int ql,int qr){
    int mid=l+r>>1;
    if (ql<=l && qr>=r) return tree2[x];
    if (no[x]){
        same[x]=lazy[x]=0;
        no[2*x]=1;
        no[2*x+1]=1;
        no[x]=0;
    }
    if (same[x]){
        same[x*2]=same[x];
        same[2*x+1]=same[x];

        tree[2*x]=same[x]*(mid-l+1);
        tree[2*x+1]=same[x]*(r-mid);

        tree2[2*x]=same[x]*same[x]*(mid-l+1);
        tree2[2*x+1]=same[x]*same[x]*(r-mid);
        same[x]=0;
    }
    if (lazy[x]){
        lazy[x*2]=lazy[x];
        lazy[2*x+1]=lazy[x];

        tree2[2*x]+=2*lazy[x]*tree[2*x]+(mid-l+1)*lazy[x]*lazy[x];
        tree2[2*x+1]+=2*lazy[x]*tree[2*x+1]+(r-mid)*lazy[x]*lazy[x];

        tree[2*x]+=lazy[x]*(mid-l+1);
        tree[2*x+1]+=lazy[x]*(r-mid);

        lazy[x]=0;
    }
    if (ql > r || qr < l || l>r) return 0;
    return query(2*x,l,mid,ql,qr)+query(2*x+1,mid+1,r,ql,qr);

}
int q;
int main(){
    int t;
    scanf("%d",&t);
    for(int cas=1;cas<=t;cas++){
        printf("Case %d:\n",cas);
        scanf("%d%d",&n,&q);
        for (int i=1;i<=n;i++){
            scanf("%lld",&a[i]);
        }
        build(1,1,n);
        for (int i=1;i<=q;i++){
            int fl,s,t,x;
            scanf("%d%d%d",&fl,&s,&t);
            switch(fl){
                case 2:
                    printf("%lld\n",query(1,1,n,s,t));
                    break;
                case 1:
                    scanf("%d",&x);
                    add(1,1,n,s,t,x);
                    break;
                case 0:
                    scanf("%d",&x);
                    make_same(1,1,n,s,t,x);
                    break;
            }
        }
    }
    return 0;
}
