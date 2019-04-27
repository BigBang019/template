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
