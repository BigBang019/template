
#include<cstdio>
#include<iostream>
#include<cstring>
#include<string>
#include<cstdlib>
#define N 200005
#define ma(a,b) ((a)>(b)?(a):(b))
#define mi(a,b) ((a)<(b)?(a):(b))
using namespace std;
struct Node{
	Node *l,*r;
	int h;
	long long key;
	int num;
	int v;
};

class AVL{
	Node* root;
	int cnt;
	int tot;
	int ex;
	bool f;
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
		return t;
	}
	static int h(Node* t){
		return (t?t->h:0);
	}
	static int g(Node* t){
		return (t?t->num:0);
	}
	static Node* llr(Node* k2){
		Node* k1;
		k1=k2->l;
		k2->l=k1->r;
		k1->r=k2;
			
		k2->h=ma(h(k2->l),h(k2->r))+1;
		k1->h=ma(h(k1->l),k2->h)+1;
		
		k2->num=g(k2->l)+g(k2->r)+1;
		k1->num=g(k1->l)+g(k2)+1;
		return k1;
	}
	static Node* rrr(Node* k1){
		Node* k2;
		k2=k1->r;
		k1->r=k2->l;
		k2->l=k1;
		
		k1->h=ma( h(k1->l), h(k1->r) )+1;
		k2->h=ma( h(k2->r), k1->h )+1;
		
		k1->num=g(k1->l)+g(k1->r)+1;
		k2->num=g(k2->r)+g(k1)+1;
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
	int update(Node* k){
		if (k==NULL) return 0;
		int tmp=k->v;
		if (k->l) tmp+=update(k->l);
		if (k->r) tmp+=update(k->r);
		k->num=tmp;
		return tmp;
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
			
		}else{//same?
			k->v++;
			f=1;//same?
		}
		k->h = ma( h(k->l), h(k->r))+1;
		return k;
	}
	Node* el(Node* k,Node* t){
		if (k==NULL || t==NULL){
			
			return NULL;
			
		}else if (t->key < k->key){
			
			k->l=el(k->l,t);
			if (h(k->r) - h(k->l) == 2){
				Node* p=k->r;
				if (h(p->l) > h(p->r))
					k=rlr(k);
				else
					k=rrr(k);
			}
			
		}else if (t->key > k->key){
			
			k->r=el(k->r,t);
			if (h(k->l) - h(k->r) == 2){
				Node* p=k->l;
				if (h(p->l) > h(p->r))
					k=llr(k);
				else
					k=lrr(k);
			}
			
		}else{
			
			if (!f){
				ex=k->v;
				f=1;
			}
			if ((k->l!=NULL) && (k->r!=NULL)){
				if (h(k->l) > h(k->r)){
					Node* p = findmax(k->l);
					k->key=p->key;
					k->l=el(k->l,p);
				}else {
					Node* p = findmin(k->r);
					k->key=p->key;
					k->r=el(k->r,p);
			}
			}else {
				Node* tmp=k;
				k= k->l ? k->l:k->r;
				delete(tmp);
			}
			
		}
		return k;
	}
	public:
		
		void init(){
			cnt=0;
			tot=0;
			destroy(root);
			root=NULL;
		}
		
		void insert(long long key){
			f=0;
			tot++;
			root=nsert(root,key);
			root->num=update(root);
			if (!f) cnt++;
		}
		
		void del(Node* t){
			f=0;
			ex=0;
			root=el(root,t);
			root->num=update(root);
			if (f) {
				cnt--;
				tot-=ex;
			}
		}
		
		Node* search(long long key){
			Node* k=root;
			while ((k!=NULL) && (k->key>=key)){
				if (key<k->key)
					k=k->l;
				else
					k=k->r;
			}
			return k;
		}
		Node* searchRank(int x){
			if (x>tot) return NULL;
			int sum=x;
			Node* k=root;
			while (k!=NULL){
				if (sum>g(k->l) && sum<=g(k->l)+k->v) break;
				if (sum<=g(k->l))
					k=k->l;
				else{
					sum-=g(k->l)+k->v;
					k=k->r;
				}
			}
			return k;
		}
		void print(){
			inorder(root);
			printf("\n");
		}
		void inorder(Node* k){
			if (k->l) inorder(k->l);
			printf("%lld ",k->key);
			if (k->r) inorder(k->r);
		}
		int size(){
			return tot;
		}
		Node* getMax(){
			return findmax(root);
		}
		Node* getMin(){
			return findmin(root);
		}
		
}
