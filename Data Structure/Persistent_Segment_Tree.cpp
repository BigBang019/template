#include<bits/stdc++.h>
using namespace std;
const int N = 6e4 + 5;
const int M = 1e4 + 5;
typedef long long ll;
struct Q{
	int l, r, k;
	Q(){}
	Q(int l, int r, int k) : l(l), r(r), k(k){}
} qu[M];

int a[N], b[N], cnt;
int T[N], S[N], tot;
int sum[N * 40], lc[N * 40], rc[N * 40];
int use[N];
int n, m;

void init(){
	tot = 0;
	cnt = 0;
}
void Init_Hash(){
	sort(b + 1, b + 1 + cnt);
	cnt = unique(b + 1, b + 1 + cnt) - b - 1;
}
int has(int x){
	return (lower_bound(b + 1, b + 1 + cnt, x) - b);
}
int lowbit(int x){
	return x & (-x);
}
void build(int &rt,int l,int r){
	rt = ++tot;
	sum[rt] = 0;
	if (l==r)
		return;
	int mid = l + r >> 1;
	build(lc[rt], l, mid);
	build(rc[rt], mid + 1, r);
}
int update(int last,int l,int r,int p,int v){
	int rt = ++tot;
	int tmp = rt;
	sum[rt] = sum[last] + v;
	while (l<r){
		int mid = l + r >> 1;
		if (p<=mid){
			lc[rt] = ++tot;
			rc[rt] = rc[last];
			rt = lc[rt];
			last = lc[last];
			r = mid;
		}else{
			rc[rt] = ++tot;
			lc[rt] = lc[last];
			rt = rc[rt];
			last = rc[last];
			l = mid + 1;
		}
		sum[rt] = sum[last] + v;
	}
	return tmp;
}
void add(int index,int x,int v){
	for (int i = index; i <= n;i+=lowbit(i))
	{
		S[i] = update(S[i], 1, cnt, x, v);
	}
}
int sumLeft(int x){
	int ans = 0;
	for (int i = x; i;i-=lowbit(i))
	{
		ans += sum[lc[use[i]]];
	}
	return ans;
}
int query(int le,int re,int k){
	int left_root = T[le - 1];
	int right_root = T[re];
	int l = 1, r = cnt;
	for (int i = le - 1; i;i-=lowbit(i))
		use[i] = S[i];
	for (int i = re; i;i-=lowbit(i))
		use[i] = S[i];
	while (l<r){
		int mid = l + r >> 1;
		int tmp = sumLeft(re) - sumLeft(le - 1) + sum[lc[right_root]] - sum[lc[left_root]];
		if (tmp>=k){
			r = mid;
			left_root = lc[left_root];
			right_root = lc[right_root];
			for (int i = le - 1; i;i-=lowbit(i))
				use[i] = lc[use[i]];
			for (int i = re; i;i-=lowbit(i))
				use[i] = lc[use[i]];
		}else{
			l = mid + 1;
			left_root = rc[left_root];
			right_root = rc[right_root];
			for (int i = le - 1; i;i-=lowbit(i))
				use[i] = rc[use[i]];
			for (int i = re; i;i-=lowbit(i))
				use[i] = rc[use[i]];
			k -= tmp;
		}
	}
	return l;
}
void work(){
	init();
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n;i++)
	{
		scanf("%d", &a[i]);
		b[++cnt] = a[i];
	}
	for (int i = 1; i <= m;i++)
	{
		char ch;
		int l, r, k;
		scanf(" %c%d%d", &ch, &l, &r);
		if (ch=='Q'){
			scanf("%d", &k);
			qu[i] = Q(l, r, k);
		}else{
			qu[i] = Q(l, r, -1);
			b[++cnt] = r;
		}
	}
	Init_Hash();
	build(T[0], 1, cnt);
	for (int i = 1; i <= n;i++)
	{
		T[i] = update(T[i - 1], 1, cnt, has(a[i]), 1);
		S[i] = T[0];
	}
	for (int i = 1; i <= m;i++)
	{
		if (~qu[i].k){
			printf("%d\n", b[query(qu[i].l, qu[i].r, qu[i].k)]);
		}else{
			add(qu[i].l, has(a[qu[i].l]), -1);
			add(qu[i].l, has(qu[i].r), 1);
			a[qu[i].l] = qu[i].r;
		}
	}
}
int main(){
	int t;
	scanf("%d", &t);
	while (t--){
		work();
	}
	return 0;
}
