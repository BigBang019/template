#include<cstdio>
#include<cstring>
#include<time.h>
#include<iostream>
using namespace std;
typedef long long ll;
const int N = 1050;
int n, sign;
ll c[N][N];
int lowbit(int x){
	return x & (-x);
}
void add(int x,int y,ll v){
	for (int i = x; i <= n;i+=lowbit(i))
	{
		for (int j = y; j <= n;j+=lowbit(j))
		{
			c[i][j] += v;
		}
	}
}
ll query(int x,int y){
	ll sum = 0;
	for (int i = x; i > 0;i-=lowbit(i))
	{
		for (int j = y; j > 0;j-=lowbit(j))
		{
			sum += c[i][j];
		}
	}
	return sum;
}
ll getSum(int x1,int x2,int y1,int y2){
	return query(x2, y2) - query(x2, y1 - 1) - query(x1 - 1, y2) + query(x1 - 1, y1 - 1);
}
int main(){
	while (scanf("%d",&sign) && sign!=3){
		if (!sign){
			scanf("%d",&n);
			n++;
			memset(c,0,sizeof(c));
		}else if (sign==1){
			int x, y;
			ll v;
			scanf("%d%d%lld",&x,&y,&v);
			x++;
			y++;
			add(x,y,v) ;
		}else{
			int l, b, r, t;
			scanf("%d%d%d%d",&l,&b,&r,&t);
			l++;
			r++;
			b++;
			t++;
			printf("%d\n",getSum(l,r,b,t));
		}
	}
	return 0;
}
