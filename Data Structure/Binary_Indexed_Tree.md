# 二位树状数组

## 单点修改区间查询

```c++
/*
	维护a[i][j]数组
	查询结果为(x2,y2)-(x1-1,y2)-(x2,y1-1)+(x1-1,y1-1)
*/
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
```

## 区间修改单点查询

```c++
/*
	维护差分数组d[i][j]
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1050;
int n, c[N][N];
int lowbit(int x){
	return x & (-x);
}
void add(int x,int y,int v){
	for (int i = x; i <= n;i+=lowbit(i))
	{
		for (int j = y; j <= n;j+=lowbit(j))
		{
			c[i][j] += v;
		}
	}
}
int query(int x,int y){
	int sum = 0;
	for (int i = x; i > 0;i-=lowbit(i))
	{
		for (int j = y; j > 0;j-=lowbit(j)){
			sum += c[i][j];
		}
	}
	return sum;
}
void add(int x1,int x2,int y1,int y2){
	add(x1, y1, 1);
	add(x1, y2 + 1, -1);
	add(x2 + 1, y1, -1);
	add(x2 + 1, y2 + 1, 1);
}
```

## 区间修改区间查询

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\sum_{i=1}^{x}\sum_{j=1}^{y}\sum_{k=1}^i\sum_{l=1}^j&space;d[k][l]&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\sum_{i=1}^{x}\sum_{j=1}^{y}\sum_{k=1}^i\sum_{l=1}^j&space;d[k][l]&space;$$" title="$$\sum_{i=1}^{x}\sum_{j=1}^{y}\sum_{k=1}^i\sum_{l=1}^j d[k][l] $$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\newline=\sum_i^x\sum_j^yd[i][j]*(x-i&plus;1)*(y-j&plus;1)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\newline=\sum_i^x\sum_j^yd[i][j]*(x-i&plus;1)*(y-j&plus;1)$$" title="$$\newline=\sum_i^x\sum_j^yd[i][j]*(x-i+1)*(y-j+1)$$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\newline=(x&plus;1)*(y&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]-(y&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]*i-(x&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]*j&plus;\sum_i^x\sum_j^y&space;d[i][j]*i*j$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\newline=(x&plus;1)*(y&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]-(y&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]*i-(x&plus;1)*\sum_i^x\sum_j^y&space;d[i][j]*j&plus;\sum_i^x\sum_j^y&space;d[i][j]*i*j$$" title="$$\newline=(x+1)*(y+1)*\sum_i^x\sum_j^y d[i][j]-(y+1)*\sum_i^x\sum_j^y d[i][j]*i-(x+1)*\sum_i^x\sum_j^y d[i][j]*j+\sum_i^x\sum_j^y d[i][j]*i*j$$" /></a>


```c++

int lowbit(int x){
    return x & (-x);
}
void add(ll x, ll y, ll z){
    for(int i = x; i <= n; i += lowbit(i))
        for(int j = y; j <= m; j += lowbit(j)){
            c1[i][j] += z;
            c2[i][j] += z * x;
            c3[i][j] += z * y;
            c4[i][j] += z * x * y;
        }
}
void range_add(ll x1, ll x2, ll y1, ll y2, ll z){ //(xa, ya) 到 (xb, yb) 的矩形
    add(x1, y1, z);
    add(x1, y2 + 1, -z);
    add(x2 + 1, y1, -z);
    add(x2 + 1, y2 + 1, z);
}
ll ask(ll x, ll y){
    ll res = 0;
    for(int i = x; i; i -= lowbit(i))
        for(int j = y; j; j -= lowbit(j))
            res += (x + 1) * (y + 1) * c1[i][j]
                - (y + 1) * c2[i][j]
                - (x + 1) * c3[i][j]
                + c4[i][j];
    return res;
}
ll range_ask(ll x1, ll x2, ll y1, ll y2){
    return ask(x2, y2) - ask(x2, y1 - 1) - ask(x1 - 1, y2) + ask(x1 - 1, y1 - 1);
}
```

