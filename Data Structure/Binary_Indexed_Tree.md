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

