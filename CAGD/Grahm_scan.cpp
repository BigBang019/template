#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 5e5 + 5;
int n;
struct NODE{
	ll x, y;
	NODE(){}
	NODE(ll x,ll y):x(x),y(y){}
} p[N];
int pos;
ll cross_product(NODE p0,NODE p1,NODE p2){
	return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}
double distance(NODE a,NODE b){
	return sqrt(1.0 * (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
bool cmp(NODE a, NODE b){
	ll tmp = cross_product(p[1], a, b);
	if (tmp==0)
		return distance(p[1], a) < distance(p[1], b);
	return tmp > 0;
}
vector<NODE> Grahm_scan(){
	for (int i = 2; i <= n;i++)
	{
		if (p[i].y<p[1].y){
			swap(p[1],p[i]);
		}else if (p[i].y==p[1].y && p[i].x<p[1].x){
			swap(p[1],p[i]);
		}
	}
	sort(p+2,p+1+n,cmp);
	// for (int i = 1; i <= n;i++)
	// {
	// 	cout << p[i].x << " " << p[i].y<<endl;
	// }
	vector<NODE> b;
	b.push_back(p[1]);
	b.push_back(p[2]);
	int top = 1;
	for (int i = 3; i <= n;i++){
		while (top>0 && cross_product(b[top-1],p[i],b[top])>0){
			top--;
			b.pop_back();
		}
		b.push_back(p[i]);
		top++;
	}
	return b;
}
