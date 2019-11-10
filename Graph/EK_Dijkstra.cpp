// 最小费用最大流 Dijkstra
#include<cstdio>
#include<iostream>
#include<queue>
using namespace std;
const int maxn=10000;
const int maxm=200000;
const int INF = 0x3f3f3f3f;

struct Edge{
	int from;
	int to;
	int next;
	int cap;
	int flow;
	int cost;
} edge[maxm];

int head[maxn];
int tot;
int n,m;
int s,t;

int pre[maxn]; //记录路径 
int dist[maxn]; //到每个点可行增广路的最小费用和 



struct Node{
	int x;
	int dist;
	bool operator < (const Node &p) const {
		return dist>p.dist;
	}
};

priority_queue<Node> que;
int h[maxn];  // 用势能 代替边权 e' = e + h[u] - h[v]


// starting from 0
inline void init() {
	tot=-1;
	for (int i=0;i<=n;i++) {
		head[i]=-1;
		h[i]=0;
	}
}

inline void addedge(int x,int y,int c,int w) {
	tot++;
	edge[tot].next=head[x];
	edge[tot].from=x;
	edge[tot].to=y;
	edge[tot].flow=0;
	edge[tot].cap=c;
	edge[tot].cost=w;
	head[x]=tot;
	tot++;
	edge[tot].next=head[y];
	edge[tot].from=y;
	edge[tot].to=x;
	edge[tot].flow=0;
	edge[tot].cap=0;
	edge[tot].cost=-w;
	head[y]=tot;
}



// dijkstra魔改需要解决反向边负环问题 
// 可以证明不可能有负环，所以只要魔改负权边 

bool Dijkstra() {
	for (int i=0;i<=n;i++) {
		dist[i]=INF;
	}
	while (!que.empty()) {que.pop();}
	Node tmp;
	tmp.dist=0;
	tmp.x=s;
	que.push(tmp);
	dist[s]=0;
	
	while (!que.empty()) {
		tmp = que.top();
		int u=tmp.x;
		
		int dis=tmp.dist;
		que.pop();
		if (dist[u]<dis) {continue;}
		
		for (int i=head[u];~i; i=edge[i].next) {
			int v = edge[i].to;
			if (edge[i].cap>edge[i].flow && dist[v] > dist[u]+ edge[i].cost + h[u]-h[v]){
				dist[v] = dist[u] + edge[i].cost + h[u] - h[v];
				pre[v]=i;
				Node tmp1;
				tmp1.dist = dist[v];
				tmp1.x = v;
				que.push(tmp1);
			}
		}
	}
	for (int i = 0;i <= n;i++) {
		h[i] += dist[i];
	}
	if(dist[t]!=INF) {
		return true;
	} else {
    	return false;
    }
}

int CostFlow(int &flow) { // EK算法 
	int mincost = 0;
	while (Dijkstra()) { // 能找到增广路
	
		int Min = INF;
		for (int i=t;i!=s;i=edge[pre[i]].from) { // 寻找最小流
			Min = min(Min,edge[pre[i]].cap - edge[pre[i]].flow);
		}
		for (int i=t;i!=s; i=edge[pre[i]].from) { //处理所有边 
			edge[pre[i]].flow+=Min;
            edge[pre[i]^1].flow-=Min;
		}
		flow += Min;
		mincost+=(h[t]*Min);
	}
	return mincost;
}

int main() {
	while (~scanf("%d%d%d%d", &n,&m,&s,&t)) {
		
		init();
		for (int i=0;i<m;i++) {
			int x,y,c,w;
			scanf("%d%d%d%d", &x,&y,&c,&w);
			addedge(x,y,c,w);
		}
		int maxFlow = 0;
		int minCost = CostFlow(maxFlow);
		printf("%d %d\n", maxFlow, minCost);
	}
}

/*
4 5 4 3
4 2 30 2
4 3 20 3
2 3 20 1
2 1 30 9
1 3 40 5

50 280
*/
