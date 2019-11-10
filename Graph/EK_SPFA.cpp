// 最小费用最大流SPFA

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

int pre[maxn]; //¼ÇÂ¼Â·¾¶ 
int dist[maxn]; //µ½Ã¿¸öµã¿ÉÐÐÔö¹ãÂ·µÄ×îÐ¡·ÑÓÃºÍ 
int vis[maxn];

// starting from 0
inline void init() {
	tot=-1;
	for (int i=0;i<=n;i++) {
		head[i]=-1;
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

queue<int> que;

// dijkstraÄ§¸ÄÐèÒª½â¾ö·´Ïò±ß¸º»·ÎÊÌâ 
bool SPFA() {
	for (int i=0;i<=n;i++) {
		vis[i]=0;
		dist[i]=INF;
	}
	while (!que.empty()) {que.pop();}
	que.push(s);
	dist[s]=0;
	vis[s]=1;
	while (!que.empty()) {
		int u=que.front();
		que.pop();
		vis[u]=0;
		for (int i=head[u];~i; i=edge[i].next) {
			int v = edge[i].to;
			if (edge[i].cap>edge[i].flow && dist[v]>dist[u] + edge[i].cost) {
				dist[v] = dist[u] + edge[i].cost;
				pre[v]=i;
				if (!vis[v]) {
					vis[v]=1;
					que.push(v);
				}
			}
		}
	}
	if(dist[t]!=INF) {
		return true;
	} else {
    	return false;
    }
}

int CostFlow(int &flow) { // EKËã·¨ 
	int mincost = 0;
	while (SPFA()) { // ÄÜÕÒµ½Ôö¹ãÂ·
		int Min = INF;
		for (int i=t;i!=s;i=edge[pre[i]].from) { // Ñ°ÕÒ×îÐ¡Á÷
			Min = min(Min,edge[pre[i]].cap - edge[pre[i]].flow);
		}
		for (int i=t;i!=s; i=edge[pre[i]].from) { //´¦ÀíËùÓÐ±ß 
			edge[pre[i]].flow+=Min;
            edge[pre[i]^1].flow-=Min;
		}
		flow += Min;
		mincost+=(dist[t]*Min);
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
