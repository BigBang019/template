// Dijkstra 分层
// 可以将k条路权重置为0
 
#include<cstdio>
#include<iostream>
#include<queue>

using namespace std;
const int maxn=1005;
const int maxm=2010;
const int INF = 0x3f3f3f3f;

struct Node{
	int dist;
	int index;
	int l;
	bool operator < (const Node &x) const {
		return dist>x.dist; 
	}
	Node(int x,int y, int z) {
		index=x;
		dist=y;
		l = z;
	}
};
int n,m,k;

struct Edge{
	int x,y;
	int next;
	int w;
} edge[maxm];
int head[maxn];
int tot;

inline void init() {
	for (int i=1;i<=n;i++) {
		head[i]=0;
	}
	tot=0;
}

inline void addedge(int x, int y,int w) {
	tot++;
	edge[tot].next=head[x];
	edge[tot].x=x;
	edge[tot].y=y;
	edge[tot].w=w;
	head[x]=tot;
}

int dis[maxn][maxn];

priority_queue<Node> que;

void Dijkstra(int x) {
	for (int i=1;i<=n;i++) {
		for (int j=0;j<=k;j++) {
			dis[i][j]=INF;
		}
	}
	while (!que.empty()) {
		que.pop();
	}
	dis[x][0]=0;
	que.push(Node(x,0,0));
	while (!que.empty()) {
		Node t = que.top();
		que.pop();
		int u=t.index;
		int dist = t.dist;
		int l = t.l;
		if (dis[u][l]<dist) {
			continue;
		}
		
		for (int i=head[u];i!=0;i=edge[i].next) {
			int v = edge[i].y;
			if (dis[u][l]+edge[i].w<dis[v][l]) {
				dis[v][l] = dis[u][l]+edge[i].w;
				que.push(Node(v,dis[v][l],l));
			}
			if (l<k && dis[u][l]<dis[v][l+1]) {
				dis[v][l+1] = dis[u][l];
				que.push(Node(v,dis[v][l+1],l+1));
			}
		}
//		printf("%d\n", u);
//		for (int i=0;i<=k;i++) {
//			printf("%d ", dis[u][i]);
//		}printf("\n");
	}
	
}


int main() {
	int s, t;
    scanf("%d%d%d%d%d", &n, &m, &s, &t, &k);
    init();
	for (int i = 1; i <= m;i++){
        int x, y, z;
        scanf("%d%d%d", &x, &y, &z);
        addedge(x, y, z);
        addedge(y, x, z);
    }
    Dijkstra(s);
    int ans = INF;
    for (int i = 0; i <= k;i++) {
    	ans = min(dis[t][i], ans);
	}
    printf("%d\n", (s == t ? 0 : ans));
    return 0;
} 
