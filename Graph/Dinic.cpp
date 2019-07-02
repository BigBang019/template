using namespace std;
int n,m;
int dep[N];
int cnt,he[N],ne[M*2],v[M*2],w[M*2],cur[N];
int tot;
void add(int x,int y,int z){
	cnt++;
	ne[cnt]=he[x];
	he[x]=cnt;
	v[cnt]=y;
	w[cnt]=z;
}
int s,e;
queue<int> q;
bool bfs(){
	memset(dep,0,sizeof(dep));
	while (!q.empty()) q.pop();
	dep[s]=1;
	q.push(s);
	while (!q.empty()){
		int j=q.front();
		q.pop();
		for (int i=he[j];i>-1;i=ne[i]){
			int p=v[i];
			if (w[i] && !dep[p]){
				dep[p]=dep[j]+1;
				q.push(p);
			}
		}
	}
	if (dep[e]==0) return 0;
	return 1;
}
int dfs(int u,int dist){
	if (u==e) return dist;
	else {
		for (int& i=cur[u];i>-1;i=ne[i]){
			int p=v[i];
			if (!w[i] || dep[p]<=dep[u]) continue;
			int di=dfs(p,mi(dist,w[i]));
			if (di){
				w[i]-=di;
				w[i^1]+=di;
				return di;
			}
		}
	}
	return 0;
}
int Dinic(){
	int ans=0;
	while (bfs()){
		for (int i=1;i<=tot;i++){
			cur[i]=he[i];
		}
		while (int di=dfs(s,INT_MAX)){
			ans+=di;
		}
	}
	return ans;
}
