
// poj 1469
#include<cstdio>
#include<cstring>
using namespace std;
const int N=505;
typedef long long ll;
bool maze[N][N],vis[N];
int mat[N];
int p,n;
bool match(int x){
    for (int i=p+1;i<=n+p;i++){
        if (!vis[i] && maze[x][i]){
            vis[i]=1;
            if (mat[i]==-1 || match(mat[i])){
                mat[i]=x;
                mat[x]=i;
                return 1;
            }
        }
    }
    return 0;
}
int getSum(){
    int ans=0;
    for (int i=1;i<=p;i++){
        memset(vis,0,sizeof(vis));
        if (match(i)) ans++;
    }
    return ans;
}
int main(){
    int t;
    scanf("%d",&t);
    while (t--){
        memset(mat,-1,sizeof(mat));
        memset(maze,0,sizeof(maze));
        scanf("%d%d",&p,&n);
        for (int i=1;i<=p;i++){
            int num;
            scanf("%d",&num);
            for (int j=1;j<=num;j++){
                int x;
                scanf("%d",&x);
                maze[i][x+p]=1;
                maze[x+p][i]=1;
            }
        }
        int ans=getSum();
        printf("%s\n",ans==p?"YES":"NO");
    }
    return 0;
}
