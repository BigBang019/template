namespace ST_2D{
    using namespace std;
    typedef long long ll;
    const int N = 5e2 + 5;
    int maze[N][N];
    int LOG[N];
    int dp[N][N][10][10];
    int mp[N][N][10][10];
    inline int Max(int a,int b){
        return a > b ? a : b;
    }
    inline int Min(int a,int b){
        return a < b ? a : b;
    }
    void init(){
        for (register int i = 2; i <= N - 5; i++)
        {
            LOG[i] = LOG[i >> 1] + 1;
        }
    }
    inline void ST(int n,int m){
        for (register int i = 1; i <= n;i++)
        {
            for (register int j = 1; j <= m;j++)
            {
                dp[i][j][0][0] = maze[i][j];
                mp[i][j][0][0] = maze[i][j];
            }
        }
        for (register int k = 0; (1 << k) <= n;k++)
        {
            for (register int l = 0; (1 << l) <= m;l++)
            {
                if (l==0 && k==0)
                    continue;
                for (register int i = 1; i + (1 << k) - 1 <= n;i++)
                {
                    for (register int j = 1; j + (1 << l) - 1 <= m;j++)
                    {
                        if (k==0){
                            dp[i][j][k][l] = Max(dp[i][j][k][l - 1], dp[i][j + (1 << (l - 1))][k][l - 1]);
                            mp[i][j][k][l] = Min(mp[i][j][k][l - 1], mp[i][j + (1 << (l - 1))][k][l - 1]);
                        }
                        else{
                            dp[i][j][k][l] = Max(dp[i][j][k - 1][l], dp[i + (1 << (k - 1))][j][k - 1][l]);
                            mp[i][j][k][l] = Min(mp[i][j][k - 1][l], mp[i + (1 << (k - 1))][j][k - 1][l]);
                        }
                    }
                }
            }
        }
    }
    inline int query(int x1,int x2,int y1,int y2){
        int k = LOG[x2 - x1 + 1], l = LOG[y2 - y1 + 1];
        int ans = dp[x1][y1][k][l];
        ans = Max(ans, dp[x1][y2 - (1 << l) + 1][k][l]);
        ans = Max(ans, dp[x2 - (1 << k) + 1][y1][k][l]);
        ans = Max(ans, dp[x2 - (1 << k) + 1][y2 - (1 << l) + 1][k][l]);
        return ans;
    }
    inline int queryM(int x1,int x2,int y1,int y2){
        int k = LOG[x2 - x1 + 1], l = LOG[y2 - y1 + 1];
        int ans = mp[x1][y1][k][l];
        ans = Min(ans, mp[x1][y2 - (1 << l) + 1][k][l]);
        ans = Min(ans, mp[x2 - (1 << k) + 1][y1][k][l]);
        ans = Min(ans, mp[x2 - (1 << k) + 1][y2 - (1 << l) + 1][k][l]);
        return ans;
    }
}
using namespace ST_2D;
