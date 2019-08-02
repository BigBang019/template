/****************************************************************************
  https://www.cnblogs.com/hyghb/p/8445112.html
  https://www.cnblogs.com/mangoyang/p/9760416.html
****************************************************************************/
struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int root, tot, last;
    ll dp[N];
    int newnode(int l)
    {
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot++] = l;
        return tot - 1;
    }
    void init()
    {
        tot = 0;
        memset(dp, -1, sizeof(dp));
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        while (p != -1 && nxt[p][x] == -1)
        {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == -1)
            fa[cur] = root;
        else
        {
            int q = nxt[p][x];
            if (len[q] == len[p] + 1)
                fa[cur] = q;
            else
            {
                int tmp = newnode(len[p] + 1);
                memcpy(nxt[tmp], nxt[q], sizeof(nxt[q]));
                fa[tmp] = fa[q];
                fa[q] = fa[cur] = tmp;
                while (p != -1 && nxt[p][x] == q)
                {
                    nxt[p][x] = tmp;
                    p = fa[p];
                }
            }
        }
        last = cur;
    }

    ll dfs(int u)
    {
        if (dp[u] != -1)
            return dp[u];
        ll res = 1;
        for (int i = 0; i < 26; ++i)
        {
            if (nxt[u][i] == -1)
                continue;
            res += dfs(nxt[u][i]);
        }
        return dp[u] = res;
    }
} sam;
