/****************************************************************************
  https://blog.csdn.net/qq_35649707/article/details/66473069
  https://www.cnblogs.com/mangoyang/p/9760416.html
  https://www.cnblogs.com/candy99/p/6374177.html
  
  每个节点表示某一endpos等价类（标记为S）
  连边nxt[x][c]=y表示Sx后面加上字符c就变成了Sy
  fa[x]=y表示Sx是Sy的真子集，x节点的孩子失配时利用fa[x]跳转，查找时相当于放宽了限制
****************************************************************************/
struct SAM
{
    int nxt[N][27], fa[N], len[N];
    int sz[N];
    int root, tot, last;
    bool T;//T=1位置不同的相同子串算多个;T=0本质相同的子串只算一次
    int newnode(int l)
    {
        ++tot;
        fa[tot] = -1;
        for (int i = 0; i < 27; ++i)
            nxt[tot][i] = -1;
        len[tot] = l;
        return tot;
    }
    void init()
    {
        tot = -1;
        last = root = newnode(0);
    }
    void extend(int x)
    {
        int p = last;
        int cur = newnode(len[p] + 1);
        sz[cur] = 1;
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

    int id[N];
    int v[N];
    int sum[N];
    
    void getSum(){
        int mx = 0;
        for (int i = 1; i <= tot; i++)
            mx = max(mx, len[i]), ++v[len[i]];
        for (int i = 1; i <= mx; i++)
            v[i] += v[i - 1];
        for (int i = 1; i <= tot; i++){
            id[v[len[i]]--] = i;
        }
        for (int i = tot; i; i--)
        {
            int t = id[i];
            if (T)
                sz[fa[t]] += sz[t];
            else
                sz[t] = 1;
        }
        // sz[0] = 0;//空串不能算在内
        for (int i = tot; ~i;i--)
        {
            int t = id[i];
            sum[t] = sz[t];
            for (int j = 0; j < 26;j++) if (~nxt[t][j])
            {
                 sum[t] += sum[nxt[t][j]];
            }
        }
    }
} sam;
