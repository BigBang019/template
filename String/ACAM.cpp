/*
    fail指针指向与当前有最长公共前缀的字符位置，相当于常数优化前的kmp的next
    nxt[x][i]指针指向x节点匹配，但是它的下一个字符i失配时下次直接匹配的位置，它的意义有两个：
        当x的nxt[i]存在时，它的意义仍是Trie的next
        当x的nxt[i]不存在时，它的意义类似优化常数后的kmp的next数组（nxt[i]失配时应该直接匹配的位置）
    因为最长公共前缀end的节点深度一定小于我，实现时使用bfs
*/
struct ACAM{
    int nxt[N][B], fail[N], cnt[N];
    int tot, root;
    int newNODE(){
        tot++;
        for (int i = 0; i < B;i++)
            nxt[tot][i] = -1;
        cnt[tot] = 0;
        return tot;
    }
    void init(){
        tot = 0;
        root = newNODE();
    }
    void update(char *s){
        int len = strlen(s);
        int now = root;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            if (nxt[now][p]==-1)
                nxt[now][p] = newNODE();
            now = nxt[now][p];
        }
        cnt[now]++;
    }
    void build(){
        queue<int> q;
        fail[root] = root;
        for (int i = 0; i < B;i++)
        {
            if (nxt[root][i]==-1)
                nxt[root][i] = root;
            else{
                int x = nxt[root][i];
                fail[x] = root;
                q.push(x);
            }
        }
        while (!q.empty()){
            int now = q.front(); 
            q.pop();
            for (int i = 0; i < B;i++)
            {
                if (nxt[now][i]==-1)
                    nxt[now][i] = nxt[fail[now]][i];
                else{
                    int x = nxt[now][i];
                    fail[x] = nxt[fail[now]][i];
                    q.push(x);
                }
            }
        }
    }
    int query(char *s){
        int len = strlen(s);
        int now = root;
        int res = 0;
        for (int i = 0; i < len;i++)
        {
            int p = s[i] - 'a';
            now = nxt[now][p];
            int tmp = now;
            while (tmp!=root){
                res += cnt[tmp];
                cnt[tmp] = 0;
                tmp = fail[tmp];
            }
        }
        return res;
    }
} ac;
