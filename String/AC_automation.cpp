namespace AC{
    typedef long long ll;
    using namespace std;
    const int N = 1e6 + 5;
    int nxt[N][30], fail[N], cnt[N];
    int tot, root;
    int newNODE(){
        tot++;
        for (int i = 0; i < 26;i++)
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
        for (int i = 0; i < 26;i++)
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
            for (int i = 0; i < 26;i++)
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
                cnt[tmp] = 0;           //已经被统计过，所以置0
                tmp = fail[tmp];
            }
        }
        return res;
    }
}
using namespace AC;
