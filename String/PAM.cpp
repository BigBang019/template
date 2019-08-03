struct PAM{
    int nxt[N][26], fail[N], len[N];
    int txt[N];
    int tot, root0, root1, last, size;
    void init(){
        last = tot = size = 0;
        txt[size] = -1;
        root0 = newnode(0);
        root1 = newnode(-1);
        fail[root0] = 1;
        fail[root1] = 0;
    }
    int newnode(int l){
        len[tot] = l;
        memset(nxt[tot], 0, sizeof(nxt[tot]));
        tot++;
        return tot - 1;
    }
    int getfail(int x){
        while(txt[size-len[x]-1]!=txt[size]) {
            x = fail[x];
        }
        return x;
    }
    void extend(int c){
        txt[++size] = c;
        int now = getfail(last);
        if(!nxt[now][c]){
            int tmp = newnode(len[now] + 2);
            fail[tmp] = nxt[getfail(fail[now])][c];
            nxt[now][c] = tmp;
        }
        last = nxt[now][c];
    }
    
};
