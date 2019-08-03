/***********************************************************************************
    点：除奇/偶根节点外的每一个节点代表一种回文串
    边：nxt[x][c]=y表示节点x表示的回文串左右加上字符c形成的字符串节点为y
    ------------------------------------------------------------------------
    fail[x]：x失配后跳转到的不等于自身的最长后缀回文子串
        若fail[x]=y，y节点串一定是x节点串的后缀
    len[x]：以x为结尾的最长回文子串的长度
    cnt[x]：与以x结尾的最长回文子串相同的子串的个数
    nxt[x][c]：编号为x的节点表示的回文串在两边添加字符c以后变成的回文串的编号
    s[x]：第x次添加的字符（一开始设S[0] = -1，也可以是任意一个在串S中不会出现的字符）
    https://www.cnblogs.com/nbwzyzngyl/p/8260921.html
    https://blog.csdn.net/stevensonson/article/details/81748093
***********************************************************************************/
struct PAM{
    int nxt[N][27]; //next指针，next指针和字典树类似，指向的串为当前串两端加上同一个字符构成
    int fail[N];    //fail指针，失配后跳转到fail指针指向的节点
    int cnt[N];
    int num[N];
    int len[N]; //len[i]表示节点i表示的回文串的长度
    int S[N];   //存放添加的字符
    int last;   //指向上一个字符所在的节点，方便下一次add
    int n;      //字符数组指针
    int p;      //节点指针

    int newnode(int l){ //新建节点
        for (int i = 0; i < N; ++i)
            nxt[p][i] = 0;
        cnt[p] = 0;
        num[p] = 0;
        len[p] = l;
        return p++;
    }

    void init(){ //初始化
        p = 0;
        newnode(0);
        newnode(-1);
        last = 0;
        n = 0;
        S[n] = -1; //开头放一个字符集中没有的字符，减少特判
        fail[0] = 1;
    }

    int get_fail(int x){ //和KMP一样，失配后找一个尽量最长的
        while (S[n - len[x] - 1] != S[n])
            x = fail[x];
        return x;
    }

    void extend(int c){
        S[++n] = c;
        int cur = get_fail(last); //通过上一个回文串找这个回文串的匹配位置
        if (!nxt[cur][c])
        {                                            //如果这个回文串没有出现过，说明出现了一个新的本质不同的回文串
            int now = newnode(len[cur] + 2);         //新建节点
            fail[now] = nxt[get_fail(fail[cur])][c]; //和AC自动机一样建立fail指针，以便失配后跳转
            nxt[cur][c] = now;
            num[now] = num[fail[now]] + 1;
        }
        last = nxt[cur][c];
        cnt[last]++;
    }

    void count(){
        for (int i = p - 1; i >= 0; --i)
            cnt[fail[i]] += cnt[i];
        //父亲累加儿子的cnt，因为如果fail[v]=u，则u一定是v的子回文串！
    }
};
