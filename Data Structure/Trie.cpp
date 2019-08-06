namespace TRIE{
    using namespace std;
    typedef long long ll;
    const int N = 1e6 + 5;
    int tot;
    struct NODE{
        int ne[26], cnt;
        void init(){
            memset(ne, -1, sizeof(ne));
            cnt = 0;
        }
    } tr[N];
    void init(){
        tot = 0;
    }
    void insert(string s){
        int now = 0;
        for (int i = 0; i < s.size();i++)
        {
            int p = s[i] - 'a';
            if (~tr[now].ne[p]){
                now = tr[now].ne[p];
            }else{
                tr[now].ne[p] = ++tot;
                now = tot;
                tr[now].init();
            }
            tr[now].cnt++;
        }
    }
    int query(string s){
        int now = 0;
        for (int i = 0; i < s.size();i++){
            int p = s[i] - 'a';
            if (~tr[now].ne[p]){
                now = tr[now].ne[p];
            }else
                return 0;
        }
        return tr[now].cnt;
    }
}
using namespace TRIE;

/*****************************************************************************
    多个Trie，update要改改
*****************************************************************************/
struct Trie{
	struct NODE{
		int nxt[B];
		int cnt;
		NODE() { memset(nxt, -1, sizeof(nxt)), cnt = 0; }
		int &operator [] (ull i){
			return nxt[i];
		}
	};
	NODE &operator [] (ull i){
		return po[i];
	}
	vector<NODE> po;
	void init(){
		po.clear();
		po.pb(NODE());
	}
	void update(int a){
		int now = 0;
		for (int i = 29; i >= 0; i--)
		{
			bool x = a & (1 << i);
			if (po[now][x]==-1){
				po[now][x] = po.size();
				po.pb(NODE());
			}
			now = po[now][x];
			po[now].cnt++;
		}
	}
};
