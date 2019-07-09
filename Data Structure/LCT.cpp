/*
	节点编号从1开始，
	不存在标志为0
	一棵splay相当于是一条链，因为一个节点的实边只有一条
	splay的中序遍历的节点编号在原树的深度是递增的，因此深度相同的两个节点不可能在一个splay中
 */
namespace LCT{
	using namespace std;
	const int N = 3e5 + 5;
	struct NODE{
		int fa, ch[2], v, sz, mx, tag;
		bool rev;
		NODE(){
			fa = ch[0] = ch[1] = tag = mx = 0;
			rev = 0;
			sz = 1;
		}
	} tr[N];
	bool isroot(int x){
		return tr[tr[x].fa].ch[0] != x && tr[tr[x].fa].ch[1] != x;
	}
	bool isleft(int x){
		return tr[tr[x].fa].ch[0] == x;
	}
	void reverse(int x){
		swap(tr[x].ch[0], tr[x].ch[1]);
		tr[x].rev ^= 1;
	}
	void pushdown(int x){
		int l = tr[x].ch[0], r = tr[x].ch[1];
		if (tr[x].rev){
			if (l)
				reverse(l);
			if (r)
				reverse(r);
			tr[x].rev ^= 1;
		}
		if (tr[x].tag){
			if (l) {
				tr[l].tag += tr[x].tag;
				tr[l].mx += tr[x].tag;
				tr[l].v += tr[x].tag;
			}
			if (r){
				tr[r].tag += tr[x].tag;
				tr[r].mx += tr[x].tag;
				tr[r].v += tr[x].tag;
			}
			tr[x].tag = 0;
		}
	}
	//*********************************************************
	void pushup(int x){
		int l = tr[x].ch[0], r = tr[x].ch[1];
		tr[x].sz = tr[l].sz + tr[r].sz + 1;
		tr[x].mx = max(tr[l].mx, max(tr[r].mx, tr[x].v));
	}
	//*********************************************************
	int st[N];
	void pushto(int x){
		int top = 0;
		while (!isroot(x)){
			st[top++] = x;
			x = tr[x].fa;
		}
		st[top++] = x;
		while (top){
			pushdown(st[--top]);
		}
	}
	
	void rotate(int x){
		bool t = !isleft(x);
		int fa = tr[x].fa, ffa = tr[fa].fa;

		tr[x].fa = ffa;
		if (!isroot(fa)) tr[ffa].ch[!isleft(fa)] = x;

		tr[fa].ch[t] = tr[x].ch[!t];
		tr[tr[fa].ch[t]].fa = fa;

		tr[x].ch[!t] = fa;
		tr[fa].fa = x;
		pushup(fa);
	}
	void splay(int x){
		pushto(x);
		for (int fa = tr[x].fa; !isroot(x);rotate(x),fa=tr[x].fa)
		{
			if (!isroot(fa))
				rotate(isleft(fa) == isleft(x) ? fa : x);
		}
		pushup(x);
	}
	void access(int x){
		for (int p = 0; x; x = tr[p = x].fa)
		{
			splay(x);
			tr[x].ch[1] = p;
			pushup(x);
		}
	}
	void makert(int x){
		access(x);
		splay(x);
		reverse(x);
	}
	int findrt(int x){
		access(x);
		splay(x);
		while (tr[x].ch[0])
			x = tr[x].ch[0];
		return x;
	}
	void split(int x,int y){
		makert(x);
		access(y);
		splay(y);
	}
	void link(int x,int y){
		makert(x);
		tr[x].fa = y;
	}
	void cut(int x,int y){
		split(x, y);
		tr[tr[y].ch[0]].fa = 0;
		tr[y].ch[0] = 0;
		pushup(y);
	}

	//*********************************************************
	void modify(int x,int y,int v){
		split(x, y);
		tr[y].tag += v;
		tr[y].mx += v;
		tr[y].v += v;
	}
	int query(int x,int y){
		split(x, y);
		return tr[y].mx;
	}
	void init(int n){
		for (int i = 0; i <= n;i++)
		{
			tr[i] = NODE();
		}
	}
	//*********************************************************
}
using namespace LCT;
