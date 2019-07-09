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
		int fa, ch[2], v, sz;
		bool rev;
		NODE(){
			fa = ch[0] = ch[1] = 0;
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
	void reverse(int x){					//LCT中的splay的有序性与深度相关
		swap(tr[x].ch[0], tr[x].ch[1]);		//x变为根时，交换左右子树，保证中序遍历的有序性
		tr[x].rev ^= 1;
	}
	void pushdown(int x){					//向下传递reverse的lazy标记
		if (tr[x].rev){
			if (tr[x].ch[0])
				reverse(tr[x].ch[0]);
			if (tr[x].ch[1])
				reverse(tr[x].ch[1]);
			tr[x].rev ^= 1;
		}
	}
	//*********************************************
	void pushup(int x){
		tr[x].sz = tr[tr[x].ch[0]].sz + tr[tr[x].ch[1]].sz + 1;
	}
	//*********************************************
	int st[N];
	void pushto(int x){						//????
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
	
	void rotate(int x){						//旋转，与正常splay不同的原因是判断根节点的方式变了
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
	void splay(int x){						//与正常splay不同的原因是判断根节点的方式变了
		pushto(x);
		for (int fa = tr[x].fa; !isroot(x);rotate(x),fa=tr[x].fa)
		{
			if (!isroot(fa))
				rotate(isleft(fa) == isleft(x) ? fa : x);
		}
		pushup(x);
	}
	void access(int x){						//打通根节点到x的一条偏爱路径
		for (int p = 0; x; x = tr[p = x].fa)
		{
			splay(x);
			tr[x].ch[1] = p;
			pushup(x);
		}
	}
	void makert(int x){						//make root
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
	void split(int x,int y){				//打通x-y的一条偏爱路径
		makert(x);
		access(y);
		splay(y);
	}
	void link(int x,int y){
		split(x, y);
		tr[x].fa = y;
	}
	void cut(int x,int y){
		split(x, y);
		if (tr[y].ch[0]!=x || tr[x].ch[1])
			return;
		tr[x].fa = tr[y].ch[0] = 0;
	}
	void modify(int x,int v){
		access(x);
		splay(x);
		tr[x].v = v;
		pushup(x);
	}
	
}
using namespace LCT;
