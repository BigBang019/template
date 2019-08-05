namespace EXKMP{
	using namespace std;
	typedef long long ll;
	const int N = 1e6 + 5;
	ll extend[N];
	ll nxt[N];
	ll min(ll x, ll y)
	{
		if (x > y)
			return y;
		return x;
	}
	void getNext(string t)
	{
		memset(nxt, 0, sizeof(nxt));
		ll len = t.length();
		nxt[0] = len;
		ll a, p;
		a = 1;
		while (a < len && t[a] == t[a - 1])
			a++; // 求出长度为1的时候 解为多少
		nxt[1] = a - 1;
		a = 1;
		for (ll i = 2; i < len; i++) // 后续的按照算法来就好
		{
			p = a + nxt[a] - 1;
			if ((i - 1) + nxt[i - a] < p)
				nxt[i] = nxt[i - a]; // 第一种情况 没有超过等于的部分
			else					 // 超过的话就不好直接用next的定义 需要后续的遍历
			{
				ll j = (p - i + 1) > 0 ? (p - i + 1) : 0;
				while (i + j < len && t[i + j] == t[j])
					j++;
				nxt[i] = j;
				a = i;
			}
		}
	}
	void exkmp(string s, string t) // s->extend  t->next
	{
		getNext(t);
		ll a, p; //
		ll slen = s.length();
		ll tlen = t.length();
		a = p = 0;
		ll len = min(s.length(), t.length());
		while (p < len && t[p] == s[p])
			p++; // after
		extend[0] = p;
		for (ll i = 1; i < slen; i++)
		{
			p = a + extend[a] - 1; // update
			if ((i - 1) + nxt[i - a] < p)
				extend[i] = nxt[i - a];
			else
			{
				ll j = (p - i + 1) > 0 ? (p - i + 1) : 0;
				while (j < tlen && i + j < slen && s[i + j] == t[j])
					j++;
				extend[i] = j;
				a = i;
			}
		}
	}
}
using namespace EXKMP;
