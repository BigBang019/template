struct Point{
	ll x, y;
	Point(){}
	Point(int x, int y) : x(x), y(y){}
	Point operator + (const Point& b) const {
		return Point(x + b.x, y + b.y);
	}
	Point operator - (const Point& b) const {
		return Point(x - b.x, y - b.y);
	}
	ll operator * (const Point& b) const {
		return x * b.y - b.x * y;
	}
};
struct Convex{
	vector<Point> st;
	Convex() { st.clear(); }
	static ll cross_product(const Point& p2,const Point& p0,const Point& p1){
		return (p2 - p0) * (p1 - p0);
	}
	void insert(const Point& a){
		int sz = st.size();
		while (sz > 1 && cross_product(a, st[sz - 2], st[sz - 1]) <= 0)
		{
			st.pop_back();
			--sz;
		}
		st.push_back(a);
	}
	bool query(const Point& a,const Point& b){
		int l = 0, r = int(st.size()) - 1;
		while (l<r){
			int mid = l + r >> 1;
			if (cross_product(st[mid], a, b) < cross_product(st[mid+1], a, b)){
				r = mid;
			}else{
				l = mid + 1;
			}
		}
		return cross_product(st[l], a, b) < 0 || cross_product(st[l + 1], a, b) < 0;
	}
};
