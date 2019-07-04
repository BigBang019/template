/*
  矩阵快速幂
*/
namespace MATRIX{
    using namespace std;
    typedef unsigned long long ull;
    typedef long long ll;
    const int MOD = 1e9 + 6;
    const int MOD2 = 1e9 + 7;

    inline ll add(ll a, ll b) { a += b; return (a >= MOD ? a - MOD : a);}

    inline ll mul(ll a, ll b) { return a * b % MOD; }

    template<typename T>
    class matrix{
        public:
            int n;
            vector<vector<T>> vc;
            vector<T> &operator [] (ull i) {
                return vc[i];
            }
            matrix<T> operator * (matrix<T> & b) const {
                matrix<T> c(n);
                for (int i = 0; i < n;i++)
                {
                    for (int j = 0; j < n;j++)
                    {
                        for (int k = 0; k < n;k++)
                        {
                            c[i][j] = add(c[i][j], mul(vc[i][k], b[k][j]));
                        }
                    }
                }
                return c;
            }
            matrix<T> &operator *= (matrix<T> &b) {
                return *this = *this * b;
            }
            matrix<T> operator ^ (ull k) const {
                matrix<T> c(n);
                matrix<T> tmp = *this;
                for (int i = 0; i < n;i++) c[i][i] = 1;
                while (k){
                    if (k&1){
                        c *= tmp;
                    }
                    tmp *= tmp;
                    k /= 2;
                }
                return c;
            }
            matrix(){
                vc.clear();
            }
            matrix(int n):n(n){
                vc = vector<vector<T>>(n, vector<T>(n, 0));
            }
    };
}
using namespace MATRIX;
