/****************************
  矩阵快速幂slower_but_convenient
  数组版本下翻
*****************************/
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

/****************************
  矩阵快速幂run_faster
  需要调整方阵大小B
*****************************/
namespace MATRIX{
    using namespace std;
    typedef unsigned long long ull;
    typedef long long ll;
    const int B = 2;///change
    ll MOD;
      
    inline ll add(ll a, ll b) { a += b; return (a >= MOD ? a - MOD : a);}
  
    inline ll mul(ll a, ll b) { return a * b % MOD; }
  
    template<typename T>
    class matrix{
        public:
            int a[B][B];
            int* operator [] (ull i) {
                return a[i];
            }
            matrix<T> operator * (matrix<T> & b) const {
                matrix<T> c;
                for (int i = 0; i < B;i++)
                {
                    for (int j = 0; j < B;j++)
                    {
                        for (int k = 0; k < B;k++)
                        {
                            c[i][j] = add(c[i][j], mul(a[i][k], b[k][j]));
                        }
                    }
                }
                return c;
            }
            matrix<T> &operator *= (matrix<T> &b) {
                return *this = *this * b;
            }
            matrix<T> operator ^ (ull k) const {
                matrix<T> c;
                matrix<T> tmp = *this;
                for (int i = 0; i < B;i++) c[i][i] = 1;
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
                for (int i = 0; i < B;i++) for (int j = 0; j < B;j++) a[i][j] = 0;
            }
    };
}
using namespace MATRIX;
