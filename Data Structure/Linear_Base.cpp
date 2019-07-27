// #pragma GCC optimize(2)
#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define sc second
#define fi first
using namespace std;
typedef long long ll;
typedef unsigned int uint;
typedef pair<int, int> pi;
const int N = 5e4 + 5;
const int B = 31;
struct Base{
    uint b[B + 3];
    Base() { init(); }
    uint &operator [] (uint i) {
        return b[i];
    }
    void init() { memset(b, 0, sizeof(b)); }
    void update(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                b[i] = x;
                break;
            }
            x ^= b[i];
        }
    }
    bool check(uint x){
        for (int i = B; ~i && x; i--) if (x&(uint(1)<<i)){
            if (!b[i]){
                break;
            }
            x ^= b[i];
        }
        return (x == 0);
    }
    static Base Merge(Base a, Base b){
        Base c, tmp_b, tmp_k;
        for (int i = 0; i <= B; i++)
        {
            tmp_b[i] = tmp_k[i] = 0;
            if (a[i]) tmp_b[i] = a[i];
        }
        for (int i = 0; i <= B; i++) c[i] = 0;
        for (int i = 0; i <= B; i++) if (b[i])
        {
            int ok = 1;
            uint v = b[i], k = 0;
            for (int j = B; ~j; j--)
            {
                if (v & (uint(1) << j)){
                    if (tmp_b[j]){
                        v ^= tmp_b[j];
                        k ^= tmp_k[j];
                    }else{
                        tmp_b[j] = v;
                        tmp_k[j] = (uint(1) << i) ^ k;
                        ok = 0;
                        break;
                    }
                }
            }
            if (ok){
                uint v = b[i];
                for (int j = 0; j <= B; j++)
                {
                    if (k & (uint(1) << j))
                    {
                        v ^= b[j];
                    }
                }
                for (int j = B; ~j; j--)
                {
                    if (v & (uint(1) << j)){
                        if (c[j]){
                            v ^= c[j];
                        }else{
                            c[j] = v;
                            break;
                        }
                    }
                }
            }
        }
        return c;
    }
};
