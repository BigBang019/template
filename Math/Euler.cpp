using namespace std;
typedef long long ll;
const int N = 1e6 + 10;
int phi[N], prime[N];
int pointer;
void Euler()
{
    phi[1] = 1;
    for (int i = 2; i < N; i++)
    {
        if (!phi[i])
        {
            phi[i] = i - 1;
            prime[pointer++] = i;
        }
        for (int j = 0; j < pointer && i * prime[j] < N; j++)
        {
            phi[i * prime[j]] = phi[i] * (prime[j] - 1);
            if (i % prime[j] == 0)
            {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
        }
    }
}
