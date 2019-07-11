const int maxn = 1700010;
int T, tot, prime[maxn], mu[maxn];
map<int, ll> ans_mu;

void sieve()
{
    fill(prime, prime + maxn, 1);
    mu[1] = 1, tot = 0;
    for (int i = 2; i < maxn; i++)
    {
        if (prime[i])
        {
            prime[++tot] = i, mu[i] = -1;
        }
        for (int j = 1; j <= tot && i * prime[j] < maxn; j++)
        {
            prime[i * prime[j]] = 0;
            if (i % prime[j] == 0)
            {
                mu[i * prime[j]] = 0;
                break;
            }
            else
            {
                mu[i * prime[j]] = -mu[i];
            }
        }
    }
    for (int i = 2; i < maxn; i++)
        mu[i] += mu[i - 1];
}

ll calc_mu(int x)
{
    if (x < maxn)
        return mu[x];
    if (ans_mu.count(x))
        return ans_mu[x];
    ll ans = 1;
    for (ll i = 2, j; i <= x; i = j + 1)
    {
        j = x / (x / i), ans -= (j - i + 1) * calc_mu(x / i);
    }
    return ans_mu[x] = ans;
}

ll calc_phi(int x)
{
    ll ans = 0;
    for (ll i = 1, j; i <= x; i = j + 1)
    {
        j = x / (x / i), ans += (x / i) * (x / i) * (calc_mu(j) - calc_mu(i - 1));
    }
    return ((ans - 1) >> 1) + 1;
}
