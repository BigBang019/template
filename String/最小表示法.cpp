/*把一个长为len的字符串围成一个圈，然后以任意一个字符作为起点，都会产生一个新的长为len的字符串，
字符串的最小表示就是所有新字符串中字典序最小的那个。下面这个函数就是解决这个问题的，返回值为字典序最小的串的在原串中的起始位置。
基本想法就是两个位置的字符比较，如果s[i+k] > s[j+k]那么i到i+k位置都不是最小表示的位置，所以i直接跳k+1步，反之j直接跳k+1步。*/
int get_min(char *s)
{
    int len = strlen(s) / 2;            //当前s为扩展过的s
    int i = 0, j = 1;
    while (i < len && j < len)
    {
        int k = 0;
        while (s[i + k] == s[j + k] && k < len)
            k++;
        if (k == len)
            break;
        if (s[i + k] < s[j + k])
        {
            if (j + k > i)
                j += k + 1;
            else
                j = i + 1;
        }
        else
        {
            if (i + k > j)
                i += k + 1;
            else
                i = j + 1;
        }
    }
    return min(i, j);
}
int get_max(char *s)
{
    int len = strlen(s) / 2;            //当前s为扩展过的s
    int i = 0, j = 1;
    while (i < len && j < len)
    {
        int k = 0;
        while (s[i + k] == s[j + k])
            k++;
        if (k == len)
            break;
        if (s[i + k] > s[j + k])
        {
            if (j + k > i)
                j += k + 1;
            else
                j = i + 1;
        }
        else
        {
            if (i + k > j)
                i += k + 1;
            else
                i = j + 1;
        }
    }
    return min(i, j);
}
