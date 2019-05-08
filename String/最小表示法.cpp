/*把一个长为len的字符串围成一个圈，然后以任意一个字符作为起点，都会产生一个新的长为len的字符串，
字符串的最小表示就是所有新字符串中字典序最小的那个。下面这个函数就是解决这个问题的，返回值为字典序最小的串的在原串中的起始位置。
基本想法就是两个位置的字符比较，如果s[i+k] > s[j+k]那么i到i+k位置都不是最小表示的位置，所以i直接跳k+1步，反之j直接跳k+1步。*/
int getMin() {
    int i = 0, j = 1;
    int l;
    while (i < len && j < len) {
        for (l = 0; l < len; l++)
            if (str[(i + l) % len] != str[(j + l) % len]) break;
        if (l >= len) break;
        if (str[(i + l) % len] > str[(j + l) % len]) {
            if (i + l + 1 > j) i = i + l + 1;
            else i = j + 1;
        }
        else if (j + l + 1 > i) j = j + l + 1; 
        else j = i + 1;
    }
    return i < j ? i : j;
}

int getMax() {
    int i = 0, j = 1, k = 0;
    while (i < len && j < len && k < len) {
        int t = str[(i + k) % len] - str[(j + k) % len];
        if (!t) k++;
        else {
            if (t > 0) {
                if (j + k + 1 > i) j = j + k + 1;
                else j = i + 1;
            }
            else if (i + k + 1 > j) i = i + k + 1;
            else i = j + 1;
            k = 0;
        }
    }
    return i < j ? i : j;

}
