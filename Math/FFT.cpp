void FFT(Complex a[],int type){
	for (int i = 0; i < length;i++)
	{
		if (i<rev[i])
			swap(a[i], a[rev[i]]);
	}
	for (int mid = 1; mid < length;mid<<=1)
	{
		Complex W = Complex(cos(PI/mid),(double)type*sin(PI/mid));
		for (int l = 0, g = mid << 1; l < length;l+=g)
		{
			Complex w = Complex(1.0, 0.0);
			for (int k = 0; k < mid; k++, w = w * W)
			{
				Complex x = a[l + k], y = w * a[l + mid + k];
				a[l + k] = x + y;
				a[l + mid + k] = x - y;
			}
		}
	}
}
