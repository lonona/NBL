import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(context="notebook", 
	palette="Spectral", 
	style = 'darkgrid', 
	font_scale = 1.5, 
	color_codes=True)
from sklearn.utils import resample
from scipy.stats import skew, norm, kurtosis, mstats


def bootstrap_outlier_removal(df, colname, n = 300, niter=9, p_rnge=[0.05, 0.95]):
	"""
		The function generates a bootstrap technique for 
		outlier removal and returns the lower and upper
		percentiles.
	"""
	data = ndata['{}'.format(colname)].values
	boot = []
	mean = []
	percentile_rnge = []

	K = niter
	for k in range(K):
	    

	    a = resample(data, replace=True, n_samples=n, random_state=128)
	    boot.append(a)
	    mean.append(a.mean())

	    pL, Q1, Q2, Q3, pH = mstats.mquantiles(a, prob=[p_rnge[0], 0.25, 0.5, 0.75, p_rnge[-1]])
	    
	    # out of bag observations
	    oob = [x for x in data if x not in a]
	    IQR = Q3 - Q1
	    
	    # 
	    outlier = (a < (Q1 - 1.5 * IQR)) |(a > (Q3 + 1.5 * IQR))
	    # print(outlier)
	    a_out = a[~outlier]
	    data = np.array(oob + list(a_out))
	    percentile_rnge.append([pL, pH])

	plt.figure(figsize=(10,5), dpi =100)
	plt.boxplot(boot, vert=True)
	plt.plot(range(1, K + 1), mean, 'bo')
	plt.ylim(-5, None)
	plt.xlabel(r"iteration steps")
	plt.ylabel(r'$\rm NO_{3}-N\,$(mg/L)')


	return np.array(percentile_rnge)
