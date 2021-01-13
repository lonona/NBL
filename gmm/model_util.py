#!/usr/bin/env python

#<<Load Python modules>>
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
from scipy.stats import skew,norm
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture
np.random.seed(seed=458)



def univariate_distplot(data, logscale=True, fsize=30, add_kde=True, label="NO3 (mg/l)"):
	"""
	This function displays
	Parameters
	----------
	data : 1-D array.
	logscale : boolean
			generates a log transform of the data
	fsize : integer
			fontsize
	add_kde : boolean
			add kernel density estimation plot
	label : str
			data label

	"""
	
	if logscale is  True:
		data = np.log1p(data) 
	

	fig = plt.figure(figsize=(25,10), dpi=150)

	ax = fig.add_subplot(121)

	ax1 = sns.distplot(data,fit=None, 
	                   kde=add_kde,
	                   color = 'darkblue',
	                   norm_hist=True, 
	                  kde_kws={'linestyle':'--', 'linewidth':5})

	# compute mean (mu) & standard deviation (sigma) of the data
	(mu, sigma) = norm.fit(data)

	ax1.set_title(r'$\mu = %0.2f, \sigma = %0.2f$' %(mu, sigma), fontsize=fsize)
	ax1.set_ylabel(r'Frequency distribution', fontsize=fsize)
	ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2g'))
	ax1.set_xlabel(r'{}'.format(label),fontsize=fsize)
	plt.xticks(fontsize=fsize, rotation=None)
	plt.yticks(fontsize=fsize, rotation=None)
	#
	x1, y1 = sns.distplot(data).get_lines()[0].get_data()
	ind = list([np.where(y1 == k)[0][0] for k in sorted(y1)[-3:]])
	px = np.log1p(x1[[ind]])
	ax1.plot(px, sorted(y1)[-3:], marker='o', linestyle='dashed', color='green', label="Peaks")

	# This part generates the PP plot
	ax = fig.add_subplot(122)

	res = stats.probplot(data, plot=plt, dist='norm', rvalue=True)
	ax.set_title(r'Probplot of Gaussian Distribution', fontsize=fsize)
	ax.set_ylabel(r'Ordered Values', fontsize=fsize)
	ax.set_xlabel(r'Quantiles', fontsize=fsize)
	plt.xticks(fontsize=fsize, rotation=None)
	plt.yticks(fontsize=fsize, rotation=None)
	ax.get_children()[2].set_fontsize(fsize)
	plt.tight_layout()


def gmm_func(data, ncomponents=10, logscale=True):
	""" 
	** This function displays the best fit of the dataset 
	using the scikit-learn Gaussian Mixture model package. 
	** The open-source package employs an Expectation-Maximization 
	iterative approach to find the best mixture of Gaussians for the data.

	Parameters
	----------
	data : 1-D array.
	logscale : boolean
			generates a log transform of the data
	fsize : integer
			fontsize
	ncompnents : integer
			define the number of components to learn from.

	"""
	
	if logscale is  True:
		data = np.log1p(data)

	if data.ndim == 1:
		data = data[:, np.newaxis]


	# fit models with ncomponents
	N = np.arange(1, ncomponents)
	models = [None for i in range(len(N))]

	for i in range(len(N)):
	    models[i] = GaussianMixture(N[i]).fit(data)

	# compute the AIC and the BIC
	aic = [m.aic(data) for m in models]
	bic = [m.bic(data) for m in models]

	return aic, bic, models

	