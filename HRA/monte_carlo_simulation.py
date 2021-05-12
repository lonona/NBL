'''
These Monte Carlo utility functions are produced
by G. Yeboah
Tuesday 11 May 2021
'''

# ++++++++++++++++++++++++++++++
# << Import Python libraries >>
# ++++++++++++++++++++++++++++++

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from pomegranate import *
from scipy.stats import truncnorm, shapiro, norm



np.random.seed(seed=4)
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = OrderedDict([
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])



# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def mixture_components(data, pdf, nsub_populations=2):

	prob = dict(normal = NormalDistribution,
		        exponential = ExponentialDistribution,
		        lognormal = LogNormalDistribution,
		        gamma = GammaDistribution
		        )


	if data.ndim == 1:
		data = data[:, np.newaxis]

	model = GeneralMixtureModel.from_samples(prob[pdf], nsub_populations, data, n_jobs=2)

	return model

# def monte_carlo_sim()

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



def compare_mc_sampling(mu_N, std_N, limits, sampSize = 1000, 
						weights=[1,1], bw_mu = 4.85, bw_std = 0.93, Ir_mu = 1.034, Ir_std = 0.83,
						fsize=[20,16], DPI=80, ci_color='orange'):
    
    plt.figure(figsize=fsize, dpi = DPI)
    nr, nc = [3,3]
    
    kwargs = dict(histtype='bar',
                  alpha=0.7, 
                  density=True, 
                  bins=40,
                  color='b',
                  edgecolor='black',
                  linewidth=1.2,
                  label=r'$\rm HQ \, distr.$')

    for i in range(3):
        
        # generate  sample
        if i != 2:
            
            if i == 0: w=weights[0]
                
            else: w=weights[-1]
            
            sample_no3 = w*get_truncated_normal(mean=mu_N[i], sd=std_N[i], low=limits[i][0], upp=limits[i][1]).rvs(sampSize)
        else:
            sample_no3 = get_truncated_normal(mean=mu_N[i], sd=std_N[i], low=limits[i][0], upp=limits[i][1]).rvs(sampSize)

        sample_bw = get_truncated_normal(mean=bw_mu, sd=bw_std, low=limits[i][0], upp=limits[i][1]).rvs(sampSize)

        sample_Ir = get_truncated_normal(mean=Ir_mu, sd=Ir_std, low=limits[i][0], upp=limits[i][1]).rvs(sampSize)
        # if i == 0:
        # 	print(np.max(sample_Ir/sample_bw))
        sample_HQ = sample_no3 * sample_Ir/(1.6*sample_bw)

        # if i == 0:
        # 	print(sample_bw)
        # stat, p = normaltest(sample_HQ)
    #     print('Size = %d; Statistics=%.3f, p=%.3f' % (sizes[i], stat, p))
    
        for v, c in enumerate([0.6827, 0.9545, 0.9973]):
            v += 1

            # plot histogram of sample
            cell = 3*i+v
            plt.subplot(nr, nc, cell)
            n, bins, patches = plt.hist(sample_HQ, **kwargs)
            # plt.title()
            stat, p = shapiro(sample_HQ)
            plt.title('Size = %d; Shapiro Test: Stats=%.3f, p=%.3f' % (sampSize, stat, p))
   
            # best fit of data
            (mu, sigma) = norm.fit(sample_HQ)

            # add a 'best fit' line
            xmin, xmax = plt.xlim()
            # print(sample_HQ.max())
            norm_x = np.linspace(xmin, xmax, len(bins))
            norm_y = norm.pdf(norm_x, mu, sigma)
            l = plt.plot(norm_x, norm_y, 'r--', linewidth=2, label='Normal fit')
            # print(n.max())

            # confident interval
            ci = norm(*norm.fit(sample_HQ)).interval(c)
            plt.fill_betweenx([0, n.max()], 
                              ci[0], ci[1], 
                              color=ci_color, 
                              alpha=0.5,
                              label=r'{:.2f} \% CI'.format(c*100) )  # Mark between 0 and the highest bar in the histogram
            sbins = bins[np.where(np.logical_and(bins >= ci[0], bins <= ci[-1]))]
            count = len(sbins[np.where(sbins > 1 )])

            plt.text(sbins.max()/2, norm_y.max(), r'risk=${:.2f}$'.format(count/100))
            # plt.xlim(limits[i][0], limits[i][1])
            plt.xlim(0)
            plt.legend(loc=0, framealpha=0.5)
            
            if cell == 1:
                plt.ylabel(r'Background MC distribution')
            elif cell == 4:
                plt.ylabel(r'human-induced MC distribution')
            elif cell == 7:
                plt.ylabel(r'single source MC distribution')


