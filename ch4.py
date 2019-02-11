import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy.stats import norm, uniform
from cmocean import cm
from seaborn import kdeplot
#import arviz as az

% matplotlib inline

d = pd.read_csv('rethinking-Experimental/data/Howell1.csv',
                delimiter=';')

d2 = d.loc[d.age >= 18]
mu = np.linspace(140, 160, num=200)
sigma = np.linspace(4, 9, num=200)

post = pd.DataFrame([[x, y] for y in sigma for x in mu],
                    columns=['mu', 'sigma']
                    )
post['LL'] = np.sum(norm.logpdf(d2.height.values.reshape(1, -1),
                                loc=post.mu.values.reshape(-1, 1),
                                scale=post.sigma.values.reshape(-1, 1)),
                    axis=1)
post['logprod'] = post.LL + norm.logpdf(post.mu, 178, 20) +\
                        uniform.logpdf(post.sigma, 0, 50)
post['prob'] = np.exp(post.logprod - post.logprod.max())
post.head()
ax=post.plot.hexbin(x='mu', y='sigma', C='prob', cmap=cm.thermal)
ax.set_xlabel('mu')
f = pl.gcf()

samples = post[['mu', 'sigma', 'prob']].sample(n=10000, replace=True, weights=post.prob)
samples.plot.scatter(x='mu', y='sigma', alpha=0.1)
samples.plot.hexbin(x='mu', y='sigma', C='prob', cmap=cm.thermal);
samples.mu.plot.density()
samples.sigma.plot.density(color='orange')

d3 = d2.height.sample(n=20)

mu = np.linspace(150, 170, num=200)
sigma = np.linspace(4, 20, num=200)
post2 = pd.DataFrame([[x, y] for y in sigma for x in mu],
                     columns=['mu', 'sigma'])
post2['LL'] = np.sum(norm.logpdf(d3.values.reshape(1,-1), loc=post2.mu.values.reshape(-1,1),
                                 scale=post2.sigma.values.reshape(-1, 1)), axis=1)
post2['prod'] = post2.LL + norm.logpdf(post2.mu, 178, 20) + uniform.logpdf(post.sigma, 0, 50)
post2['prob'] = np.exp(post2['prod'] - post2['prod'].max())
sample2 = post2.sample(n=10000, replace=True, weights=post2.prob)
sample2.head()
sample2.plot.hexbin(x='mu', y='sigma', C='prob', cmap=cm.thermal)
sample2.sigma.plot.density()
