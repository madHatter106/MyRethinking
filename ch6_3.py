"""
THE HAUNTED DAG: Sometimes unmeasured causes can induce collider bias.
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as ar
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel
from cmocean.cm import balance_r
from seaborn import heatmap

"""
Example: infer direct influence of both parents (P) and grand parents (G) on the
educational achievement of children (C).
"""
dag_ed1 = CausalGraphicalModel(nodes=['P', 'G' ,'C'],
                               edges=[('G', 'P'), ('G', 'C'), ('P', 'C')])
dag_ed1.draw()

"""
But we suppose ther are unmeasured, common influences on parents and their
children (e.g. neighborhoods, not shared by grandparent who live elsewhere).
"""

dag_ed2 = CausalGraphicalModel(nodes=['G', 'P', 'C', 'U'],
                               edges=[('G', 'P'), ('U', 'P'),
                                      ('G', 'C'), ('P', 'C'), ('U', 'C')])
dag_ed2.draw()

"""
The DAG above implies that:
(1) P is some function of G and U
(2) C is some function of G, P, and U
(3) G and U are not functions of any other known variables.

Note that below U is made binary only to make the example easier to
understand. The concept doesn't depend on this assumption.
"""

n = 200 # numbrer of grandparent-parent-child triads
b_GP = 1 # direct effect of G on P
b_GC = 0 # direct effect of G on C
b_PC = 1 # direct effect of P on C
b_U = 2 # direct effect of U on P and C

np.random.seed(1)
U = 2* np.random.binomial(n=1, p=0.5, size=n)-1
G = np.random.normal(size=n)
P = np.random.normal(loc=b_GP*G + b_U*U)
C = np.random.normal(b_PC*P + b_GC*G + b_U*U)
d_ = pd.DataFrame(dict(C=C, P=P, G=G, U=U))
d_.head()
d_.U.value_counts()
dcorr = d_.corr()
dcorr.head()
_, ax = pl.subplots()
heatmap(dcorr.iloc[1:, :-1], mask=np.triu(np.ones([dcorr.shape[0]-1]*2), k=1),
        ax=ax, vmin=-1, vmax=1, cmap=balance_r, annot=True)
ax.set_facecolor('black')


with pm.Model() as m11:
    σ = pm.Exponential('σ', 1)
    β_PC = pm.Normal('β_PC', 0, 1)
    β_GC = pm.Normal('β_GC', 0, 1)
    α = pm.Normal('α', 0, 1)
    μ = α + β_PC * d_.P.values + β_GC * d_.G.values
    C = pm.Normal('C', mu=μ, sd=σ, observed=d_.C.values)

pm.model_to_graphviz(m11)
with m11:
    trc11 = pm.sample(1000, tune=1500)
pm.summary(trc11, alpha=0.11)
"""
The unmeasured U makes P a collider, and conditioning on P produces collider
bias. Assuming we can measure U, below is the multivariate regression that also
conditions on U:
"""
with pm.Model() as m12:
    σ = pm.Exponential('σ', 1)
    β_PC = pm.Normal('β_PC', 0, 1)
    β_GC = pm.Normal('β_GC', 0, 1)
    β_U = pm.Normal('β_U', 0, 1)
    α = pm.Normal('α', 0, 1)
    μ = α + β_PC * d_.P.values + β_GC * d_.G.values + β_U * d_.U.values
    C = pm.Normal('C', mu=μ, sd=σ, observed=d_.C.values)
    trc_12 = pm.sample(1000, tune=1500)

pm.summary(trc_12, alpha=0.11)
"""
The above is an example of Simpson's Paradox. Controlling for one predictor,
P in this case, can reverse the direction of association between some other
predictor (G) and the outcome (C).
Usually Simpson's paradox is presented in cases where adding the new predictor
helps us. The opposite happens here. 
"""
