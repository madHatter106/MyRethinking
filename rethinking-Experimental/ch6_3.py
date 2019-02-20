"""
THE HAUNTED DAG
"""



import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as ar
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel

np.random.seed(71)
n_plants = 100

#
