#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[ ]:


# ====================== Metadata ======================


# In[ ]:


__author__ = 'Till Kauffeldt'
__version__ = '1.0'
__github__ = 'https://github.com/ProfKauf/Modules/'


# In[ ]:


# ====================== Libraries ======================


# In[ ]:


'''Basics'''
import dill
import os
import pickle
import numpy as np
import math
import pandas as pd 
import warnings
import itertools
warnings.filterwarnings("ignore")
import importlib
import random
from datetime import datetime
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3d
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from itertools import product
'''Statistics'''
import statsmodels.stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import statsmodels.stats.weightstats as ws
import pingouin as pg
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats, spatial
from scipy.stats import ttest_ind, ttest_ind_from_stats, uniform, norm, expon, truncnorm, chi2, t, bernoulli, binom, poisson, loguniform, randint, logser
from scipy.special import stdtr
from scikit_posthocs import outliers_tietjen
from yellowbrick.regressor import CooksDistance
'''Experimentals'''
#from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
'''Preprocessing'''
#from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, tree
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler, OneHotEncoder
#from sklearn.impute import IterativeImputer
from sklearn.utils import resample
#from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as Pipeline_imb
from category_encoders.ordinal import OrdinalEncoder as ordenc
'''Classifiers'''
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
#from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, IsolationForest, ExtraTreesRegressor, RandomForestRegressor
#from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, cross_validate, RandomizedSearchCV, HalvingGridSearchCV
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix, make_scorer, jaccard_score, balanced_accuracy_score, accuracy_score, f1_score,precision_score,recall_score
'''Hyperparameter Tuning'''
#from skopt.space import Real, Categorical, Integer
#from skopt import BayesSearchCV

