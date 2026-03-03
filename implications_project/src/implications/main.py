import pandas as pd
import operator
from collections import Counter
from scipy.stats import binomtest
from itertools import product, combinations

#========
#global
#========
OPS = {
    "==": operator.eq,
    "<":  operator.lt,}
#========
#helpers
#========
def _unique_midpoints(values):
    """Return midpoints between sorted unique non-null values."""
    v = sorted(set(pd.Series(values).dropna().tolist()))
    return [(v[i] + v[i + 1]) / 2 for i in range(len(v) - 1)]

def _possible_values(data,col,num_cols):
    return _unique_midpoints(data[col]) if col in num_cols else list(pd.unique(data[col].dropna()))

def _wilson_interval(y, confidence=0.95, positive_label=None, method="wilson"):
    """Compute Wilson confidence intervals"""
    n= len(y)
    counts = Counter(y)
    classes = list(counts)

    # binary: one interval for chosen positive label
    if len(classes) == 2:
        if positive_label is None:
            positive_label = sorted(classes, key=lambda x: str(x))[1]
        k = counts[positive_label]
        ci = binomtest(k, n).proportion_ci(confidence_level=confidence, method=method)
        return {"n": n, "k": k, "ci": (ci.low, ci.high)}

    # multiclass: one-vs-rest for each class
    out = {}
    for cls, k in counts.items():
        ci = binomtest(k, n).proportion_ci(confidence_level=confidence, method=method)
        out[cls] = {"n": n, "k": k, "ci": (ci.low, ci.high)}
    return out[y.value_counts().idxmax()]

#========
#main
#========
def empirical_implications(data,conclusion,num_cols=[],confidence=0.95,n_premise=1):
    '''compute quasi-implications with confidence interval and support.
    Arguments:
        -data=pandas dataframe
        -conclusion=column name of the attribute that is being implied
        -num_cols=numeric columns
        -confidence=confidence level of intervals
        -n_premise=number of attributes in the premise
    '''
    #preliminaries
    data=data.copy().dropna()
    cat_cols=[x for x in data.columns if x not in num_cols and x!=conclusion]
    for col in cat_cols:
        data[col] = (
        data[col]
          .astype("string")
          .str.replace("\u00A0", " ", regex=False)  # NBSP -> normal space
          .str.replace(r"\s+", " ", regex=True)     # optional: collapse whitespace
          .str.strip())
    cols=cat_cols+num_cols
    #sanity checks

    #combinatorics
    cols_comb=list(combinations(cols, n_premise))
    #dic to store results
    dic={}
    for k in ['attribute(s)','value(s)','support','predicted class','ML estimate',f'{confidence} Wilson interval']:
        dic[k]=[]
    #fill dic
    for c in cols_comb:
        vals_per_col = [_possible_values(data, col, num_cols) for col in c]

        for v in product(*vals_per_col):  # v is a tuple, same length as c
            mask = pd.Series(True, index=data.index)

            for col, val in zip(c, v):
                op = operator.eq if col in cat_cols else operator.lt
                mask &= op(data[col], val)

            count = int(mask.sum())
            if count > 0:
                vc = data.loc[mask, conclusion].value_counts(normalize=True)
                pred = vc.idxmax()

                interval = _wilson_interval(
                    data.loc[mask, conclusion],
                    confidence=confidence,
                    positive_label=pred,
                    method="wilson",
                )

                dic["attribute(s)"].append(c)
                dic["value(s)"].append(v)
                dic["support"].append(count)
                dic["predicted class"].append(pred)
                dic["ML estimate"].append(vc.max())
                dic[f"{confidence} Wilson interval"].append(interval["ci"])

    return pd.DataFrame(dic)