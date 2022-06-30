import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TenMultiplier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * 10
    
    
    
multiplier = TenMultiplier()

X = np.array([6, 3, 7, 4, 7])
multiplier.transform(X)

multiplier = TenMultiplier()

X = np.array([6, 3, 7, 4, 7])
multiplier.transform(X)

// outut array([60, 30, 70, 40, 70])


""" Remember, all estimators have a fit method, and since this is a transformer, it also has a transform method.

FIT METHOD: This takes in a 2d array X for the feature data and a 1d array y for the target labels. Inside the fit method, we simply return self. This allows us to chain methods together, since the result on calling fit on the transformer is still the transformer object. This method is required to be compatible with scikit-learn.
TRANSFORM METHOD: The transform function is where we include the code that well, transforms the data. In this case, we return the data in X multiplied by 10. This transform method also takes a 2d array X.  """ 