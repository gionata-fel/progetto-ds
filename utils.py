from sklearn.base import ClassifierMixin, BaseEstimator

from mock_up_ds.constants import *


class BaselineClassifierTitanic(BaseEstimator, ClassifierMixin):

    def __init__(self, column_to_use_as_predictor=SEX_MALE):

        self.column_to_use_as_predictor = column_to_use_as_predictor

    def fit(self, X, y):

        return self

    def predict(self, X):
        return 1 - X[self.column_to_use_as_predictor]
