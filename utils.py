import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from keras.layers import Dense, Input, Add, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from constants import *


class BaselineClassifierTitanic(BaseEstimator, ClassifierMixin):

    def __init__(self, column_to_use_as_predictor=SEX_MALE):

        self.column_to_use_as_predictor = column_to_use_as_predictor

    def fit(self, X, y):

        return self

    def predict(self, X):
        return 1 - X[self.column_to_use_as_predictor]


class TitanicNNClassifier(BaseEstimator, ClassifierMixin):

    @staticmethod
    def _create_model(input_len):

        k_in = Input(shape=(input_len, ))

        dense_1_1 = Dense(units=input_len)(k_in)
        add_1_1 = Add()([k_in, dense_1_1])
        dense_2_1 = Dense(units=input_len)(add_1_1)
        add_2_1 = Add()([dense_1_1, dense_2_1])
        dense_3_1 = Dense(units=5)(add_2_1)

        dense_1_2 = Dense(units=input_len)(k_in)
        add_1_2 = Add()([k_in, dense_1_2])
        dense_2_2 = Dense(units=input_len)(add_1_2)
        add_2_2 = Add()([dense_1_2, dense_2_2])
        dense_3_2 = Dense(units=5)(add_2_2)

        concat_4 = Add()([dense_3_1, dense_3_2])
        dense_4 = Dense(5)(concat_4)

        dense_5 = Dense(1, activation="sigmoid")(dense_4)

        model = Model(inputs=k_in, outputs=dense_5)

        return model

    def __init__(self):
        self.input_len = None
        self.optimizer = None
        self.loss = None

    def compile(self, optimizer=Adam(), loss="mse"):

        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, X, y, batch_size=None, epochs=10, verbose=0, **kwargs):

        self.input_len = np.shape(X)[1]
        self.model = self._create_model(self.input_len)
        self.compile()

        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def predict(self, X):

        predictions = self.model.predict(X)
        return predictions > 0.5
