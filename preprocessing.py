import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from mock_up_ds.constants import *

DATA_PATH = "./data"
DATA_FILENAME = "train.csv"
DATA_FILEPATH = os.path.join(DATA_PATH, DATA_FILENAME)

raw_data = pd.read_csv(DATA_FILEPATH)
data = raw_data.copy()

if __name__ == "main":
    print(data.describe())

    plt.figure()
    sns.heatmap(data.isnull())
    plt.show()


data.drop([CABIN, PASSENGER_ID, TICKET, NAME], axis=1, inplace=True)

data.dropna(axis=0, subset=[EMBARKED], inplace=True)

data[AGE].fillna(data[AGE].mean(), inplace=True)

data = pd.get_dummies(data, columns=[EMBARKED, SEX], drop_first=True)

y = data[SURVIVED]

x = data.drop([SURVIVED], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)
