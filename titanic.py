import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, GridSearchCV

DATA_PATH = "./data"
DATA_FILENAME = "train.csv"
DATA_FILEPATH = os.path.join(DATA_PATH, DATA_FILENAME)

data = pd.read_csv(DATA_FILEPATH)

# print(data.describe())

# plt.figure()
# sns.heatmap(data.isnull())
# plt.show()

CABIN = "Cabin"
EMBARKED = "Embarked"
AGE = "Age"
SEX = "Sex"
SEX_MALE = "Sex_male"
TICKET = "Ticket"
PASSENGER_ID = "PassengerId"
NAME = "Name"
SURVIVED = "Survived"
TEST_SCORE = "test_score"
TRAIN_SCORE = "train_score"
AVERAGE = "Average"
STANDARD_DEVIATION = "% Standard Deviation"

data.drop([CABIN, PASSENGER_ID, TICKET, NAME], axis=1, inplace=True)

data.dropna(axis=0, subset=[EMBARKED], inplace=True)

data[AGE].fillna(data[AGE].mean(), inplace=True)

data = pd.get_dummies(data, columns=[EMBARKED, SEX], drop_first=True)

y = data[SURVIVED]

x = data.drop([SURVIVED], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)


baseline_test_score = np.mean(x_test[SEX_MALE] != y_test)
print("Baseline test accuracy: {:.3f}".format(baseline_test_score))

models = {
          "Linear Regression": LogisticRegression(),
          "Ridge Regression": RidgeClassifier(),
          "K Neighbors": KNeighborsClassifier(),
          "Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gradient Boosting": GradientBoostingClassifier(),
          "Neural Network": MLPClassifier(),
         }

train_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, STANDARD_DEVIATION])
test_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, STANDARD_DEVIATION])


for model_name, model in models.items():

    model.fit(x_train, y_train)
    cv = cross_validate(model, x, y, cv=10)

    avg_train_accuracy = np.mean(cv[TRAIN_SCORE])
    std_train_accuracy = np.std(cv[TRAIN_SCORE])
    train_accuracies.loc[model_name] = [avg_train_accuracy, std_train_accuracy / avg_train_accuracy * 100]

    avg_test_accuracy = np.mean(cv[TEST_SCORE])
    std_test_accuracy = np.std(cv[TEST_SCORE])
    test_accuracies.loc[model_name] = [avg_test_accuracy, std_test_accuracy / avg_test_accuracy * 100]

print("Train accuracies:\n", train_accuracies.astype("float").round(2))
print("Test accuracies:\n", test_accuracies.astype("float").round(2))
