import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, GridSearchCV

from mock_up_ds.preprocessing import *
from mock_up_ds.constants import *
from mock_up_ds.utils import BaselineClassifierTitanic


# baseline_test_score = np.mean(x_test[SEX_MALE] != y_test)
# print("Baseline test accuracy: {:.3f}".format(baseline_test_score))


models = {
          "Baseline": BaselineClassifierTitanic(),
          "Linear Regression": LogisticRegression(),
          "Ridge Regression": RidgeClassifier(),
          "K Neighbors": KNeighborsClassifier(),
          "Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gradient Boosting": GradientBoostingClassifier(),
          "Neural Network": MLPClassifier(),
         }

train_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, PCT_STANDARD_DEVIATION])
test_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, PCT_STANDARD_DEVIATION])


for model_name, model in models.items():

    cv = cross_validate(model, x, y, cv=10)

    avg_train_accuracy = np.mean(cv[TRAIN_SCORE])
    std_train_accuracy = np.std(cv[TRAIN_SCORE])
    pct_std_train_accu = std_train_accuracy / avg_train_accuracy * 100
    train_accuracies.loc[model_name] = [avg_train_accuracy, pct_std_train_accu]

    avg_test_accuracy = np.mean(cv[TEST_SCORE])
    std_test_accuracy = np.std(cv[TEST_SCORE])
    pct_std_test_accu = std_test_accuracy / avg_test_accuracy * 100
    test_accuracies.loc[model_name] = [avg_test_accuracy, pct_std_test_accu]

print("Train accuracies:\n", train_accuracies.astype("float").round(2))
print("Test accuracies:\n", test_accuracies.astype("float").round(2))


parameters_gb = {"loss": ["deviance", "exponential"], "learning_rate": [0.1, 0.001, 10], "n_estimators": [10, 100, 1000]}
grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters_gb, cv=5)

grid_search_gb.fit(x, y)

columns_to_show_gb = [
                      PARAM + "_" + LOSS,
                      PARAM + "_" + LEARNING_RATE,
                      PARAM + "_" + N_ESTIMATORS,
                      MEAN_TEST_SCORE,
                      # STD_TEST_SCORE,
                      RANK_TEST_SCORE,
                      ]
print(pd.DataFrame(grid_search_gb.cv_results_)[columns_to_show_gb])


parameters_rf = {CRITERION: ["gini", "entropy"], N_ESTIMATORS: [10, 100, 1000]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=parameters_rf, cv=5)

grid_search_rf.fit(x, y)

columns_to_show_rf = [
                      PARAM + "_" + CRITERION,
                      PARAM + "_" + N_ESTIMATORS,
                      MEAN_TEST_SCORE,
                      # STD_TEST_SCORE,
                      RANK_TEST_SCORE,
                      ]
print(pd.DataFrame(grid_search_rf.cv_results_)[columns_to_show_rf])