import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from constants import *
from preprocessing import *


# descriptive statistics for numerical variables
print(raw_data.describe())

# Information about the DF columns. What we care most are missing values
print(raw_data.info())

# Visualize missing data
plt.figure()
sns.heatmap(raw_data.isna())
plt.show()

# Separate numerical from non numerical variables

train_data_num = raw_data.select_dtypes(include=[np.number])
train_data_non_num = raw_data.select_dtypes(exclude=[np.number])

# Separate continuous from categoriacal variables
# Visualize distributions of continuous variables
train_data_cont = raw_data[[AGE, FARE]]

plt.figure()
sns.kdeplot(train_data_cont[AGE])
plt.show()

plt.figure()
sns.kdeplot(train_data_cont[FARE])
plt.show()

plt.figure()
sns.jointplot(data=train_data_cont, x=AGE, y=FARE, kind="kde")
plt.show()

plt.figure()
sns.heatmap(raw_data.corr(), annot=True)
plt.show()
