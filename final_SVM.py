import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# df = pd.read_csv("Dataset/pose_angles.csv")
df = pd.read_csv("Dataset/pose_angles_3.csv")

df = df.drop('num', axis = 1)
df.head()

# get X, y
X = df.values[:, :-1]
y = df.values[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=1, gamma=0.01, C=0.001)
svm.fit(X_train, y_train)

import joblib
filename = '221203SVM.pkl'
joblib.dump(svm, filename)