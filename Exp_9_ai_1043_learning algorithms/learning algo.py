import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
Import required modules and packages
dataset = pd.read_csv('….\student_scores.csv')
dataset.head()
Import data set
Choose the right path for the dataset
dataset.describe() Descriptive statistics of the attributes
available in the dataset
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()
Visualize the data.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
Identify the independent (X) and
dependent variables (y) in the data set
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=0)
print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
Splitting the given data in to training set
(80%) and testing set (20%)
Beginners Level
Lab 11 - Implementation of Learning Algorithms for an Application
18CSC305J - ARTIFICIAL INTELLIGENCE Page 4
print('Y test shape: ', Y_test.shape)
regressor = LinearRegression()
Model instantiation
regressor.fit(X_train, y_train) Model Training
print(regressor.intercept_)
print(regressor.coef_)
Finding out the coefficient (a) and
intercept (b) value of linear model
(y=aX+b)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
Testing the model
print('Mean Absolute Error:',
metrics.mean_absolute_error (y_test, y_pred))
print('Mean Squared Error:',
metrics.mean_squared_error (y_test, y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
MAE, MSE, RMSE – Evaluation metrics
of Model
