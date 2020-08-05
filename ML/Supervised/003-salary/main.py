import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('salary/Salary.csv')

X = data['YearsExperience'].values
y = data['Salary'].values


X = X.reshape(-1,1)
y = y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
reg = LinearRegression()
reg.fit(x_train, y_train)

train_acc = reg.score(x_train, y_train)
test_acc = reg.score(x_test, y_test)
print('Train Accuracy: ' + str(train_acc))
print('Test Accuracy:  ' + str(test_acc))