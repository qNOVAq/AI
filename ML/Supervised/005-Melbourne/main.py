import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv('Melbourne_housing_FULL.csv')

#dropping useless columns
#making a drop function to use multiple times

#dropping subrub
data = data.drop(data.iloc[:, 0:1], axis=1)
#dropping address
data = data.drop(data.iloc[:, 0:1], axis=1)
#dropping method
data = data.drop(data.iloc[:,3:4 ], axis=1)
#dropping sellerG
data = data.drop(data.iloc[:, 3:4], axis = 1)
#dropping date
data = data.drop(data.iloc[:, 3:4], axis=1)
#dropping
data = data.drop(data.iloc[:, 3:4], axis=1)
#dropping postcode
data = data.drop(data.iloc[:, 3:4], axis=1)
#dropping council area
data = data.drop(data.iloc[:, 9:10], axis=1)
#dropping reagion name
data = data.drop(data.iloc[:, 11:12], axis=1)
#dropping property count
data = data.iloc[:, 0:11]


#encoding objects
encoder = LabelEncoder()
data.iloc[:, 1] = encoder.fit_transform(data.iloc[:, 1])

#filling missing data
imputer = SimpleImputer(np.nan, 'mean')
data.iloc[:, 2:] = imputer.fit_transform(data.iloc[:, 2:])

#split to x and y
x = data.drop(data.iloc[:, 2:3], axis=1)
y = data.iloc[:, 2:3]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
 
#scaling data 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#fitting to linear regression
reg= LinearRegression()
reg.fit(x_train, y_train)

train_acc = reg.score(x_train, y_train)
test_acc = reg.score(x_test, y_test)


print(f'Train Accuracy: {train_acc}' )
print(f'Test Accuracy: {test_acc}' )



