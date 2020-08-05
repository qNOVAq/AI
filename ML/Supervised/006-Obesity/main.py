import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#filling missing header names in the dataset
heads = ['number', 'country', 'year', 'obesity', 'sex']
data = pd.read_csv('obesity-cleaned.csv', names = heads, header=0)

#dropping useless columns
data = data.drop(data.iloc[:, 0:1], axis=1)

#x = data.drop(data.iloc[:, 2:3], axis=1).values
#y = data.iloc[:, 2:3].values

#there are no null values

#encoding objects in diffrent ''
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), ['country', 'obesity', 'sex'])], remainder='passthrough')
encoded_Data  = ct.fit_transform(data[['country', 'obesity','sex']] )

data = data.drop(['country', 'obesity', 'sex'], axis=1)
data = pd.concat([data, encoded_Data])


