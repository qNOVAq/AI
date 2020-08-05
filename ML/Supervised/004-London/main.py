import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
data = pd.read_csv('housing_in_london_yearly_variables.csv')
data = data.drop('code', axis =1)
data = data.drop('date', axis =1)

#filling salary column
imputer = SimpleImputer(np.nan, 'mean')
imputer.fit(data.iloc[:, 1:2])
data.iloc[:, 1:2] = imputer.transform(data.iloc[:, 1:2])

#filling life_satisfactory column
imputer.fit(data.iloc[:, 2:3])
data.iloc[:, 2:3] = imputer.transform(data.iloc[:, 2:3])

#filling recycling_pct
imputer.fit(data.iloc[:, 4:5])
data.iloc[:, 4:5] = imputer.transform(data.iloc[:, 4:5])

#filling population_sizes
imputer.fit(data.iloc[:, 5:6])
data.iloc[:, 5:6] = imputer.transform(data.iloc[:, 5:6])

#filling number_of_jobs
imputer.fit(data.iloc[:, 6:7])
data.iloc[:, 6:7] = imputer.transform(data.iloc[:, 6:7])

#filling area size
imputer.fit(data.iloc[:, 7:8])
data.iloc[:, 7:8] = imputer.transform(data.iloc[:, 7:8])

#filling no_of_houses
imputer.fit(data.iloc[:, 8:9])
data.iloc[:, 8:9] = imputer.transform(data.iloc[:, 8:9])

#encoding
ct = ColumnTransformer(transformers=['encoder', OneHotEncoder, data.iloc[:, 0:1]], remainder='passthrough')
ct.fit(data.iloc[:, 0])
data.iloc[:, 0] = ct.transform(data.iloc[:, 0])