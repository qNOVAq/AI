import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics import confusion_matrix

ds = pd.read_csv('housing_in_london_monthly_variables.csv')
ds.head()

# filling null values with their corresponding mean of columns
mean_of_hs = ds['houses_sold'].mean()        
mean_of_noc = ds['no_of_crimes'].mean()

ds = ds.fillna({'houses_sold' : mean_of_hs})
ds = ds.fillna({'no_of_crimes' : mean_of_noc})

from sklearn.preprocessing import LabelEncoder

area = LabelEncoder()
code = LabelEncoder()

ds['area_n'] = area.fit_transform(ds['area'])
ds['code_n'] = code.fit_transform(ds['code'])
ds.drop(['area','code','date'], axis = 1, inplace=True)

borough_flag_1 = ds[ds['borough_flag']==1]
borough_flag_0 = ds[ds['borough_flag']==0]

borough_flag_1 = borough_flag_1.sample(n=borough_flag_0.shape[0])

x = ds.drop('borough_flag', axis=1)
y = ds['borough_flag']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)
x_train.shape , x_test.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier





dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
score = dt.score(x_test,y_test)