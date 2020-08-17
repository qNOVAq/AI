import pandas as pd
import numpy as np
data = pd.read_csv('housing.csv')


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(np.nan, 'mean')
data.iloc[:, 4:5] = imputer.fit_transform(data.iloc[:, 4:5])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])


x = data.drop('median_house_value', axis=1)
y = data.iloc[:, -2]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

pipe.fit(X_train, Y_train)

score = pipe.predict(X_test)

from sklearn.metrics import accuracy_score

cm = accuracy_score(score, Y_test)


print(cm)

