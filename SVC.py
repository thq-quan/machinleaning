import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\hk1-gd1-4\mechineleaning\weather.csv")
X = data.drop(['play'], axis=1)
y = data['play']
encoder = ce.OrdinalEncoder(cols=['outlook','temperature','humidity','wind'])
X_pred = encoder.fit_transform(X)
svc = SVC( kernel='linear')
svc.fit(X_pred,y)
w = svc.coef_
b = svc.intercept_
y_pred=svc.predict(X_pred)
print(y_pred)
print(w)
print(b)
print(accuracy_score(y, y_pred)*100)