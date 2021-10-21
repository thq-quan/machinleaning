import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\hk1-gd1-4\mechineleaning\weather.csv")
X = data.drop(['play'], axis=1)
y = data['play']
encoder = ce.OrdinalEncoder(cols=['outlook','temperature','humidity','wind'])
X_pred = encoder.fit_transform(X)
logreg = LogisticRegression()
logreg.fit(X_pred,y)
y_pred=logreg.predict(X_pred)
print(y_pred)
print(accuracy_score(y, y_pred)*100)