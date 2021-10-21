import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\hk1-gd1-4\mechineleaning\weather.csv")
X = data.drop(['play'], axis=1)
y = data['play']
encoder = ce.OrdinalEncoder(cols=['outlook','temperature','humidity','wind'])
X_pred = encoder.fit_transform(X)

svc = SVC( kernel='linear')
logre = LogisticRegression()
clf = DecisionTreeClassifier(criterion='entropy', max_depth= None, random_state=0)
pct = Perceptron()

svc.fit(X_pred,y)
logre.fit(X_pred,y)
clf.fit(X_pred,y)
pct.fit(X_pred,y)

y_svc = svc.predict(X_pred)
y_logre = logre.predict(X_pred)
y_clf = clf.predict(X_pred)
y_pct = pct.predict(X_pred)

print('Độ chính xác của svc:',accuracy_score(y, y_svc)*100)
print('Độ chính xác của LogisticRegression:',accuracy_score(y, y_logre)*100)
print('Độ chính xác của DecisionTreeClassifier:',accuracy_score(y, y_clf)*100)
print('Độ chính xác của Perceptron:',accuracy_score(y, y_pct)*100)