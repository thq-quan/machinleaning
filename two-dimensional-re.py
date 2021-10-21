import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import category_encoders as ce

X = np.arange(20).reshape(-1, 2)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

svc = SVC( kernel='linear')
logre = LogisticRegression()
clf = DecisionTreeClassifier(criterion='entropy', max_depth= None, random_state=0)
pct = Perceptron()

svc.fit(X,y)
logre.fit(X,y)
clf.fit(X,y)
pct.fit(X,y)

y_svc = svc.predict(X)
y_logre = logre.predict(X)
y_clf = clf.predict(X)
y_pct = pct.predict(X)

print('Độ chính xác của svc:',accuracy_score(y, y_svc)*100)
print('Độ chính xác của LogisticRegression:',accuracy_score(y, y_logre)*100)
print('Độ chính xác của DecisionTreeClassifier:',accuracy_score(y, y_clf)*100)
print('Độ chính xác của Perceptron:',accuracy_score(y, y_pct)*100)