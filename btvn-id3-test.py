import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\hk1-gd1-4\mechineleaning\weather.csv")
X = data.drop(['play'], axis=1)
y = data['play']
encoder = ce.OrdinalEncoder(cols=['outlook','temperature','humidity','wind'])
X_pred = encoder.fit_transform(X)
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_pred,y)
y_pred_gini = clf_gini.predict(X_pred)
print(y_pred_gini)
print(accuracy_score(y, y_pred_gini)*100)