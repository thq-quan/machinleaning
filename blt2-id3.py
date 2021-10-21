import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import category_encoders as ce

data = pd.read_csv("D:\hk1-gd1-4\mechineleaning\indian_liver_patient_dataset.csv")
X = data.drop(['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
encoder = ce.OrdinalEncoder(cols=['gender'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf = DecisionTreeClassifier(criterion='entropy', max_depth= None, random_state=0)
clf.fit(X_train, y_train)
y_pred_gini = clf.predict(X_test)
print(y_pred_gini)
print(accuracy_score(y_test, y_pred_gini)*100)  