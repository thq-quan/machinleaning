import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import category_encoders as ce

X = np.arange(20).reshape(-1, 2)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

svc = SVC( kernel='linear')
svc.fit(X,y)
y_pred=svc.predict(X)
print(y_pred)
print(accuracy_score(y, y_pred)*100)