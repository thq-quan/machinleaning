import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x = np.arange(20).reshape(-1, 2)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)
print(x)
print(model.predict(x))
print(accuracy_score(y,model.predict(x))*100)