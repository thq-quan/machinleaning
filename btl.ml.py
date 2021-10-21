from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



df = pd.read_csv("car_train.csv")
X = np.array(df[['engine_power','age_in_days','km','previous_owners','lat','lon']].values)
Y = np.array(df['pice']).T
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.7, shuffle = False)
regr2 = linear_model.LinearRegression()
regr2.fit(xtrain, ytrain)
w_20=regr2.intercept_
w_21=regr2.coef_[0]
for i in range(len(xtest)):
    kqDuDoan = np.dot(w_21,xtest[i].T) + w_20
    print(kqDuDoan)


