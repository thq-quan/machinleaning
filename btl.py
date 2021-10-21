import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#The original data can never be added or deleted columns
original_data = pd.read_csv("Fish.csv")
#The data variable is used to make modifications on it
data = copy.deepcopy(original_data)
data.head()

np.sum(data.isnull())

original_data["Species"] = pd.DataFrame(original_data["Species"]).apply(LabelEncoder().fit_transform)

sns.heatmap(data.corr(), annot=True)

corr = data.corr()["Weight"].drop("Weight")
print(corr)

#Variables for keeping track of errors are initialized
e_train = []
e_test = []
e_train_hist = []
e_test_hist = []
alpha_hist = []
alpha = []

#Max degree of the regression
max_degree = 5

#No. of training times
training_times = 50

#Iterate over the different degrees
for degree in range(1,max_degree):
    poly = PolynomialFeatures(degree)
    data = copy.deepcopy(original_data)
    y = pd.DataFrame(data["Weight"])
    data = data.drop("Weight", axis = 1)
    x = poly.fit_transform(data)
    for i in range(training_times):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random.randint(100))
        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 4, 6, 8, 16, 32, 40, 50, 80, 100, 150, 200, 250, 300, 350, 400])
        model.fit(x_train, y_train)
        #Training error is recorded
        e = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
        e_train.append(e)
        #Test error is recorded
        e = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))
        e_test.append(e)
        #The alpha hyperparameter is recorded
        alpha.append(model.alpha_)
    #The records of the current degree are saved
    e_train_hist.append(e_train)
    e_train = []
    e_test_hist.append(e_test)
    e_test = []
    alpha_hist.append(alpha)
    alpha = []

#The mean for each degree is calculated
e_train = np.mean(np.array(e_train_hist),axis=1)
e_test = np.mean(np.array(e_test_hist),axis=1)
alpha = np.mean(np.array(alpha_hist),axis=1)

#The errors and alpha record is plotted
plt.plot(range(1,max_degree), e_train, 'o-', label = "train")
plt.plot(range(1,max_degree), e_test, 'o-',label = "test")
plt.legend()
plt.figure()
plt.plot(range(1,max_degree), alpha, 'o-',label = "alpha")
plt.legend()