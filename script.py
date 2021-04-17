import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year=df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)

X = prod_per_year['year']
X = X.values.reshape(-1, 1)
print(X)

y = prod_per_year['totalprod']

regr = linear_model.LinearRegression()
regr.fit(X,y)

print(regr.coef_[0])
print(regr.intercept_)

y_predict = regr.predict(X)

x_future = np.array(range(2013, 2050))
x_future = x_future.reshape(-1, 1)


future_predict = regr.predict(x_future)
plt.scatter(X ,y)
plt.plot(X,y_predict)
plt.plot(x_future, future_predict)
plt.show()

