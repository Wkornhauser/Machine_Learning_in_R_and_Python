#Polynomial Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
""" Theres not enough data to justify splitting here. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Make both a linear and polynomial model to compare which is better!

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting the Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the Linear Regression Results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#Visualising the Polynomial Regression Results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#Now add a degree to make it even better.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualising the Polynomial Regression Results with degree of 3
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#Now add a degree to make it even better.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualising the Polynomial Regression Results with degree of 4
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#CLean shit.

#What about 6? or 10?

#Now add a degree to make it even better.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualising the Polynomial Regression Results with degree of 6
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#6 is like perfect lmao

#10?
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualising the Polynomial Regression Results with degree of 6
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#Can't even tell a difference from 6 to 10. Perhaps this is where it begins to fit the data too well, and would not work accurately outside of the bounds, but without outside data you cannot tell regardless.

#What about using degree of 6 but with a smooth curve, no straight lines? You need to not plot X as 1,2,3...10 but rather with smaller increments.
#Do the following.
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

#This gives a matrix for X with 10x more intervals to make for nicer looking plots. Try plotting with degree of 4.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#Visualising the Polynomial Regression Results with degree of 6
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
#Using poly_reg.fit_transform instead of X_poly as the input for the lin_reg_2.predict() function above allows for dynamic plotting
plt.title("Truth or Bluff (Polynomial Regression )")
plt.xlabel("Position Level")
plt.ylabel ("Salary")
plt.show

#This shows a much nicer curve!

#Predicting a new result at 6.5 with a Linear Regression
lin_reg.predict([[6.5]])

#Predicting a new result at 6.5 with a Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))