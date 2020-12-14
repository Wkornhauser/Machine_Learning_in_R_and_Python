#Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding column 1
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encoding column 2
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Making dummy variables for column 1 (any encoded column with more than 2 variables needs this done.) This produces 3 new columns and sticks them in the front (assigns them as columns 0, 1, and 2.)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Removing one dummy variable to prevent dummy variable trap (this is done for country here, however it needs to be done any time there is a category with more than 2 strings that was encoded to numbers.)
# Dummy variable trap is where you overspecify (DOF is too low!)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Rectifier function is arguably the best activation functions. Sigmoid functions are good for probabilistic not deterministic models (ie chance of meeting proft?) Therefor use rectifying for hidden layers and sigmoid for output layer!

# The output_dim argument in the Dense() is how many nodes you want for that particular hidden layer. The ideal number is described as being an "art" by Haedlin. There are some rules however. 1. If your data is linearly seperable, you don't need a hidden layer or NN. 2. His tip is to choose the number of nodes in the hidden layer as the average number of nodes in the input layer and number of nodes in the output layer. In this case it would be (11 + 1) / 2 = 6. If you want better, you need to practise/tune your parameters(experiment). This can include K-fold cross validation in part 10.  
# init = initialisation method, to make all weights close to 0 uniformly.
# activation is for assigning the activation function. Since we've been told that rectifying is best for hidden layers, we're using that. It's called 'relu'
# input_dim is required, and says how many input independent variables there are. In this case its 11.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer (you can add infinite more hidden layers. The more the "better", but also more computatively intense.)
# Output dimension is 1 since it is a binary predictor, ie a yes or no. If you have a dependent variable with more than two categories, ie three plus categories, you need to change output dimension to the number of classes. With the onehotencoder this means you have one output dimension for each potential output. ie 3 for 3, 11 for 11, etc. The other thing that must be changed is activation function. You would use softmax instead of sigmoid. Softmax is a sigmoid function applied to a dependent variable with more than two categories. Theres only one output of a 1 or 0. Sigmoid is also selected as the activation function for aforementioned reasons.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Optimizer is algorithm to find optimal set of weights in the NN. We'll use the stochastic gradient descent, and a specific good algorith is known as 'adam'. Loss refers to loss in the adam function. This will be the logarithmic loss function, as it is also the loss function for sigmoid functions (our output layer!). If the output is binary it is known as binary_crossentropy. If it is nonbinary it is categorical_crossentropy.
# The third and final argument is metrics. It is the criterium to evaluate the model. Haedlin likes the accuracy criterium. Metrics wants it's input as a list, even though here we're only goingto use one element. Hence, we use the [] to create a list of 1 element, accuracy.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
# epochs arguments is how many times you want the weighting between layers to be adjusted. Higher is better but more computatively intensive.
# batch_size is number of observations before weights are adjusted.
# There is no rule of thumb for what to choose for these. Again, you must be an "artist" as Haedlin describes.
# We'll use 10 and 100 just for times sake.
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert percentage to true or false for y_pred
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# We got a 86% accuracy without any tuning. This is insane.