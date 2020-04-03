# -*- coding: utf-8 -*-
"""
Coursework 2: Logistical regression
"""

#%%
"""
Imports and definitions
"""


import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as opt

def carga_csv(file_name):
    """
    carga el fichero csv especificado y lo
    devuelve en un array de numpy
    """
    valores = read_csv(file_name,header=None).values
    #suponemos que siempre trabajaremos con float
    return valores.astype(float)

def sigmoid(z):
    """
    sigmoid function
    can be applied on numbers and on numpy arrays of any dimensions
    """
    return 1 / (1 + np.exp(-z))

def coste(theta, X, Y):
    """
    cost function
    computes J(theta) for a given dataset
    """
    m = np.shape(X)[0]
    H = sigmoid((np.dot(X,theta)))
    J = -1/m * ( np.log(H).transpose().dot(Y)
                 + np.log(1-H).transpose().dot(1-Y)) 
    return J

def gradiente(theta, X, Y):
    """
    gradient function
    computes the gradient of J at a given theta for a given dataset
    """
    m = np.shape(X)[0]
    H = sigmoid(np.dot(X,theta))
    G = 1/m * X.transpose().dot(H - Y)
    return G

def coste_reg(theta, X, Y, lamb):
    """
    cost function with regularization
    computes J(theta) for a given dataset 
    with a regularization factor of lambda
    """
    m = np.shape(X)[0]
    H = sigmoid((np.dot(X,theta)))
    J = -1/m * ( np.log(H).transpose().dot(Y)
                 + np.log(1-H).transpose().dot(1-Y)) 
    reg = lamb/(2*m)*np.sum(theta[1:]**2)
    return J + reg

def gradiente_reg(theta, X, Y, lamb):
    """
    gradient function with regularization
    computes the gradient of J at a given theta for a given dataset 
    with a regularization factor of lambda
    """
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = sigmoid(np.dot(X,theta))
    G = 1/m * X.transpose().dot(H - Y)
    reg = np.ones(n,)
    reg[0] = 0
    reg = (lamb/m)*reg*theta
    return G + reg

def precision(theta,X,Y):
    """
    accuracy function
    computes the accuracy of the logistic model theta on X with true target variable Y
    """
    m = np.shape(X)[0]
    H = sigmoid(np.dot(X,theta))
    H[H >= 0.5] = 1
    H[H < 0.5] = 0
    return np.sum(H == Y)/m

def pinta_frontera_recta(X, Y, theta):
    """
    draws straight decision borders
    """
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
               np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                      xx1.ravel(),
                      xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    # el cuarto parámetro es el valor de z cuya frontera se    
    # quiere pintar    
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    # Add the data to the graph
    plt.scatter(X[pos, 0],X[pos, 1], marker='+', c='k')
    plt.scatter(X[not_pos, 0], X[not_pos, 1], marker='o', c='y')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()
    
def pinta_frontera_curva(X, Y, theta, poly, lamb):
    """
    draws polynomial decision borders
    adds the value of the regularization parameter for wich it was computed
    """
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
                            xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    # el cuarto parámetro es el valor de z cuya frontera se    
    # quiere pintar    
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    # add the data to the graph
    plt.scatter(X[pos, 0],X[pos, 1], marker='+', c='k')
    plt.scatter(X[not_pos, 0], X[not_pos, 1], marker='o', c='y')
    plt.title('lambda = ' + str(lamb))
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


#%% 
"""
1 - Logistical regression
"""


# loading the data
datos = carga_csv('ex2data1.csv')
X = datos[:, :-1]
Y = datos[:, -1]
m = np.shape(X)[0]
n = np.shape(X)[1]

# Visualizing the data
pos = np.where(Y == 1)
not_pos = np.where(Y == 0)
plt.scatter(X[pos, 0],X[pos, 1], marker='+', c='k', label = 'admited')
plt.scatter(X[not_pos, 0], X[not_pos, 1], marker='o', c='y', label = 'not admited')
plt.legend()
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Add the column of 1s
X = np.hstack([np.ones([m,1]),X])

theta = np.zeros((n+1,))
# Compute the best value for theta
result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X,Y), messages=0)
theta_opt = result[0]

# Display the border
pinta_frontera_recta(X[:,1:], Y, theta_opt)

# Display the accuracy of the model
p = precision(theta_opt,X,Y)
print('Model accuracy : {0:1.3g}'.format(p))


#%% 
"""
2 - Regularized logistical regression
"""


datos = carga_csv('ex2data2.csv')
X = datos[:, :-1]
Y = datos[:, -1]
m = np.shape(X)[0]

# Visualizing the data
pos = np.where(Y == 1)
not_pos = np.where(Y == 0)
plt.scatter(X[pos, 0],X[pos, 1], marker='+', c='k', label = 'y = 1')
plt.scatter(X[not_pos, 0], X[not_pos, 1], marker='o', c='y', label = 'y = 0')
plt.legend()
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')

# Adding new attributes
poly = PolynomialFeatures(6)
X_pol = poly.fit_transform(X)
n = np.shape(X_pol)[1]

theta = np.zeros((n,))
lamb_arr = [0,0.3,1,3,10,30]
# Compute the optimal value for theta using various values of lambda
for lamb in lamb_arr:
    # Compute the best value for theta
    result = opt.fmin_tnc(func=coste_reg, x0=theta, fprime=gradiente_reg,
                                        args=(X_pol,Y,lamb), messages = 0)
    theta_opt = result[0]
    # Display the border
    pinta_frontera_curva(X_pol[:,1:], Y, theta_opt,poly,lamb)
    # Display the accuracy of the model
    p = precision(theta_opt,X_pol,Y)
    print('With lambda = ' + str(lamb) + ' we get an accuracy of {0:1.3g}'.format(p))

