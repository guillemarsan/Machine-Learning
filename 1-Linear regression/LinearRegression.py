# -*- coding: utf-8 -*-
"""
Coursework 1: Linear regression
"""

import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import cm
import matplotlib.pyplot as plt

# Download data
def carga_csv(file_name):
    valores=read_csv(file_name,header=None).values
    return valores.astype(float)

# Normal equation for exact computation of theta
def ecuacion_normal(X, Y):
    X_t = np.transpose(X)
    Aux = np.dot(X_t,X)
    X_plus = np.linalg.pinv(Aux).dot(X_t)
    return X_plus.dot(Y)

# Normalize dataset
def normalizar(X):
    mu = np.mean(X,  axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Compute the gradient of the cost function
def gradiente(X, Y, theta):
    res = np.zeros((np.shape(X)[1],1))
    m = np.shape(X)[0]
    H = np.dot(X,theta)
    res = 1/m * np.dot(np.transpose(X), H-Y)
    return res

# Compute the cost function
def coste(X, Y, theta):
    H = np.dot(X, theta)
    Aux = (H - Y)**2
    return Aux.sum() / (2*len(X))

# Compute gradient descent for a maximum of maxIter iterations or until
# the norm of the gradient is lower than eps
def descenso_gradiente(X, Y, alpha, maxIter, eps):
    theta = np.zeros((np.shape(X)[1],1))
    grad = gradiente(X, Y, theta) 
    it = 0
    costes = []
    while np.linalg.norm(grad) > eps and it < maxIter:
        theta = theta - alpha*grad
        grad = gradiente(X, Y, theta)
        costes.append(coste(X, Y, theta))
        it += 1
    return theta, costes

#%%
"""
One variable case
"""
datos = carga_csv('ex1data1.csv')
X = datos[:,:-1]
m = np.shape(X)[0]
Y = datos[:,-1]


# Add the column of 1s
X=np.hstack([np.ones([m,1]),X])
Y = Y[:,np.newaxis]

alpha = 0.01
maxIter = 1500
eps = 1e-3
Thetas, costes = descenso_gradiente(X, Y, alpha, maxIter, eps)

# Plot data an line
x = np.linspace(0,23,23)
plt.scatter(X[:,1],Y,s=15)
plt.plot(x, Thetas[0] + Thetas[1]*x, c='r')
plt.xlabel('Población de la ciudad en 10 000s')
plt.ylabel('Ingresos en $10 000s')

#%%
"""
Plot cost function
"""
# Plot function in 3D
fig = plt.figure()
ax = fig.gca(projection = '3d')

Th0 = np.arange(-10,10,0.05)
Th1 = np.arange(-1,4,0.05) 
Th0m,Th1m = np.meshgrid(Th0,Th1)
cost = np.zeros((len(Th1),len(Th0)))
for i in range(len(Th1)) :
    for j in range(len(Th0)):
        cost[i][j] = coste(X,Y, np.array([[Th0[j]],[Th1[i]]]))
curve = ax.plot_surface(Th0m,Th1m,cost, cmap = cm.rainbow)
ax.view_init(elev = 15, azim=230)
plt.xlabel(r'$\theta_{0}$')
plt.ylabel(r'$\theta_{1}$')
ax.set_zlabel(r'$J(\theta)$',rotation=0)

# Plot function as contours
plt.figure()
plt.contour(Th0m,Th1m,cost, np.logspace(-2,3,20))
plt.scatter(Thetas[0],Thetas[1], marker='X',c='r')
plt.xlabel(r'$\theta_{0}$')
plt.ylabel(r'$\theta_{1}$')      

#%%
"""
Check for multivariate case different learning rates alpha
"""
datos = carga_csv('ex1data2.csv')
X_old = datos[:,:-1]
m = np.shape(X_old)[0]
Y_old=datos[:,-1]

# Normalize data
X, mu_x, sigma_x = normalizar(X_old)

# Add the column of 1s
X_old = np.hstack([np.ones([m,1]),X_old])
X = np.hstack([np.ones([m,1]),X])
Y = Y_old[:,np.newaxis]

alpha_arr = [0.3,0.1,0.03,0.01,0.003,0.001]
maxIter = 1500
eps = 1e-3
plt.figure()
for alpha in alpha_arr:
    Thetas, costes = descenso_gradiente(X, Y, alpha, maxIter, eps)
    plt.plot(costes, label = ("With alpha = " + str(alpha)))
plt.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$J(\theta)$') 
#%%
"""
Multivariate case
"""
datos = carga_csv('ex1data2.csv')
X_old = datos[:,:-1]
m = np.shape(X_old)[0]
Y_old=datos[:,-1]

# Normalize data
X, mu_x, sigma_x = normalizar(X_old)

# Add the column of 1s
X_old = np.hstack([np.ones([m,1]),X_old])
X = np.hstack([np.ones([m,1]),X])
Y = Y_old[:,np.newaxis]

alpha = 0.3
maxIter = 1500
eps = 1e-3
Thetas, costes = descenso_gradiente(X, Y, alpha, maxIter, eps)

# Compute with the normal equation and compare
th = ecuacion_normal(X_old,Y)
test = np.array([1650,3])
test_n = (test - mu_x)/sigma_x
test = np.hstack([1,test])
test_n = np.hstack([1,test_n])
price = np.dot(test,th)[0]
price_n = np.dot(test_n,Thetas)[0]
print("The value computed with the normal equation for x = [1650,3] is y = ", price)
print("The value computed with gradient descent for x = [1650,3] is y = ", price_n)
print("The relative distance between the two values is ", np.abs((price - price_n)/ price))


