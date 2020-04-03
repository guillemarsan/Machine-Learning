# -*- coding: utf-8 -*-
"""
Coursework 0: Integration
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Usando numpy
def integra_mc(f,a,b,num_puntos=10000):
    x = np.linspace(a,b,num_puntos)
    y = f(x)
    M = np.amax(y)
    p = np.random.rand(num_puntos,2)
    p[:,0] = p[:,0]*(b-a)+a
    p[:,1] = p[:,1]*M
    r = p[:,1] < f(p[:,0])
    N = np.sum(r)
    I = (N/num_puntos) * (b-a) * M
    return I,p

# Sin usar numpy
def integra_mc2(f,a,b,num_puntos=10000):
    x = a
    step = (b-a)/num_puntos
    M = 0
    for i in range(num_puntos):
        M = max(M,f(x))
        x = x + step
    p = np.random.rand(num_puntos,2)
    N = 0
    for j in range(num_puntos):
        p[j,0] = p[j,0]*(b-a)+a
        p[j,1] = p[j,1]*M
        N = N + (p[j,1] < f(p[j,0]))
    I = (N/num_puntos) * (b-a) * M
    return I,p

f = lambda x: x**2
a = 0
b = 3
tic = time.process_time()
I,p = integra_mc(f,a,b,100000)
toc = time.process_time()
print("Integral calculada:",I, "en tiempo",(toc-tic)*1000, " ms")

tic = time.process_time()
I,p = integra_mc2(f,a,b,100000)
toc = time.process_time()
print("Integral calculada:",I, "en tiempo",(toc-tic)*1000," ms")

x = np.linspace(a,b,1000)
plt.scatter(p[:,0],p[:,1], s=1)
plt.plot(x,f(x), 'r')

#%%
"""
Measure times and accuracy
"""

tnp = []
t = []
Inp = []
Isnp = []
num_puntos = range(1000,100000,1000)
for n in num_puntos:
    tic = time.process_time()
    I,p = integra_mc(f,a,b,n)
    toc = time.process_time()
    tnp.append((toc-tic)*1000)
    Inp.append(I)
   
    tic = time.process_time()
    I,p = integra_mc2(f,a,b,n)
    toc = time.process_time()
    t.append((toc-tic)*1000)
    Isnp.append(I)
 
plt1, = plt.plot(num_puntos,tnp, 'r', label = 'Con Numpy')
plt2, = plt.plot(num_puntos,t, 'g', label = 'Sin Numpy')
plt.legend(handles=[plt1,plt2])
plt.xlabel('$N_{total}$')
plt.ylabel('Tiempo en ms')
plt.show()

plt1, = plt.plot(num_puntos,Inp, 'r', label = 'Con Numpy')
plt2, = plt.plot(num_puntos,Isnp, 'g', label = 'Sin Numpy')
plt.legend(handles=[plt1,plt2])
plt.xlabel('$N_{total}$')
plt.ylabel('I')
plt.show()
\end{minted}