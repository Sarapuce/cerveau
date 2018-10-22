# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 23:25:02 2018

@author: Utilisateur
"""
from xorv2 import Cerveau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
plt.gray() 
plt.matshow(digits.images[5]) 
plt.show()


def f(x):
    return 1/(1 + np.exp(-x))

def fp(x):
    return f(x)*(1-f(x))

#(1797,)
temp = np.eye(10)
temp = [temp[i] for i in range(10)]
for i in range(len(temp)):
    temp[i] = np.array([[temp[i][j]] for j in range(len(temp[i]))])

fin = int(0.75*1797)

XX = (1/16)*digits.data[:fin]
YY = np.array([temp[digits.target[i]] for i in range(fin)])

Xtest = (1/16)*digits.data[fin:]
Ytest = digits.target[fin:]

def newSet():
    choix = np.arange(fin)
    np.random.shuffle(choix)
    XXX = []
    YYY = []
    for i in range(len(choix)):
        XXX.append(XX[choix[i]])
        YYY.append(YY[choix[i]])
    return XXX,YYY




a = Cerveau([64, 32, 16, 16, 10], [f, f, f, f], [fp, fp, fp, fp])



def resCerveau(cerveau, elt):
    cerveau.propagation(elt)
    res = np.array([cerveau.couches[-1].neuronnes[i][0] for i in range(len(cerveau.couches[-1].neuronnes))])
    return np.argmax(res)
    

def matriceConfusion(cerveau, Xtest, Ytest):
    mat = np.zeros((10,10))
    for i in range(len(Xtest)):
        mat[resCerveau(cerveau, Xtest[i])][Ytest[i]] += 1
    return mat

def precision(cerveau, Xtest, Ytest):
    pres = 0
    for i in range(len(Xtest)):
        if resCerveau(cerveau, Xtest[i]) == Ytest[i]:
            pres += 1
    return pres / len(Xtest)
    

c = []
t = np.linspace(0, 999, 1000)

for i in tqdm(range(1000)):
    Xdata, Ydata = newSet()
    a.apprendre(Xdata, Ydata)
    c.append(precision(a, Xtest, Ytest))
    #print(matriceConfusion(a, XX[fin:], digits.target[fin:]))
    
plt.plot(t,c)
plt.show()

    

    
plt.plot(t, c)
plt.show()