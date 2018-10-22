# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:44:39 2018

@author: Utilisateur
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1/(1 + np.exp(-x))

def fp(x):
    return f(x)*(1-f(x))

class CoucheEntree():
    
    def __init__(self, nbNeuronnes):
        self.nbNeuronnes = nbNeuronnes
        self.neuronnes = np.array([[0] for i in range(nbNeuronnes)])
        
        
    def setValue(self, valeurs):
        self.neuronnes = np.array([[valeurs[i]] for i in range(self.nbNeuronnes)])
        self.z = self.neuronnes
        
    def affiche(self):
        print("Neuronnes : ", self.neuronnes)

class Couche():
    
    def __init__(self, nbNeuronnes, preCouche, f, fp):
        self.nbNeuronnes = nbNeuronnes
        self.neuronnes = np.array([[0] for i in range(nbNeuronnes)])
        self.preCouche = preCouche
        self.z = np.array([])
        self.f = f
        self.fp = fp
        self.w = np.random.normal(size = (nbNeuronnes, preCouche.nbNeuronnes))
        self.b = np.random.normal(size = (nbNeuronnes, 1))
        

    def propag(self):
        self.z = np.dot(self.w, self.preCouche.neuronnes) + self.b
        #print("z : ", self.z, "sans b", np.dot(self.w, self.preCouche.neuronnes))
        self.neuronnes = self.f(self.z)
        
    def setPoid(self, W):
        self.w = np.array(W)
        
    def setBiais(self, B):
        self.b = np.array(B)
        
    def retroPropag(self, y, W_apres, alreadyDelta = False):
        
        if(not(alreadyDelta)):
            delta = 2*(self.neuronnes - y)
        else:
            delta = np.array(y)
        
        df = fp(self.z)
        df = np.array(df)
        
        delta = np.multiply(delta, df)
        self.deltaNeuronnes = np.dot(np.transpose(self.w), delta)
        
        self.deltaPoid = np.dot(delta, np.transpose(self.preCouche.neuronnes))
        self.deltaBiais = delta
        
        return self.deltaNeuronnes
        
    def addGrad(self, n):
        self.gradPoid = self.gradPoid + ((1/n)*self.deltaPoid)
        self.gradBiais = self.gradBiais + ((1/n)*self.deltaBiais)
        
    def subGrad(self):
        self.w = self.w - 0.1*self.gradPoid
        self.b = self.b - 0.1*self.gradBiais
        
    def resetGrad(self):
        self.gradPoid = np.zeros((self.nbNeuronnes, self.preCouche.nbNeuronnes))
        self.gradBiais = np.zeros((self.nbNeuronnes, 1))
    
    def affiche(self):
        print("Z : ", self.z, "\nNeuronnes : ", self.neuronnes, "\nPoid : ", self.w, "\nBiais : ", self.b)
        
        
class Cerveau():
    
    def __init__(self, structure, f, fp, listePoid = "w", listeBiais = "b"):
        
        self.couches = [CoucheEntree(structure[0])]
        print(structure)
        for i in range(len(structure) - 1):
            self.couches.append(Couche(structure[i+1], self.couches[-1], f[i], fp[i]))
            if type(listePoid) != str:
                self.couches[-1].setPoid(listePoid[i])
            if type(listeBiais) != str:
                self.couches[-1].setBiais(listeBiais[i])
            
    def propagation(self, X):
        self.couches[0].setValue(X)
        for i in range(len(self.couches) - 1):
            self.couches[i+1].propag()
        return self.couches[-1].neuronnes
    
    def retroPropag(self, Y):
        self.couches[-1].retroPropag(Y,np.eye(len(Y)))
        for i in range(-2, -len(self.couches), -1):
            self.couches[i].retroPropag(self.couches[i+1].deltaNeuronnes,self.couches[i+1].w, True)
            
        #for i in range(len(self.couches) - 1):
          #  self.couches[i+1].w = self.couches[i+1].w - 0.1*self.couches[i+1].deltaPoid
         #   self.couches[i+1].b = self.couches[i+1].b - 0.1*self.couches[i+1].deltaBiais
    
    def apprendre(self, listX, listY):
        if len(listX) != len(listY):
            print("Les listes ne fon aps la même taille")
        
        for i in range(len(self.couches) - 1):
            self.couches[i+1].resetGrad()
        
        for i in range(len(listX)):
            self.propagation(listX[i])
            self.retroPropag(listY[i])
            for j in range(len(self.couches) - 1):
                self.couches[j+1].addGrad(len(listX))
        for i in range(len(self.couches) - 1):
            self.couches[i+1].subGrad()
        
    
    def affiche(self):
        for c in range(len(self.couches)):
            print('Couche n°' + str(c))
            self.couches[c].affiche()
            print('--------------------------')
    """
Ptest = np.array([ [[-0.31271959, -0.17394585], [-1.22006486, -0.85849702], [-0.02863739, -0.19461938]], [[-0.46563202, -0.12006771, -0.78827869]] ])
Btest = np.array([ [[-0.87754955], [-0.89073197], [2.22737199]], [[-1.5952977]] ])

a = Cerveau([2,3,3,1], [f, f, f], [fp, fp, fp])

X = [[0,0], [1,0], [0,1], [1,1]]
Y = [np.array([np.array([0])]), np.array([np.array([1])]), np.array([np.array([1])]), np.array([np.array([0])])]

XX = [X[i%4] for i in range(20)]
YY = [Y[i%4] for i in range(20)]

def cout():
    res = 0
    res = res + ((a.propagation([0,0])[0][0])**2)
    res = res + (((a.propagation([1,0])[0][0] - 1))**2)
    res = res + (((a.propagation([0,1])[0][0] - 1))**2)
    res = res + ((a.propagation([1,1])[0][0])**2)
    return res

c = []
t = np.linspace(0, 1401, 1401)
for l in range(1400):
    c.append(cout())
    a.apprendre(XX, YY)
c.append(cout())

plt.plot(t, c)
plt.show()
    """

    
        
        