# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:02:29 2022

@author: mirof
"""

"""2D wave equation on a heterogeneous model"""

import numpy as np
import matplotlib.pyplot as plt
#%%

c0 = 2000       #Menor velocidade de propagação para o critério CFL
f0 = 25         #Frequência dominante da fonte, também utilizado para o critério CFL
t = 3.5
isrx = 100      #Localização da fonte no eixo x
isrz = 100      #Localização da fonte no eixo z
isnap = 10      #Frequencia de snapshot

#Satisfazendo os critérios do CFL:

dx = (c0/f0)/20 #divisão da velocidade pela frequencia e um número arbitrário de pontos do grid pra imagear cada wavelet
dz = dx
nx = 500 #Quantidade de pontos de grid que vai precisar de acordo com o tamanho da menor célula
nz = 500

dt = (0.7*dx)/c0  #Menor tamanho da célula do tempo de acordo com o critério de CFL
nt = 2000       #número de pontos do grid do tempo
t = np.linspace(0, nt*dt, nt) #tamanho do tempo

#Criando um array para a velocidade, de 0-400(c=3000), 400-600(c=2400), 600-1000(c=3000)
c = np.zeros((nz, nx))
c[0:nx//2-100, :] = 2000
c[nx//2-100:nx//2+100, :] = 3000
c[nx//2+100:nx, :] = 3500 
plt.imshow(c)



#%%

#Fonte de ricker/Ricker source

def ricker(f, length=250, dt=0.001):
    t = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

f = 25 # A low wavelength of 25 Hz
t, w = ricker(f)

#%%

#2D Wave equation

#assigning the variables

p = np.zeros((nx, nz))
pnew = np.zeros((nx, nz))
pold = np.zeros((nx, nz))
d2px = np.zeros((nx, nz))
d2pz = np.zeros((nx, nz))



#Calculting the wave equation through finite differences
for it in range(nt):
    #Using a 3 point operator for the second derivative

    for i in range (1, nx-1):
        d2px[i, :] = (p[i + 1, :] - 2 * p[i, :] + p[i - 1, :]) / dx ** 2
        
    for j in range (1, nz-1):
        d2pz[:, j] = (p[:, j + 1] - 2 * p[:, j] + p[:, j - 1]) / dz ** 2

    
    pnew = (dt ** 2) * (c ** 2) * (d2px + d2pz) + 2 * p - pold
    
    pnew[isrz, isrx] = pnew[isrz, isrx] + w[it] / (dx * dz) * (dt ** 2)
    
    pold, p = p, pnew

plt.figure()
plt.title("2D Wave equation")
plt.imshow(pnew)
# plt.clim(-1e05,1e05)
plt.colorbar()
plt.show()






























