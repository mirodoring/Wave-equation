# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:45:22 2021

@author: mirof
"""

"""Equação da onda"""

# Import Libraries 
# ----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


#%%

#Estabelecendo os domínios x e t
xmax = 10000                    # physical domain (m)
nx = 10000                      # número de amostras 
dx = xmax/(nx-1)                # incremento na direção x


dt = 0.002                     # incremento na direção t
nt = 1001                       # numero de passos em t
time = np.linspace(0, nt*dt, nt)


# Parâmetros da onda
l = 7                           # wavelength
c = 343                         # velocidade
f0 = 25                         # frequencia

t0 = 4./f0         
               
src  = np.zeros(nt + 1)
#1 derivada da gaussiana (função source time)
src  = -8. * (time - t0) * f0 * (np.exp(-1.0 * (4*f0) ** 2 * (time - t0) ** 2))

plt.figure()
plt.title("Função gaussiana da fonte")
plt.xlabel("tempo")
plt.ylabel("Amplitude da pressão") 
plt.plot(time,src)
plt.show()


#%%

p    = np.zeros(nx) # p no tempo n (now)
pold = np.zeros(nx) # p no tempo n-1 (past)
pnew = np.zeros(nx) # p no tempo n+1 (present)
d2px = np.zeros(nx) # 2 derivada no espaço de p



for i in range (1, nx-1):
    d2px[i] = (p[i + 1] - 2*p[i] + p[i -1])/dx**2
    
    # pnew[it+1] = 2 * p[it] - pold[it-1] + c ** 2 * dt ** 2 * d2px
    pnew = 2 * p - pold + c ** 2 * dt ** 2 * d2px
    
    pold, p = p, pnew

plt.plot(time,pnew)


#%%

 


    
    












































