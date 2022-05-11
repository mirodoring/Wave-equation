# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:02:29 2022

@author: mirof
"""

"""2D wave equation on a heterogeneous model"""

import numpy as np
import matplotlib.pyplot as plt
#%%

c0 = 2000       #Smallest speed of propagation for the CFL criterion
f0 = 25         #Dominant frequency at source, also for CFL criteria
t = 3.5
isrx = 100      #Source location at x-axis
isrz = 100      #Source location at z-axis

#Satisfying the CFL criteria:

dx = (c0/f0)/20 #dividing velocity by frequency and an arbitrary number of grid points to image each wavelet
dz = dx
nx = 500 #Number of grid points you will need according to the smallest cell size
nz = 500

dt = (0.7*dx)/c0  #Smallest time cell size according to CFL criteria
nt = 2000       #number of time grid points
t = np.linspace(0, nt*dt, nt) #Time

#Creating an array for speed, from 0-400(c=3000), 400-600(c=2400), 600-1000(c=3000)
c = np.zeros((nz, nx))
c[0:nx//2-100, :] = 2000
c[nx//2-100:nx//2+100, :] = 3000
c[nx//2+100:nx, :] = 3500 
plt.imshow(c)



#%%

#Ricker source - Its the second derivative of a gaussian source, the simple 2D case had the first derivative of the gaussian source

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
