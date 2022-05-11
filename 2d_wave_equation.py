# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:52:06 2021

@author: mirof
"""

#2D Wave equation

import numpy as np
import matplotlib.pyplot as plt
#%%
c = 343         #Speed of propagation
xmax = 300      #size of x-axis
nx = 500        #size of samples of x
zmax = 300      #size of z-axis
nz = 500        #size of samples of z
dx = 1          #size of step in x
dz = dx         #size of step in z
isrx = int(xmax/2)      #Location of source in x axis
isrz = int(zmax/2)      #Location of source in z axis
oper = 5        #Finite-difference operator number (3 or 5)

f0 = 20         #Dominant frequency
t0 = 1./f0      #Source time shift
nt = 1000       #size of samples of t
dt =  0.001

#Determining the CFL criterion
eps  = c * dt / dx # epsilon value
print('Critério de estabilidade =', eps)

t_s = np.linspace(0, 1, nt)
t = np.linspace(0, nt*dt, nt)
src = -8. * (t_s - t0) * f0 * (np.exp(-1.0 * (4 * f0)**2 * (t_s-t0)**2))

# plt.figure()
# plt.title("Gaussian function")
# plt.xlabel("time")
# plt.ylabel("Amplitude of gaussian function")
# plt.plot(t,src)
# plt.show()

#%%

spec = np.fft.fft(src) # Source time function in the frequency domain
freq = np.fft.fftfreq(spec.size, d = dt) # Time domain in frequency
plt.plot(np.abs(freq), np.abs(spec)) # plot frequency and amplitude
plt.xlim([0,250])

#%%
#Wave equation

#assigning the variables

p = np.zeros((nx, nz))
pnew = np.zeros((nx, nz))
pold = np.zeros((nx, nz))
d2px = np.zeros((nx, nz))
d2pz = np.zeros((nx, nz))

for it in range(nt):
    #3 point operator for the finite-difference
    if oper == 3
        for i in range (1, nx-1):
            d2px[i, :] = (p[i + 1, :] - 2 * p[i, :] + p[i - 1, :]) 
             #Second pressure derivative in respect to x
        
        for j in range (1, nz-1):
            d2pz[:, j] = (p[:, j + 1] - 2 * p[:, j] + p[:, j - 1])
            #Second pressure derivative in respect to z
            
    #5 point operator for the finite-difference
    if oper == 5:
        for i in range (2, nx-2):
            d2px[i, :] = -1/12*p[i + 2, :] + 4/3*p[i + 1, :] \
                          -5/2 * p[i, :] + 4/3*p[i - 1, :] + -1/12 * p[i - 2, :]
                
        for j in range (2, nz-2):
            d2pz[:, j] = -1/12*p[:, j + 2] + 4/3 * p[:, j + 1] \
                          -5/2 * p[:, j] + 4/3 * p[:, j - 1] + -1/12 * p[:, j - 2]
    
    d2px /= dx ** 2
    d2pz /= dz ** 2
    
    #Calculation of the presure field
    pnew = (dt ** 2) * (c ** 2) * (d2px + d2pz) + 2 * p - pold

    #Again calculating the pressure field but with loop for time
    pnew[isrz, isrx] = pnew[isrz, isrx] + src[it] / (dx * dz) * (dt ** 2)
    
    #Plotting the snapshots
    if (it % snap == 0):
        plt.figure()
        plt.imshow(pnew)
        plt.colorbar()
        #plt.clim()
        
    pold, p = p, pnew #Assigning the variables for an effective loop

#plt.figure()
#plt.title("2D Wave equation")
#plt.imshow(pnew)
#plt.colorbar()
#plt.show()
