# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:02:29 2021

@author: mirof
"""

"""2D wave equation on a heterogeneous model"""

import numpy as np
import matplotlib.pyplot as plt

"""For stability the CFL condition has to be[1]:
[1] Bording, Lines P., and R. L. Slawinski. "A Recipe For Stability Analysis Of Finite Difference Wave Equation Computations." (1998).
"""

#1D- maximum k equal to nyquist wavenumbr knyq = np.pi/dx
#Stability requires CFL criteria <= 2/np.pi -> 0.636

#2D - kmax = np.sqrt(2)*knyq
#Stability requires CFL criteria <= np.sqrt(2)/np.pi -> 0.45

#3D - kmax = np.sqrt(3)*knyq
#Stability requires CFL criteria <= 2/np.sqrt(3)*np.pi -> 0.367

#%%

c0 = 2000       #Smallest speed of propagation for the CFL criteria
f0 = 8          #Dominant frequency at source, also for CFL criteria
t = 3.5
isrx = 100      #Source location at x-axis
isrz = 100      #Source location at z-axis
snap = 200      #Snapshot frequency
t0 = 999./f0
oper = 3
#Satisfying the CFL criteria:

dx = (c0/f0)/20 #dividing velocity by frequency and an arbitrary number of grid points to image each wavelet
dz = dx
nx = 500 #Number of grid points you will need according to the smallest cell size
nz = 500
#Inserting boundary conditions
nx_bor = 40 #boundary of 20 on each size of the model
nz_bor = 40
nz_new = nz + nz_bor #Size of the new model with the extra boundary
nx_new = nx + nx_bor

dt = 0.001  #Smallest time cell size according to CFL criteria
nt = 500       #number of time grid points
t = np.linspace(0, nt*dt, nt) #Time

#Creating an array for speed, from 0-400(c=3000), 400-600(c=2400), 600-1000(c=3000)
c = np.zeros((nz, nx))
c[0:nx//2-100, :] = 2000
c[nx//2-100:nx//2+100, :] = 3000
c[nx//2+100:nx, :] = 3500 

# plt.figure() 
# plt.imshow(c)
# plt.colorbar()
# plt.title("Velocity model")

cmax = 3500
eps  = cmax * dt / dx # epsilon value
print('CFL condition =', eps)



#%%

#Ricker source - Its the second derivative of a gaussian source, the simple 2D case had the first derivative of the gaussian source

def ricker(f, length=250, dt=0.001):
    t = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*((t+t0)**2)) * np.exp(-(np.pi**2)*(f**2)*((t+t0)**2))
    return t, y

f = 8 # A low wavelength of 25 Hz
t, w = ricker(f)

# plt.figure()
# plt.title("Ricker function")
# plt.xlabel("time")
# plt.ylabel("Amplitude of ricker function")
# plt.plot(t,w)
# plt.show()

spec = np.fft.fft(w) # Temporal source function in the frequency domain
freq = np.fft.fftfreq(spec.size, d = dt) # Time domain in frequency
# plt.plot(np.abs(freq), np.abs(spec)) # Plotting the frequency and amplitude
# plt.xlim([0,80]) #Limiting x window


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
        d2px[i, :] = (p[i + 1, :] - 2 * p[i, :] + p[i - 1, :])
        
    for j in range (1, nz-1):
        d2pz[:, j] = (p[:, j + 1] - 2 * p[:, j] + p[:, j - 1])
       
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
    pnew = (dt ** 2) * (c ** 2) * (d2px + d2pz) + 2 * p - pold
    
    pnew[isrz, isrx] = pnew[isrz, isrx] + w[it] / (dx * dz) * (dt ** 2)
    
    #Plotting the snapshots   
    # if (it % snap == 0):
    #     plt.figure()
    #     plt.imshow(pnew)
    #     plt.colorbar()
    #     plt.clim(-0.2,0.2)
    
    pold, p = p, pnew

plt.figure()
plt.title("2D Wave equation")
plt.imshow(pnew)
plt.clim(-0.2,0.2)
plt.colorbar()
plt.show()
