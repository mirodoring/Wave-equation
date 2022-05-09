# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:52:06 2022

@author: mirof
"""

#2D Wave equation

import numpy as np
import matplotlib.pyplot as plt
#%%
c = 343        #Velocidade de propagação
xmax = 300      #Tamanho do eixo x
nx = 500        #número de amostras do eixo x
zmax = 300      #Tamanho do eixo z
nz = 500        #Número de amostras do eixo z
dx = 1          #Passo do eixo x
dz = dx         #Passo do eixo z
isrx = int(xmax/2)      #Localização da fonte no eixo x
isrz = int(zmax/2)      #Localização da fonte no eixo z

f0 = 20         #Frequência dominante da fonte
t0 = 1./f0      #mudança de tempo na fonte
nt = 1000        #número de amostras em t
dt =  0.001

print('Frequencia da fonte: ', f0, 'Hz')
#Determinando o critério de estabilidade CFL
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

spec = np.fft.fft(src) # Função temporal da fonte no domínio da frequencia
freq = np.fft.fftfreq(spec.size, d = dt) # domínio do tempo na frequencia
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
    for i in range (1, nx-1):
        d2px[i, :] = (p[i + 1, :] - 2 * p[i, :] + p[i - 1, :]) / dx ** 2
        
    for j in range (1, nz-1):
        d2pz[:, j] = (p[:, j + 1] - 2 * p[:, j] + p[:, j - 1]) / dz ** 2
    
    pnew = (dt ** 2) * (c ** 2) * (d2px + d2pz) + 2 * p - pold
    
    # pnew[isrc] = pnew[isrc] + dt**2 * src[it]
    # pnew[isrx, isrz] = pnew[isrx, isrz] + src[it] * dt ** 2
    pnew[isrz, isrx] = pnew[isrz, isrx] + src[it] / (dx * dz) * (dt ** 2) 
    
    pold, p = p, pnew

plt.figure()
plt.title("2D Wave equation")
plt.imshow(pnew)
# plt.clim(-1e05,1e05)
plt.colorbar()
plt.show()















