import numpy as np
import matplotlib.pyplot as plt

from other_functions import lower_diagonal_matrix

#set initial conditions
"""Initial conditions should specify the average velocity and density 
    in each cell"""
#set model size and precision

N = 5 #number of cells
dx = 2. #size of a cell in meters
dt = 2. #time step
##set average velocity and density
v0 = np.array(N * [40e3*60])#initial uniform velocity of 40 km/h given in m/min
u = np.zeros((5,N)) # N cells with a max time of 200*dt
v_M = 120e3 * 60 #120 km/h in m/min
rho_M = 0.3 #in cars/m
u[0] = np.array(N * [0])
u[0][0]=0.2
#solve equations 



def propagate_one_step(u):
    """"Returns u[n+1] array at time t+dt given initial u[n] array at time t.
    
    
    The godunov scheme to be used is for the 1st order fundamental diagram:
    q(rho) = v_M *  rho(x,t) * (1-rho/rho_M).

    If u[n][i] = rho(i * dx, n * dt), then the following equation follows:
    
        u[n+1][i]  =  (1 - v_M*dt/dx) * u[n][i]  +  v_M*dt/dx * u[n][i-1]  + 
                      +  2*v_M*dt/dx/rho_M * u[n][i] * u[n][i]   
                      -  2*v_M*dt/dx/rho_M * u[n][i] * u[n][i-1]
    
    The first 2 terms are the result of a matrix multiplication and the next two
    of a product of the vector u[n] with itself."""
    c = v_M * dt/dx #much used coeff
    print("c:",c)
    M = np.eye(N) * (1 - c) + lower_diagonal_matrix(N) * c
    M[0][N-1]=c #due to periodic boundary conditions 
    print("M:",M)
    u_offset = np.append(u[N-1],u[0:-1])
    print("u offset:",u_offset,"u*uoff:",u_offset*u,u**2)
    print("M@u:",M@u)
    u_result = M@u  +  (2*c/rho_M) * u**2  +  (-2*c/rho_M) * u * u_offset
    print("u next:",u_result)
    return u_result

for i in range(1,u.shape[0]):
    u[i]=propagate_one_step(u[i-1])
print("\nu matrix:\n",u)

#plot and analyse data

#declare 
fig, ax = plt.subplots(1,1,figsize=(6,4))

#do a colorbar style plot of density
data2D = np.random.random((50, 50))
print(np.transpose(u))
data_visual = ax.imshow(np.transpose(u), cmap="viridis")
fig.colorbar(data_visual, ax=ax, location='right')
plt.show()
