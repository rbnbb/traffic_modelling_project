import numpy as np

from other_functions import lower_diagonal_matrix

#set initial conditions
"""Initial conditions should specify the average velocity and density 
    in each cell"""
#set model size and precision

N = int(10) #number of cells
dx = 2. #size of a cell in meters
dt = 2. #time step
##set average velocity and density
v0 = np.array(N * [40e3*60])#initial uniform velocity of 40 km/h given in m/min
u = np.zeros((200,N)) # N cells with a max time of 200*dt
v_M = 120e3 * 60 #120 km/h
rho_M = 1 #2 lane highway and a car is 2 m long
u[0] = np.array(N * [0.2])
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
    M = np.eye(N) * (1 - c) + lower_diagonal_matrix(N) * c
    u_offset = np.append(u[N-1],u[0:-1])
    return M@u  +  (2*c/rho_M) * u**2  +  (-2*c/rho_M) * u * u_offset

print(u[0],propagate_one_step(u[0]))

#plot and analyse data
