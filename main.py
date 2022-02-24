import numpy as np
import matplotlib.pyplot as plt

from other_functions import lower_diagonal_matrix

#set initial conditions
"""Initial conditions should specify the average velocity and density 
    in each cell"""
#set model size and precision
M = 200
N = 50 #number of cells
dx = 50. #size of a cell in meters
dt = 0.02 #time step
##set average velocity and density
v0 = np.array(N * [40e3*60])#initial uniform velocity of 40 km/h given in m/min
u = np.zeros((M,N)) # N cells with a max time of 200*dt
v_M = 120e3 / 60 #120 km/h in m/min
rho_M = 0.3 #in cars/m
u[0] = np.array(N * [0.1])
#u[0][0]=0.08

#ccheck nu,bers





#solve equations 



def propagate_one_step(u, bc='periodic', rho_in=None):
    """"Returns u[n+1] array at time t+dt given initial u[n] array at time t.
    
  
    u - array giving the density in cars/m of all cells at previous time
    bc - string that specifies boundary conditions, periodic by default,
         if not periodic one should specify rho_in
    rho_in - incoming density taken as previous cell for the first cell

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
    bc_correction = np.zeros(N) # non zero for non periodic bc
    if(bc == 'periodic'):
        M[0][N-1]=c 
        u_offset = np.append(u[N-1],u[0:-1])
    else:
        assert rho_in is not None, 'For non periodic bc rho_in must be given!'
        u_offset = np.append(rho_in,u[0:-1])
        bc_correction[0] = c*rho_in
    u_result = M@u  +  (2*c/rho_M) * u**2  +  (-2*c/rho_M) * u * u_offset + bc_correction
    assert np.all((u_result > 0)), 'The density should never be negative'
    return u_result


#sinusoidal bc modulation
bcs = abs(np.sin(np.linspace(0.2,20,M))/8)
print('bcs ',bcs)


for i in range(1,u.shape[0]):
    u[i]=propagate_one_step(u[i-1],'sinusoidal',bcs[i-1])
print("\nu matrix:\n",u)

#check density is conserved
print(f"SUM at subsequent tymes is {u.sum(axis=1)}")

#plot and analyse data

#declare 
fig, ax = plt.subplots(1,1,figsize=(6,4))

#do a colorbar style plot of density
data_visual = ax.imshow(np.transpose(u), cmap="viridis")
fig.colorbar(data_visual, ax=ax, location='right')
plt.show()
