import numpy as np

#set initial conditions
"""Initial conditions should specify the average velocity and density in each cell"""
#set model size and precision

N = int(10) #number of cells
dx = 2. #size of a cell in meters
dt = 2. #time step
##set average velocity and density
v0 = np.array(N*[40e3*60]) #initial uniform velocity of 40 km/h expressed in meters per minute
u = np.zeros((200,N)) # N cells with a max time of 200*dt
v_M = 120e3*60 #120 km/h
rho_M = 1 #2 lane highway and a car is 2 m long
u[0] = np.array(N*[0.2])
#solve equations 


""""The godunov scheme to be implemented is for the 1st order fundamental diagram:
q(rho) = v_M *  rho(x,y) * (1-rho/rho_M).

If u[n][i] = rho(i * dx, n * dt) it leads to the following propagation equation:
    u[n+1][i]  =  (1 - v_M*dt/dx) * u[n][i]  +  v_M*dt/dx * u[n][i-1]  + 
                  + 2*v_M*dt/dx/rho_M * u[n][i] * u[n][i]   
                  -  2*v_M*dt/dx/rho_M * u[n][i] * u[n][i-1]

The first 2 terms are the result of a matrix multiplication and the next two
of a product of the vector u[n] with itself.
"""

def lower_diagonal_matrix(n):
    """Returns a lower diagonal matrix of size n x n.
        

       For a 3x3 matrix this would look like:
                        0 0 0
                        1 0 0
                        0 1 0
    """
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if(i-1==j):
                M[i][j] = 1
    return M

moff = lower_diagonal_matrix(N)
M = np.eye(N)*(1-v_M*dt/dx)+moff*v_M*dt/dx
u_offset = np.append(u[0][N-1],u[0][0:-1])
t1 = (2*v_M/rho_M*dt/dx)*u[0]**2
t2 = (-v_M*2/rho_M*dt/dx)*(u[0]*u_offset)
u[1] = M@u[0]+t1+t2
print(u[0],u[1])

#plot and analyse data
