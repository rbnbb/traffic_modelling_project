import numpy as np

#set initial conditions
"""Initial conditions should specify the average velocity and density in each cell"""
#set model size and precision

N=int(10) #number of cells
dx=2. #size of a cell in meters
dt=2. #time step
##set average velocity and density
v0=np.array(N*[40e3*60]) #initial uniform velocity of 40 km/h expressed in meters per minute
u=np.zeros((200,N)) # N cells with a max time of 200*dt
v_M=120e3*60 #120 km/h
rho_M=1 #2 lane highway and a car is 2 m long
u[0]=np.array(N*[0.2])
#solve equations 

#define off diagonal 1 everywhere matrix
moff=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if(i-1==j):
            moff[i][j]=1
M=np.eye(N)*(1-v_M*dt/dx)+moff*v_M*dt/dx
u_offset=np.append(u[0][N-1],u[0][0:-1])
t1=(2*v_M/rho_M*dt/dx)*u[0]**2
t2=(-v_M*2/rho_M*dt/dx)*(u[0]*u_offset)
u[1]=M@u[0]+t1+t2
print(u[0],u[1])

#plot and analyse data
