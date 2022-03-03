import textwrap

import numpy as np
import matplotlib.pyplot as plt

from other_functions import lower_diagonal_matrix


class TrafficModel:
    def __init__(self):
        #set model size and precision
        self.N = 50 #number of cells
        self.dx = 50. #size of a cell in meters
        self.dt = 0.02 #time step
        self.u = np.zeros((1,self.N)) # N cells 
        self.v_M = 120e3 / 60 #120 km/h in m/min
        self.rho_M = 0.3 #in cars/m
        self.u[0] = np.array(self.N * [0.1])
        self.__declare_graphical_objects()

    def propagate_one_step(self, u_in , bc='periodic', rho_in=None):
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
        c = self.v_M * self.dt/self.dx #much used coeff
        M = np.eye(self.N) * (1 - c) + lower_diagonal_matrix(self.N) * c
        bc_correction = np.zeros(self.N) # non zero for non periodic bc
        if bc == 'periodic':
            M[0][self.N-1]=c
            u_offset = np.append(u_in[self.N-1],u_in[0:-1])
        else:
            assert rho_in is not None, 'For non periodic bc rho_in must be given!'
            u_offset = np.append(rho_in,u_in[0:-1])
            bc_correction[0] = c*rho_in
        u_result = M@u_in  +  (2*c/self.rho_M) * u_in**2  +\
                   (-2*c/self.rho_M) * u_in * u_offset + bc_correction
        assert np.all((u_result > 0)), 'The density should never be negative'
        return u_result

    def q_car_flow(self, rho):
        """Returns car flow q(x,t) as per fundamental diagram."""
        return self.v_M * rho * (1-rho/self.rho_M)

    def run(self, steps=10):
        """Advance simulation by the requested number of steps."""
        u_next = np.zeros((steps,self.N))
        u_next[0] = self.propagate_one_step(self.u[-1])
        for i in range(1,steps):
            u_next[i] = self.propagate_one_step(u_next[i-1])
        self.u = np.vstack((self.u, u_next))
        self.plot()

    def plot(self):
        """Update TrafficModel class visualisation canvas."""
        # do a colorbar style plot of density rho(x,t)
        rho_map = self.axd['rho'].imshow(np.transpose(self.u), cmap="viridis")
        self.fig.colorbar(rho_map,
                                          cax=self.axd['rho_cb'],
                                          label="$\\rho(x,t)$ value")
        self.__update_legend()
        plt.show()

    def __declare_graphical_objects(self):
        """Declares a matplotlib fig and axd objects."""
        gs_kw = dict(width_ratios=[1, 8], height_ratios=[1])
        self.fig, self.axd = plt.subplot_mosaic([['legend','rho']],
                                                gridspec_kw=gs_kw,
                                                figsize=(16,9),
                                                constrained_layout=True)
        rho_map = self.axd['rho'].imshow(np.transpose(self.u), cmap="viridis")
        self.colorbar = self.fig.colorbar(rho_map,
                                          ax=self.axd['rho'],
                                          location='right')
        self.axd['rho_cb'] = self.colorbar.ax
        self.__update_legend()
        self.axd['rho'].set_title(r"Traffic density $\rho(x,t)$")
        self.axd['rho'].set_xlabel("# time step")
        self.axd['rho'].set_ylabel("# cell")

    def __update_legend(self):
        self.axd['legend'].clear()
        self.axd['legend'].axis('off')
        self.axd['legend'].set_title("Parameters")
        message = textwrap.dedent(f"""\
            $N={self.N:.0f}$ cells
            $\\Delta x={self.dx:.2f}$ meters
            $\\Delta t={self.dt:.2f}$ minutes
            $v_M={self.v_M:.2f}$ m/min
                ={self.v_M*60/1000:.2f} km/h
            $\\rho_M={self.rho_M:.2f}$ cars/m
            {self.u.shape[0]:.0f} time steps computed""")
        print(message)
        self.axd['legend'].text(0, 0.8, message)
