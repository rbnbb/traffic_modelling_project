"""\
traffic_model
=============

Provides the powerful TrafficModel class for traffic simulations
based on continuous differential equations.
"""
import textwrap
import logging

import numpy as np
import matplotlib.pyplot as plt

from other_functions import lower_diagonal_matrix


logging.basicConfig(format='%(name)s:%(levelname)s:%(lineno)d:%(message)s')
logger = logging.getLogger(__name__)

class TrafficModel:
    """\
    Traffic Model
    ============

    Class representing an instance of a road simulation with initial
    all initial conditions and assumptions.
    """

    _all_initial_conditions = {
            'uniform': (lambda x, mu: mu),
            'sin': (lambda x, mu: mu*abs(np.sin(4*np.pi*x))),
            'normal': (lambda x, mu: mu*np.exp(-((x-.5)/.5)**2))}

    def __init__(self, params=None, ic=None, ic_avg=None, bc="periodic", visual=True):
        if params is None:
            params = {}
        self.__set_params(params)
        # define the array rho(x,t) with time the first index
        self.u = np.zeros((1, self.params['N']))
        # set initial conditions according to argument
        self.u[0] = self.__initial_conditions(ic, ic_avg)
        self.is_visual = visual
        if visual:
            self.__declare_graphical_objects()

    def propagate_one_step(self, u_in, bc='periodic', rho_in=None):
        """"\
        Returns u[n+1] array at time t+dt given initial u[n] array at time t.


        u - array giving the density in cars/m of all cells at previous time
        bc - string that specifies boundary conditions, periodic by default,
             if not periodic one should specify rho_in
        rho_in - incoming density taken as previous cell for the first cell

        The godunov scheme to be used is for the 1st order fundamental diagram:
        q(rho) = v_M *  rho(x,t) * (1-rho/rho_M).
        If u[n][i] = rho(i * dx, n * dt), then the following equation follows:

            u[n+1][i]  =  (1 - v_M*dt/dx) * u[n][i]  +  v_M*dt/dx * u[n][i-1]
                          +  2*v_M*dt/dx/rho_M * u[n][i] * u[n][i]
                          -  2*v_M*dt/dx/rho_M * u[n][i] * u[n][i-1]

        The first 2 terms are the result of a matrix multiplication and the
        next two of a product of the vector u[n] with itself.
        """
        # define a useful coefficient
        c = self.params['v_M'] * self.params['dt']/self.params['dx']
        M = np.eye(self.params['N']) * (1 - c)\
            + lower_diagonal_matrix(self.params['N']) * c
        # term to correct for boundary conditions, non 0 for non periodic bc
        bc_correction = np.zeros(self.params['N'])
        if bc == 'periodic':
            M[0][self.params['N']-1] = c
            u_offset = np.append(u_in[self.params['N']-1], u_in[0:-1])
        else:
            assert rho_in is not None, "Non periodic bc and rho_in is None!"
            u_offset = np.append(rho_in, u_in[0:-1])
            bc_correction[0] = c*rho_in
        c = 2 * c / self.params['rho_M']
        u_result = M@u_in + c * u_in**2 - c * u_in * u_offset + bc_correction
        assert np.all((u_result > 0)), 'The density should never be negative'
        assert np.all((u_result < self.params['rho_M'])),\
                'The density should never be bigger than rho_M.'
        return u_result

    def q_car_flow(self, rho):
        """Returns car flow q(x,t) as per fundamental diagram."""
        return self.params['v_M'] * rho * (1-rho/self.params['rho_M'])

    def run(self, steps=10, time=None):
        """Advance simulation by the given # of steps or time minutes."""
        # Should time be provided advance instead by time minutes
        if time is not None:
            steps = int(time / self.params['dt'])
        u_next = np.zeros((steps, self.params['N']))
        u_next[0] = self.propagate_one_step(self.u[-1])
        for i in range(1, steps):
            u_next[i] = self.propagate_one_step(u_next[i-1])
        self.u = np.vstack((self.u, u_next))
        if self.is_visual:
            self.plot()

    def plot(self):
        """Update TrafficModel class visualisation canvas."""
        # do a colorbar style plot of density rho(x,t)
        rho_map = self.axd['rho'].imshow(np.transpose(self.u), cmap="viridis")
        self.fig.colorbar(rho_map,
                          cax=self.axd['rho_cb'],
                          label="$\\rho(x,t)$ value")
        self.__update_legend()
        self.axd['rho'].set_ylim(0,self.params['N'])
        self.axd['rho'].set_aspect(0.7*self.u.shape[0]/self.u.shape[1])
        plt.draw()
        plt.show(block=False)

    def __declare_graphical_objects(self):
        """Declares a matplotlib fig and ax objects."""
        gs_kw = dict(width_ratios=[1, 8], height_ratios=[1])
        self.fig, self.axd = plt.subplot_mosaic([['legend', 'rho']],
                                                gridspec_kw=gs_kw,
                                                figsize=(16, 9),
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
            road length={self.params['N']*self.params['dx']/1e3:.0f} km
            $N={self.params['N']:.0f}$ cells
            $\\Delta x={self.params['dx']:.2f}$ meters
            $\\Delta t={self.params['dt']:.2f}$ minutes
            $v_M={self.params['v_M']:.2f}$ m/min
                ={self.params['v_M']*60/1000:.2f} km/h
            $\\rho_M={self.params['rho_M']:.2f}$ cars/m
            {self.u.shape[0]:.0f} time steps computed
            total time: {self.params['dt']*self.u.shape[0]:.0f} min""")
        self.axd['legend'].text(0, 0.8, message)

    def __set_params(self, params):
        default_params = {'N': 200,  # number of cells
                          'dx': 50.,  # size of a cell in m
                          'dt': 0.02,  # time step in min
                          'v_M': 120e3 / 60,  # v_M in m/min
                          'rho_M': 0.3}  # in cars/m
        self.params = default_params | params
        self.__check_params()

    def __check_params(self):
        """Performs sanity test to ensure good values."""
        assert isinstance(self.params['N'], int), "# cells must be an integer!"
        errmsg = "Incoherent dx value! Cell length scale must be large"\
                 " compared to vehicle size and smaller than road length"
        # set max dx to 1/5 of total road length
        dx_max = 0.2 * self.params['N'] * self.params['dx']
        assert 10 < self.params['dx'] < dx_max, errmsg
        errmsg = "Incoherent v_M value! Maximum speed must be positive"\
                 " and no more than 200 km/h"
        assert 0 < self.params['v_M'] < 200e3/60, errmsg
        errmsg = "Incoherent dt value! Time step should be positive and"\
                 " less than the time it takes a vehicle to travel cell"\
                 " length dx"
        dt_max = self.params['dx'] / self.params['v_M']
        assert 0 < self.params['dt'] < dt_max, errmsg

    def __initial_conditions(self, ic_type, ic_avg):
        """Returns np array of rho(x) at t=0 given string ic_type."""
        if ic_type is None:
            ic_type = 'uniform'
        try:
            func = self._all_initial_conditions[ic_type]
        except KeyError:
            logger.error("Initial conditions type must be among "\
                  f"{self._all_initial_conditions.keys()}. Defaulting "\
                  "to uniform...")
            func = self._all_initial_conditions['uniform']
        # set average rho halfway by default
        m =  0.5 * self.params['rho_M']
        if ic_avg is not None and 0.1<ic_avg<0.9:
            m = ic_avg * self.params['rho_M']

        xs = np.linspace(0.01, 1, self.params['N'])
        ic = np.array((lambda x: func(x, m))(xs))
        return ic
