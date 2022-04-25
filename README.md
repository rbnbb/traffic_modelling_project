# traffic_modelling_project

## Introduction
This repository is for a master of physics group homework on the topic of traffic modelling. It simulates traffic flow by way of a first order differential model using the Godunov integration scheme. It is meant to be used via the powerful class [TrafficModel](/traffic_model.py).

## Usage
Before direct usage running the [tests](/tests.py) and ensuring they are all passed is highly recommended.

The code can be used by importing the traffic_model module and creating an instance of the TrafficModel class.

### Calling the constructor
All constructor arguments are optional.
The notable constructor arguments are:
    - params - dictionary containing number of chunks N, size in meters of a chunk dx, integration time step in minutes dt, maximum speed in meters/minute v_M and maximum car density in cars/meter rho_M.  Default value is {'N': 200, 'dx': 50., 'dt': 0.02, 'v_M': 120e3 / 60, 'rho_M': 0.3}
    - ic_avg - average value of car density in cars/meter used for initial conditions
    - ic - specifies the type of initial conditions. It is a string that can take values 'uniform' (same value in all chunks), 'sin' (sinusoidal distribution around average value), 'normal' (normal distribution around average value with centroid in the center chunk)
    - bc - boundary conditions type, 'periodic' by default

### Running the simulation
Once a simulation is created in a class instance it can be propagated through time using the run method. It can be called by specifying the number of time steps or the time by which to advance the simulation.

### Exploiting the results
A default visualisation method is included in the class and it can be accessed via the member fig which is a matplotlib figure.
For other needs, the car density in all chunks at all times can be found in the class member u which is a 2D numpy array. One can access it and save it to a file and/or do a different visualisation.

## Examples of Use
The following is an excerpt from a interactive python session
>>> import traffic_model as tm  # importing module
>>> m = tm.TrafficModel()  # creating class instance
>>> print(m.params)  # view default parameters
{'N': 200, 'dx': 50.0, 'dt': 0.02, 'v_M': 2000.0, 'rho_M': 0.3}
>>> m.run(10)  # advancing simulation by 10 steps
>>> m.u  # car density array
array([[0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15],
       [0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15],
       [0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15],
       ...,
       [0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15],
       [0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15],
       [0.15, 0.15, 0.15, ..., 0.15, 0.15, 0.15]])
>>> m.u.shape  # size of the car density matrix
(11, 200)
>>> m.run(time=20)  # advancing simulation by 20 minutes
>>> m.u.shape
(1011, 200)
>>> m.fig.savefig("sample.png")  # save default visualisation as png image

## Technologies
Python 3
See requierments.txt for list of python packages used.
