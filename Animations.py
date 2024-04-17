# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:10:08 2024

@author: mmanc
"""

# =============================================================================
# Animated Plots
# =============================================================================
# =============================================================================
# Perihelian/Elliptical orbits
# We first need to solve the following ode using numerical integration. 
# =============================================================================
# Using RK4 Method
# =============================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%%
c = 1 # set speed of light in a vacuum equal to one
rs = 1 # schwarzchild radius 
J = 2 # angular momentum
E = 0.92999 # total energy #set to 0.92999


#starting values
phi0 = 0 # initial value for tau
r0 = 4.673*rs # initial value for r. set it to r0 = 4.673*rs
#r0 = 5*rs # change radius to 5rs to see a smaller shift

# now we are going to define f(y,t) our ODE

def f(phi, r):
    return (r**2/J) * np.sqrt(E - (1 - rs/r)*(c**2 + (J/r)**2))# remember this is plus or minus

# Now we are going to set few lists:

phi_values = [] # will collect all the values of tau
r_value = [] # will collect all the values of r at proper time tau

phi_old = phi0
r_old = r0 # used for iterating procedure 

phi_values.append(phi_old)
r_value.append(r_old) # appending old values to lists

# let delta in the RK-4 method be rappresented by h, the smaller
# we chose h to be the lower will the error be (h is the step size):
h = 0.01 # chose h = 0.01 for accurate result
phi_end  = 100 # end of t values 

Nsteps = np.int(np.floor((phi_end-phi0)/h)) # this will allow us to define a range in 
# our for loop
# now we can start our loop
for i in range(0, Nsteps):
    
    phi_now = phi_old + h
    
    k1 = f(phi_old,r_old)
    k2 = f(phi_old + 0.5*h, r_old + 0.5*k1*h)
    k3 = f(phi_old + 0.5*h, r_old + 0.5*k2*h)
    k4 = f(phi_old + h, r_old + k3*h)
    
    r_now = r_old + h*(k1 + 2*k2 + 2*k3 + k4)/6
    
    phi_values.append(phi_now)
    r_value.append(r_now)
    
    phi_old = phi_now
    r_old = r_now
    
r_values = [x for x in r_value if not math.isnan(x)]


r_inverse = r_values[::-1]
r_reverse = r_inverse[1:-1]

rs1 = r_values + r_reverse + r_values + r_reverse + r_values
rs2 = r_reverse + r_values + r_reverse + r_values + r_reverse

# for animation animation
rs0 = rs1 + rs2 

a = phi_values[-1]
a_list = []
iterations = abs(len(rs0)-len(phi_values))

for _ in range(iterations):
    
    a += 0.01
    
    a_list.append(a)

phis = phi_values + a_list


if len(rs0) == len(phis):
    phi = phis
else:
    m = abs(len(rs0) - len(phis))
    phi = phis[:-m]
    
xs = []
ys = []

for i in range(len(rs0)):
    x = rs0[i]*np.cos(phi[i])
    y = rs0[i]*np.sin(phi[i])
    xs.append(x)
    ys.append(y)

# Define the figure and axis
fig, ax = plt.subplots()

# Define the circle
rs = 1
circle = plt.Circle((0, 0), rs, color='black', alpha=0.5, label=r'Static black hole of radius $R_s = 1$')
ax.add_artist(circle)

# Plot the trajectory in the xy-plane

line, = ax.plot([], [], label='perihelion orbit', color = 'black')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_ylabel(r'$\frac{y}{R_s}$')
ax.set_xlabel(r'$\frac{x}{R_s}$')
ax.set_title('Perihelion shift in the proximity of a static black hole the xy-plane (RK4)')
#ax.scatter(0, 0, label = 'Particle', color = 'red', s=0)

particle, = ax.plot([], [], 'ro', label = 'Particle')  # red dot for the particle

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to animate the plot
def animate(frame):
    x_data = xs[:frame]
    y_data = ys[:frame]
    line.set_data(x_data, y_data)
    # Update the position of the particle
    particle_x = xs[frame - 1] if frame > 0 else np.nan  # previous x coordinate
    particle_y = ys[frame - 1] if frame > 0 else np.nan  # previous y coordinate
    particle.set_data(particle_x, particle_y)
    return line, particle, 

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=0, blit=True)

ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

plt.show()
#%%
# =============================================================================
# Using EinsteinPy
# =============================================================================
import numpy as np 

from einsteinpy.geodesic import Timelike
from einsteinpy.plotting.geodesic import GeodesicPlotter, StaticGeodesicPlotter, InteractiveGeodesicPlotter

r0 = 9.346
r1 = 12 #circular orbit
r2 = 6

X = [r0, np.pi/2 , 0] #position vector, set initial position radial value as 9.346, unstable circular orbit
# at r = 4, stable circular orbit at r = 12
P = [0., 0., 4] #momentum set p_phi = 4

a = 0.

steps = 2500 #good plot set steps = 1000, 26 steps for unstable circular orbit and 227 for stable

delta = 1.

geod = Timelike(metric = 'Schwarzschild', metric_params=(a,), position = X, momentum = P, steps = steps,
                delta = delta, return_cartesian = True)

gpl = GeodesicPlotter(ax=None, bh_colors=('#000', '#FFC'), draw_ergosphere=False)

trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
iterations = []
for i in range(0,steps):
    x1 = trajectory[i][1] # X1 values
    x2 = trajectory[i][2] # X2 values 
    ite = i # keep the iteartions
    x1_list.append(x1)
    x2_list.append(x2)
    iterations.append(ite)

fig, ax = plt.subplots()

# Define the circle
rs = 2
circle = plt.Circle((0, 0), rs, color='black', alpha=0.5, label=r'Static black hole of radius $R_s = 2$')
ax.add_artist(circle)

# Plot the trajectory in the xy-plane

line, = ax.plot([], [], label='perihelion orbit', color = 'black')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
lim = 20
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_ylabel(r'$\frac{y}{R_s}$')
ax.set_xlabel(r'$\frac{x}{R_s}$')
ax.set_title('Perihelion shift in the proximity of a static black hole the xy-plane (EinsteinPy)')
#ax.scatter(0, 0, label = 'Particle', color = 'red', s=0)

particle, = ax.plot([], [], 'ro', label = 'Particle')  # red dot for the particle

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to animate the plot
def animate(frame):
    x_data = x1_list[:frame]
    y_data = x2_list[:frame]
    line.set_data(x_data, y_data)
    # Update the position of the particle
    particle_x = x1_list[frame - 1] if frame > 0 else np.nan  # previous x coordinate
    particle_y = x2_list[frame - 1] if frame > 0 else np.nan  # previous y coordinate
    particle.set_data(particle_x, particle_y)
    return line, particle, 

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(x1_list), interval=0, blit=True)

ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

plt.show()
#%%
# =============================================================================
# Animation of the carr black hole for different values of a
# =============================================================================
Rs = 2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def ergosphere(r, a, theta):
    return (Rs + np.sqrt(Rs**2 - 4*a**2*np.cos(theta)**2))/2
def horizon(r, a):
    return (r + np.sqrt(r**2 - 4*a**3))/2

def kerr_black_hole(a):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r_bh = horizon(Rs, a)
    r_ergo = ergosphere(Rs, a, phi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x_ergo = r_ergo * np.sin(theta) * np.cos(phi)
    y_ergo = r_ergo * np.sin(theta) * np.sin(phi)
    z_ergo = r_ergo * np.cos(theta)
    
    x_bh = r_bh * np.sin(theta) * np.cos(phi)
    y_bh = r_bh * np.sin(theta) * np.sin(phi)
    z_bh = r_bh * np.cos(theta)
    
    return x_ergo, y_ergo, z_ergo, x_bh, y_bh, z_bh

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update_plot(a):
    ax.clear()
    ax.set_title(f'Kerr Black Hole (a={a:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([], [], [], 'o',color = 'gray', label='Ergosphere', alpha = 0.5)
    ax.plot([], [], [], 'ko', label='Outer horizon', alpha = 0.8)

    Xe, Ye, Ze, Xh, Yh, Zh = kerr_black_hole(a)
    ax.plot_surface(Xe, Ye, Ze, color='gray', alpha=0.4)
    ax.plot_surface(Xh, Yh, Zh, color='black', alpha=0.8)
    ax.legend()
    

a = np.arange(0.1, 1.0, 0.01)
ani = FuncAnimation(fig, update_plot, frames= a, interval=100)
plt.show()
#%%
import numpy as np
import time
# =============================================================================
# Timelike
# =============================================================================
from einsteinpy.geodesic import Timelike
from einsteinpy.plotting import StaticGeodesicPlotter, GeodesicPlotter

start_time = time.time()

b = 0.767851
# Constant Radius Orbit
position = [4, np.pi/ 3, 0.] # [4, np.pi / 3, 0.] original
momentum = [0, b, 2.] # [0., 0.767851, 2.]
a = 0.99 # 0.99
steps = 400 # set to 10000 for pi/4
delta = 0.5 #set to 0.5 by default
omega = 1
suppress_warnings = True

geod = Timelike(
    metric="Kerr",
    metric_params = (a,),
    position=position,
    momentum=momentum,
    steps=steps,
    delta=delta,
    return_cartesian=True,
)

end_time = time.time()
ellapse_time = end_time - start_time

print('The ellapse time is', ellapse_time, 'seconds')

trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,steps):
    x1 = trajectory[i][1] # X1 values
    x2 = trajectory[i][2] # X2 values
    x3 = trajectory[i][3]
    ite = i # keep the iteartions
    x1_list.append(x1)
    x2_list.append(x2)
    x3_list.append(x3)
    iterations.append(ite)

Rs = 2

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Assuming x1_list, x2_list, and x3_list are already defined

skip_frames = 1
def ergosphere(r, a, theta):
    return (Rs + np.sqrt(Rs**2 - 4*a**2*np.cos(theta)**2))/2
def horizon(r, a):
    return (r + np.sqrt(r**2 - 4*a**3))/2

fig = plt.figure(dpi=100)
ax = fig.add_subplot(projection='3d')

# Initialize the scatter plot for the particle
particle = ax.scatter([], [], [], marker='o', label='Particle', color='red')

# Initialize the orbit plot
orbit, = ax.plot([], [], [], color='black', alpha=0.6, label='Orbit')

def update(frame):
    ax.cla()
    particle_x = x1_list[frame - 1] if frame > 0 else np.nan  # previous x coordinate
    particle_y = x2_list[frame - 1] if frame > 0 else np.nan  # previous y coordinate
    particle_z = x3_list[frame - 1] if frame > 0 else np.nan  # previous x coordinate
    
    ax.scatter(particle_x, particle_y, particle_z, marker='o', label='Particle', color='red')
    ax.plot3D(x1_list[:frame], x2_list[:frame], x3_list[:frame], color='black', alpha = 0.6, label = 'Orbit')
    
    lim = 5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r_bh = horizon(Rs, a)
    r_ergo = ergosphere(Rs, a, phi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x_ergo = r_ergo * np.sin(theta) * np.cos(phi)
    y_ergo = r_ergo * np.sin(theta) * np.sin(phi)
    z_ergo = r_ergo * np.cos(theta)
    
    x_bh = r_bh * np.sin(theta) * np.cos(phi)
    y_bh = r_bh * np.sin(theta) * np.sin(phi)
    z_bh = r_bh * np.cos(theta)
    
    
    # Plot ergosphere
    ax.plot_surface(x_ergo, y_ergo, z_ergo, color='gray', alpha=0.5)
    ax.plot([], [], [], 'o',color = 'gray', label='Ergosphere', alpha = 0.5)
    # Plot event horizon
    ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha = 0.5)
    ax.plot([], [], [], 'ko', label='Outer horizon', alpha = 0.5)

    ax.set_title(f" Animated 3D plot of time-like geodisic, Kerr balck hole (a = {a})")

    ax.set_xlabel(r'$\frac{X}{R_s}$', labelpad=20)
    ax.set_ylabel(r'$\frac{Y}{R_s}$', labelpad=20)
    ax.set_zlabel(r'$\frac{Z}{R_s}$', labelpad=20)
    
    ax.legend()


fps = 500
ani = FuncAnimation(fig=fig, func=update, frames = len(x1_list), interval = 1/fps,repeat=True)

plt.show()
#%%








