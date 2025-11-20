import numpy as np
# using atomic unit
kT=0.0
hbar=1
c0=137.036                  # speed of light in au
epsilon=1/(4*np.pi)         # vacuum permittivity in atomic unit

# Cavity setup
l=2.362E5                   # cavity length
alpha=np.arange(1,400+1,1)  # cavity mode
N = alpha.size
w=(np.pi * c0 * alpha)/l    # cavity mode frequency

# Atom setup
num_atom=1
energy=np.array([-0.6738, -0.2798])
mu12=1.034
ini_wavefn=np.array([0, 1])
r_atom = np.array([l/2])

# running time setup
tmax=1000
dt=0.01
savestep=1000

proc = 1                       # number of processors
num_trj = 1                      # for each processor. Recommend 10 to 100
total_trj = 1           # total number of trajectories. total_trj/(proc*num_trj) will define the number of runs and should be a integer.
calcdir = 'data'
status = 'NORESTART'  #'NORESTART' or 'RESTART'

r_resolution=1000
intens_save_t=100