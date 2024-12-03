'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

Running MLP Plots for Jordan & Siddiqi Quantum Measurement Book.
Code Written August & September 2022, Revised November & December 2024,
based on earlier codes from projects completed as a graduate student in AN Jordan's group: 
    arXiv:1612.07861, arXiv:1612.03189, arXiv:1803.07615

'''

import numpy as np
import QMTP_MLP_aux as AUX

# Divergence of the co-states for extremely rare readout sequences / post-selection boundary conditions is typical.
# The integrator used here is built to identify and remove such diverging paths from the final calculations / plots.  
np.seterr(over='ignore',invalid='ignore',divide='ignore') # consequently the internal generation of NaN is not of interest here.

# Langrangian Manifold: Driven Homodyne Flourescence

x0 = 0.0; z0 = 1.0; Qi = np.asarray([x0,z0]) # initial state
gam = 1.0; om = 2.0; eta = 0.5; eqargs = np.asarray([gam,om,eta]) # constants: decay rate, Rabi drive rate, measurement efficiency

Tf = 2.5

# range of initial "momenta", and momentum mesh resolution
pxmin = -5.0; pxmax = 5.0; Nx = 150 #400 # use for dense sampling
pzmin = -5.0; pzmax = 5.0; Nz = 150 #400
Pmesh = np.asarray([[pxmin,pxmax,Nx],[pzmin,pzmax,Nz]])

AUX.LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,'_sparse_QMTheoryPractice')
#AUX.LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,'_dense_QMTheoryPractice')

# Stroboscopic Phase Portrait: X Measurement with kicked Z Measurement

prange = np.pi+0.5 # cut the mesh a bit above the first large resonance
Np = 20 #50 #use larger values for dense mesh
Nth = 20 #50

off = 1.0 # base measurement strengths
amp = 0.96 # amplitude of the z-measurement kick (tau ~ off - amp). 
# Use any value in [0,1), where 0 is trivially integrable, and a transition to chaos occurs as amp increases
sig = 0.01 # width of the z-measurement kick
eqargs_xz = np.asarray([off, amp, sig]) 

Tf = 100.0 # final integration time

dt_plot = 1.0
dt_comp_factor = 1000

dth0 = 0.01

AUX.StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,'_sparse_QMTheoryPractice_k96_T100_N20')
#AUX.StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,'_dense_QMTheoryPractice_k96_T100_N50')