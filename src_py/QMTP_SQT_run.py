'''
Philippe Lewalle

AN Jordan Group at University of Rochester (2014-2021)
KB Whaley Group at UC Berkeley (2021-current)

Running Two-Qubit Entanglement Plots for Jordan & Siddiqi Quantum Measurement Book.
This script creates figures like Fig. 9.10 of the textbook.

Code Written December 2024, based on older codes written as a graduate student in Andrew Jordan's group.  
Python 3 code update: Stochastic Quantum Trajectory Simulations Applied to Two-Qubit Entanglement Generation via Measurement
See Advances in Optics and Photonics, 13 (3), 517-583 (2021), and/or Phys. Rev. A 102 062219 (2020) for initial related publications. 

Run this script to execute the methods included in other files.
'''
# code tested in python 3.12.7 
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
import QMTP_SQT_aux as aux

# initial state is |ee}
rh0 = np.zeros((4,4)) + 1.0j*np.zeros((4,4))
rh0[0,0] = 1.0

dt = 1.0e-3 # 1.0e-4
dtsf = 10 # 200
# save every 10th computed point for actual plotting; this is primarily to save memory

gam = 1.0 
T = 5.0

N_SQT = 12
img_name = 'sparse'
ctp, c_rho_jump = aux.SIM_jump_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,concplot = True,avgconc = False)
ctp, c_rho_homk = aux.SIM_hom_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,concplot = True,avgconc = False,method = 'Kraus')
ctp, c_rho_homr = aux.SIM_hom_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,concplot = True,avgconc = False,method = 'Rouchon')

N_SQT = 6000
img_name = 'dense_avg'
ctp, c_rho = aux.SIM_jump_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,et3 = 1.0,et4 = 1.0,concplot = True,avgconc = True,qpanels=True)
ctp, c_rho_homk = aux.SIM_hom_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,concplot = True,avgconc = True,method = 'Kraus',qpanels=True)
#ctp, c_rho_homr = aux.SIM_hom_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,concplot = True,avgconc = True,method = 'Rouchon',qpanels=True)
