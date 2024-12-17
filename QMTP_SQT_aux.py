'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

code created December 2024, based on older codes (2019-2021)

Python 3 code update: Stochastic Quantum Trajectory Simulations Applied to Two-Qubit Entanglement Generation via Measurement
See Advances in Optics and Photonics, 13 (3), 517-583 (2021), and/or Phys. Rev. A 102 062219 (2020) for initial related publications. 

This file contains the main computational methods, but does not execute them. 
It is paired with a script which loads and runs these methods for specific values.
'''
# code tested in python 3.9.20 
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
import numba as nb #                                               numba 0.60.0
from numpy.random import normal as gauss
from numpy.random import multinomial as nomnoms
import SQT_qubits_aux as sqt

from matplotlib import pyplot as pl #                          matplotlib 3.9.2
from cycler import cycler # used for color selection in plotting

'''
CONTENTS

SEC. 1: Kraus Operators and Trajectory Simulations ........................ 045
    a. Kraus operators for Jump trajectories (photodetection) ............. 055
    b. Jump Probabilities and Sampling .................................... 150
    c. Jump Trajectory Integration ........................................ 197
    d. Kraus operators for Diffusive trajectories (homodyne) .............. 257
    e. Homodyne Readout Probability / Sampling ............................ 315
    f. Homodyne Trajectory Integration .................................... 346
    
SEC. 2: Some Plotting Functions for Two-Qubit Problems  ................... 389
    a. Jump Trajectory Ensemble Simulator (Kraus Operator Only) ........... 399
    b. Diffusive Ensemble Simulator (Kraus or Rouchon/Ralph) .............. 434
    c. Concurrence Plotter ................................................ 479
    d. Two-Qubit Density Matrix Visualization ............................. 531

'''

''' 111111111111111111111111111111111111111111111111111111111111111111111111111
Section 1: Definition of Two-Qubit Fluorescence Measurement Operators

    See especially appendix D of https://doi.org/10.1364/AOP.399081

111111111111111111111111111111111111111111111111111111111111111111111111111 '''

@nb.njit(inline='always')
def DAG(op):
    return np.conjugate(np.transpose(op))

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Definition of the Jump-Trajectory Kraus Operators '''

@nb.njit(inline='always')
def KM_0000(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[0,0] = 1.0-ep
    mat[1,1] = np.sqrt(1.0-ep)
    mat[2,2] = np.sqrt(1.0-ep)
    mat[3,3] = 1.0
    return mat

@nb.njit(inline='always')
def KM_1000(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[1,0] = np.sqrt(ep*(1.0-ep)*et3/2)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*et3/2)
    mat[3,1] = np.sqrt(ep*et3/2)
    mat[3,2] = np.sqrt(ep*et3/2)
    return mat

@nb.njit(inline='always')
def KM_0100(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[1,0] = -np.sqrt(ep*(1.0-ep)*et4/2)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*et4/2)
    mat[3,1] = np.sqrt(ep*et4/2)
    mat[3,2] = -np.sqrt(ep*et4/2)
    return mat

@nb.njit(inline='always')
def KM_0010(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[1,0] = np.sqrt(ep*(1.0-ep)*(1.0-et3)/2)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*(1.0-et3)/2)
    mat[3,1] = np.sqrt(ep*(1.0-et3)/2)
    mat[3,2] = np.sqrt(ep*(1.0-et3)/2)
    return mat

@nb.njit(inline='always')
def KM_0001(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[1,0] = -np.sqrt(ep*(1.0-ep)*(1.0-et4)/2)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*(1.0-et4)/2)
    mat[3,1] = np.sqrt(ep*(1.0-et4)/2)
    mat[3,2] = -np.sqrt(ep*(1.0-et4)/2)
    return mat

@nb.njit(inline='always')
def KM_1010(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = ep*np.sqrt(et3*(1.0-et3))
    return mat

@nb.njit(inline='always')
def KM_0101(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = -ep*np.sqrt(et4*(1.0-et4))
    return mat

@nb.njit(inline='always')
def KM_2000(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = ep*et3/np.sqrt(2)
    return mat

@nb.njit(inline='always')
def KM_0200(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = -ep*et4/np.sqrt(2)
    return mat

@nb.njit(inline='always')
def KM_0020(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = ep*(1.0-et3)/np.sqrt(2)
    return mat

@nb.njit(inline='always')
def KM_0002(gam,dt,et3,et4):
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    ep = gam*dt
    mat[3,0] = -ep*(1.0-et4)/np.sqrt(2)
    return mat

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Jump Probabilities and Sampling '''

@nb.njit(inline='always')
def WP_00(rho,gam,dt,et3,et4): # probability of no jump
    m0000 = KM_0000(gam,dt,et3,et4)
    m0010 = KM_0010(gam,dt,et3,et4)
    m0001 = KM_0001(gam,dt,et3,et4)
    m0020 = KM_0020(gam,dt,et3,et4)
    m0002 = KM_0002(gam,dt,et3,et4)
    M2TOT = DAG(m0000) @ m0000 + DAG(m0010) @ m0010 + DAG(m0001) @ m0001 + DAG(m0020) @ m0020 + DAG(m0002) @ m0002
    return np.real(np.trace(rho @ M2TOT))

@nb.njit(inline='always')
def WP_10(rho,gam,dt,et3,et4): # probability of a click at out 1 (port 3)
    m1000 = KM_1000(gam,dt,et3,et4)
    m1010 = KM_1010(gam,dt,et3,et4)
    M2TOT = DAG(m1000) @ m1000 + DAG(m1010) @ m1010
    return np.real(np.trace(rho @ M2TOT))

@nb.njit(inline='always')
def WP_01(rho,gam,dt,et3,et4): # probability of a click at out 2 (port 4)
    m0100 = KM_0100(gam,dt,et3,et4)
    m0101 = KM_0101(gam,dt,et3,et4)
    M2TOT = DAG(m0100)@m0100 + DAG(m0101)@m0101
    return np.real(np.trace(rho @ M2TOT))

@nb.njit(inline='always')
def WP_20(rho,gam,dt,et3,et4): # probability of a double click at out 1 (port 3)
    return np.real(np.trace(KM_2000(gam,dt,et3,et4) @ rho @ DAG(KM_2000(gam,dt,et3,et4))))

@nb.njit(inline='always')
def WP_02(rho,gam,dt,et3,et4): # probability of a double click at out 2 (port 4)
    return np.real(np.trace(KM_0200(gam,dt,et3,et4) @ rho @ DAG(KM_0200(gam,dt,et3,et4))))

# readout sampling: which photodetector clicks?
def JUMP_eta_PRO(rho,dt,gam,et3,et4):
    w00 = WP_00(rho,dt,gam,et3,et4)
    w10 = WP_10(rho,dt,gam,et3,et4)
    w01 = WP_01(rho,dt,gam,et3,et4)
    w20 = WP_20(rho,dt,gam,et3,et4)
    w02 = WP_02(rho,dt,gam,et3,et4)
    # acquire the stochastic readout (make a multinomial choice between these options, according to their probability)
    wlist = [w00,w10,w01,w20,w02] # list probabilities for the different possible click outcomes
    ro_click = nomnoms(1,wlist,size=1)[0] 
    return ro_click

''' ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
Jump Trajectory Integration '''

# take a step, updating the density matrix, for a photodetection trajectory
def JUMP_eta_step(rho,dt,gam,et3,et4,rro=False):
    ro = JUMP_eta_PRO(rho,dt,gam,et3,et4) # get a readout
    # update the state accordingly
    if ro[0] == 1: # a click was not registered
        M0 = KM_0000(gam,dt,et3,et4)
        M1 = KM_0010(gam,dt,et3,et4)
        M2 = KM_0001(gam,dt,et3,et4)
        M3 = KM_0020(gam,dt,et3,et4)
        M4 = KM_0002(gam,dt,et3,et4)
        numer = M0 @ rho @ DAG(M0) + M1 @ rho @ DAG(M1) + M2 @ rho @ DAG(M2) + M3 @ rho @ DAG(M3) + M4 @ rho @ DAG(M4)
    elif ro[1] == 1: # one click was registered at port 3
        M0 = KM_1000(gam,dt,et3,et4)
        M1 = KM_1010(gam,dt,et3,et4)
        numer = M0 @ rho @ DAG(M0) + M1 @ rho @ DAG(M1)
    elif ro[2] == 1: # one click was registered at port 4
        M0 = KM_0100(gam,dt,et3,et4)
        M1 = KM_0101(gam,dt,et3,et4)
        numer = M0 @ rho @ DAG(M0) + M1 @ rho @ DAG(M1)
    elif ro[3] == 1: # two clicks were registered at port 3 
        M = KM_2000(gam,dt,et3,et4)
        numer = M @ rho @ DAG(M)
    elif ro[4] == 1: # two clicks were registered at port 4
        M = KM_0200(gam,dt,et3,et4)
        numer = M @ rho @ DAG(M)
    else: # something went wrong
        numer = 0.25*np.ones((4,4)) + 1.0j*np.zeros((4,4))
        print('Error obtaining readout from rho = '+str(rho)+str(ro))
    denom = np.trace(numer) # normalization of new rho
    if rro == True:
        return numer/denom, np.asarray(ro) # return the updated density matrix and the click record
    else:
        return numer/denom # return the updated density matrix only

# run a single jump trajectory (photodetection at both outputs after BS)
def JUMP_eta_traj(rh0,dt,dtsf,gam,T,et3,et4,rro = False): # rh0 is initial density matrix.
    dtp = dt*dtsf # dtsf is factor of saved time points. Used to reduce memory use for large ensembles
    Ntp = np.rint(T/dtp).astype(np.int64)+1 # number of timesteps saved
    Nt = np.rint(T/dt).astype(np.int64)+1 # number of timesteps run
    ctp = np.linspace(0,T,Ntp) # initialize time array
    c_rho = np.zeros((Ntp,4,4)) + 1.0j*np.zeros((Ntp,4,4)) # initialize array for the state coordinates
    c_rho[0,:,:] = rh0
    if rro == True: # only use this for debugging
        ct = np.linspace(0,T,Nt) # initialize time array
        cr = np.zeros((Nt-1,5)) # in order to save the measurement record
        for k in range(1,Nt):
            rh0, cr[k-1,:] = JUMP_eta_step(rh0,dt,gam,et3,et4,rro = True)
            if k%dtsf == 0: # save a point
                c_rho[np.rint(k/dtsf).astype(np.int64),:,:] = rh0
        return ctp, c_rho, ct, cr # return the results
    else:
        for k in range(1,Nt):
            rh0 = JUMP_eta_step(rh0,dt,gam,et3,et4)
            if k%dtsf == 0: # save a point
                c_rho[np.rint(k/dtsf).astype(np.int64),:,:] = rh0
        return ctp, c_rho # return the results

''' ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
Definition of the Diffusive-Trajectory Kraus Operators '''

@nb.njit(inline='always')
def KM_XY00(r3,r4,dt,gam,the,vth,et3,et4):
    X = r3*np.sqrt(dt/2)
    Y = r4*np.sqrt(dt/2)
    ep = dt*gam
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    mat[0,0] = 1.0-ep
    mat[1,0] = np.sqrt(ep*(1.0-ep))*(np.sqrt(et3)*X*np.exp(1.0j*the)-np.sqrt(et4)*Y*np.exp(1.0j*vth))
    mat[1,1] = np.sqrt(1.0-ep)
    mat[2,0] = np.sqrt(ep*(1.0-ep))*(np.sqrt(et3)*X*np.exp(1.0j*the)+np.sqrt(et4)*Y*np.exp(1.0j*vth))
    mat[2,2] = np.sqrt(1.0-ep)
    mat[3,0] = ep*(et3*np.exp(2.0j*the)*(X**2-0.5)-et4*np.exp(2.0j*vth)*(Y**2-0.5))
    mat[3,1] = np.sqrt(ep)*(np.sqrt(et3)*X*np.exp(1.0j*the)+np.sqrt(et4)*Y*np.exp(1.0j*vth))
    mat[3,2] = np.sqrt(ep)*(np.sqrt(et3)*X*np.exp(1.0j*the)-np.sqrt(et4)*Y*np.exp(1.0j*vth))
    mat[3,3] = 1.0
    return mat

@nb.njit(inline='always')
def KM_XY10(r3,r4,dt,gam,the,vth,et3,et4):
    X = r3*np.sqrt(dt/2)
    ep = dt*gam
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    mat[1,0] = np.sqrt(ep*(1.0-ep)*(1.0-et3)/2.0)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*(1.0-et3)/2.0)
    mat[3,0] = ep*np.exp(1.0j*the)*X*np.sqrt(2.0*et3*(1.0-et3))
    mat[3,1] = np.sqrt(ep*(1.0-et3)/2.0)
    mat[3,2] = np.sqrt(ep*(1.0-et3)/2.0)
    return mat

@nb.njit(inline='always')
def KM_XY01(r3,r4,dt,gam,the,vth,et3,et4):
    Y = r4*np.sqrt(dt/2)
    ep = dt*gam
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    mat[1,0] = -np.sqrt(ep*(1.0-ep)*(1.0-et4)/2.0)
    mat[2,0] = np.sqrt(ep*(1.0-ep)*(1.0-et4)/2.0)
    mat[3,0] = -ep*np.exp(1.0j*vth)*Y*np.sqrt(2.0*et4*(1.0-et4))
    mat[3,1] = np.sqrt(ep*(1.0-et4)/2.0)
    mat[3,2] = -np.sqrt(ep*(1.0-et4)/2.0)
    return mat

@nb.njit(inline='always')
def KM_XY20(r3,r4,dt,gam,the,vth,et3,et4):
    ep = dt*gam
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    mat[3,0] = ep*(1.0-et3)/np.sqrt(2.0)
    return mat

@nb.njit(inline='always')
def KM_XY02(r3,r4,dt,gam,the,vth,et3,et4):
    ep = dt*gam
    mat = np.zeros((4,4))+1.0j*np.zeros((4,4))
    mat[3,0] = -ep*(1.0-et4)/np.sqrt(2.0)
    return mat 

''' eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
Readout Sampling for Homodyne Trajectories '''

# corresponding Lindblad operators
@nb.njit(inline='always')
def LXY3(gam,the):
    sa = np.kron(sqt.sigm(),sqt.ID(2))
    sb = np.kron(sqt.ID(2),sqt.sigm())
    return np.sqrt(0.5*gam)*np.exp(1.0j*the)*(sa+sb)
@nb.njit(inline='always')
def LXY4(gam,vth):
    sa = np.kron(sqt.sigm(),sqt.ID(2))
    sb = np.kron(sqt.ID(2),sqt.sigm())
    return np.sqrt(0.5*gam)*np.exp(1.0j*vth)*(sa-sb)

# signals in each port
@nb.njit(inline='always')
def sig3(gam,the,et3):
    L3 = LXY3(gam,the)
    return np.sqrt(et3)*(L3 + DAG(L3))
@nb.njit(inline='always')
def sig4(gam,vth,et4):
    L4 = LXY4(gam,vth)
    return np.sqrt(et4)*(L4 + DAG(L4))

# draw the readouts
def HOM_eta_RO(rho,dt,gam,the,vth,et3,et4):
    r3 = gauss(loc = np.real(np.trace(rho @ sig3(gam,the,et3))),scale = np.sqrt(1.0/dt))
    r4 = gauss(loc = np.real(np.trace(rho @ sig4(gam,vth,et4))),scale = np.sqrt(1.0/dt))
    return r3, r4

''' fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
Homodyne Trajectory Integrators '''

# take a step, updating the density matrix, for a diffusive homodyne trajectory
def HOM_eta_step(rho,dt,gam,the,vth,et3,et4,rro=False):
    r3, r4 = HOM_eta_RO(rho,dt,gam,the,vth,et3,et4) # get a readout
    # update the state accordingly
    M0 = KM_XY00(r3,r4,dt,gam,the,vth,et3,et4)
    M1 = KM_XY01(r3,r4,dt,gam,the,vth,et3,et4)
    M2 = KM_XY10(r3,r4,dt,gam,the,vth,et3,et4)
    M3 = KM_XY02(r3,r4,dt,gam,the,vth,et3,et4)
    M4 = KM_XY20(r3,r4,dt,gam,the,vth,et3,et4)
    numer = M0 @ rho @ DAG(M0) + M1 @ rho @ DAG(M1) + M2 @ rho @ DAG(M2) + M3 @ rho @ DAG(M3) + M4 @ rho @ DAG(M4)
    denom = np.trace(numer) # normalization of new rho
    if rro == True:
        return numer/denom, np.asarray([r3,r4]) # return the updated density matrix and the click record
    else:
        return numer/denom # return the updated density matrix only
    
# run a single diffusive trajectory (homodyne at both outputs after BS)
def HOM_eta_traj(rh0,dt,dtsf,gam,T,the,vth,et3,et4,rro = False): # rh0 is initial density matrix.
    dtp = dt*dtsf # dtsf is factor of saved time points. Used to reduce memory use for large ensembles
    Ntp = np.rint(T/dtp).astype(np.int64)+1 # number of timesteps saved
    Nt = np.rint(T/dt).astype(np.int64)+1 # number of timesteps run
    ctp = np.linspace(0,T,Ntp) # initialize time array
    c_rho = np.zeros((Ntp,4,4)) + 1.0j*np.zeros((Ntp,4,4)) # initialize array for the state coordinates
    c_rho[0,:,:] = rh0
    if rro == True: # only use this for debugging
        ct = np.linspace(0,T,Nt) # initialize time array
        cr = np.zeros((Nt-1,2)) # in order to save the measurement record
        for k in range(1,Nt):
            rh0, cr[k-1,:] = HOM_eta_step(rh0,dt,gam,the,vth,et3,et4,rro = True)
            if k%dtsf == 0: # save a point
                c_rho[np.rint(k/dtsf).astype(np.int64),:,:] = rh0
        return ctp, c_rho, ct, cr # return the results
    else:
        for k in range(1,Nt):
            rh0 = HOM_eta_step(rh0,dt,gam,the,vth,et3,et4)
            if k%dtsf == 0: # save a point
                c_rho[np.rint(k/dtsf).astype(np.int64),:,:] = rh0
        return ctp, c_rho # return the results

''' 222222222222222222222222222222222222222222222222222222222222222222222222222
Section 2: Simulation Wrappers and Two-Qubit Plotters

    Both jump trajectory and diffuisve trajectory implementations are included.
    The diffusive implementation can be viewed as a supplement to the methods 
    in the auxiliary sqt file. While the sqt file is easier to generalize, the 
    below better matches the physical reasoning and presentation found in 
    AOP, 13 (3), 517-583 (2021).

222222222222222222222222222222222222222222222222222222222222222222222222222 '''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Jump Trajectory Ensemble Simulator '''

# rh0 ~ initial density matrix (complex 4x4 hermitian array with trace = 1)
# dt ~ timestep (real float)
# gam ~ flourescence rate (assumed same for both qubits; real float)
# T ~ total simulation runtime (real float)
# N_SQT ~ number of trajectories to run for t in [0,T] in increments dt (int >= 1)
# img_name ~ used to name plots created as part of this simulation (string)
# et3, et4 ~ efficiencies of the photodetection measurements (float in [0.,1.], with default 1)
# concplot ~ Boolean specifying whether to plot the concurrence of the simulated trajectories
# avgconc ~ Boolean specifying whether the average concurrence should be computed and plotted (irrelevant if concplot == False)
# qpanels ~ Boolean specifying a density matrix coordinate parameterization is to be plotted
# cap_override ~ Boolean specifying whether more than 12 trajectories should be plotted in most panels of qpanels (irrelevant if qpanels = False)
def SIM_jump_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,et3 = 1.0,et4 = 1.0,concplot = True,avgconc = True,qpanels = False,cap_override = False,rro = False):
    dtp = dt*dtsf
    Ntp = np.rint(T/dtp).astype(np.int64)+1 # number of timesteps saved
    c_rho = np.zeros((N_SQT,Ntp,4,4)) + 1.0j*np.zeros((N_SQT,Ntp,4,4)) # initialize array for the state coordinates
    if rro == True:
        Nt = np.rint(T/dt).astype(np.int64)+1 # number of timesteps run
        cr = np.zeros((N_SQT,Nt-1,5),dtype = np.int8) # in order to save the measurement record; not used in this instance
    for n in range(0,N_SQT): # run an N_SQT-sized ensemble of trajectories
        if rro == False:
            ctp, c_rho[n,:,:,:] = JUMP_eta_traj(rh0,dt,dtsf,gam,T,et3,et4) #, ct, cr[n,:,:] = JUMP_eta_traj(rh0,dt,dtsf,gam,T,et3,et4,rro=True)
        else:
            ctp, c_rho[n,:,:,:], ct, cr[n,:,:] = JUMP_eta_traj(rh0,dt,dtsf,gam,T,et3,et4,rro=True)
    if concplot == True: # are we plotting traces of the concurrence along trajectories?
        concur_plot(ctp, c_rho,avgconc,img_name)
    if qpanels == True:
        qpanels_plot(ctp, c_rho, img_name, cap_override)
    if rro == True:
        return ctp, c_rho, ct, cr # trajectories are passed back in case of further plotting, analysis, debugging, etc.
    else:
        return ctp, c_rho

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Diffusive Trajectory Ensemble Simulation '''

# rh0 ~ initial density matrix (complex 4x4 hermitian array with trace = 1)
# dt ~ timestep (real float)
# gam ~ flourescence rate (assumed same for both qubits; real float)
# T ~ total simulation runtime (real float)
# N_SQT ~ number of trajectories to run for t in [0,T] in increments dt (int >= 1)
# img_name ~ used to name plots created as part of this simulation (string)
# et3, et4 ~ efficiencies of the photodetection measurements (float in [0.,1.], with default 1)
# concplot ~ Boolean specifying whether to plot the concurrence of the simulated trajectories
# avgconc ~ Boolean specifying whether the average concurrence should be computed and plotted (irrelevant if concplot == False)
# qpanels ~ Boolean specifying a density matrix coordinate parameterization is to be plotted
# cap_override ~ Boolean specifying whether more than 12 trajectories should be plotted in most panels of qpanels (irrelevant if qpanels = False)
# method ~ Specify whether to use the Kraus Operators or Rouchon + Ralph Integrator for the simulation
def SIM_hom_ensemble(rh0,dt,dtsf,gam,T,N_SQT,img_name,the=0.0,vth=np.pi/2,et3 = 1.0,et4 = 1.0,concplot = True,avgconc = True,qpanels = False,cap_override = False,rro = False,method = 'Kraus'):
    dtp = dt*dtsf
    Ntp = np.rint(T/dtp).astype(np.int64)+1 # number of timesteps saved
    c_rho = np.zeros((N_SQT,Ntp,4,4)) + 1.0j*np.zeros((N_SQT,Ntp,4,4)) # initialize array for the state coordinates
    OM = np.zeros((4,4)) + 1.0j*np.zeros((4,4))
    L = np.asarray([LXY3(gam,the),LXY4(gam,vth)])
    eta = np.asarray([et3,et4])
    if rro == True:
        Nt = np.rint(T/dt).astype(np.int64)+1 # number of timesteps run
        cr = np.zeros((N_SQT,Nt-1,2),dtype = np.float16) # in order to save the measurement record; not used in this instance
    for n in range(0,N_SQT): # run an N_SQT-sized ensemble of trajectories
        if rro == False:
            if method == 'Kraus':
                ctp, c_rho[n,:,:,:] = HOM_eta_traj(rh0,dt,dtsf,gam,T,the,vth,et3,et4) 
            elif method == 'Rouchon':
                ctp, c_rho[n,:,:,:] = sqt.LowMem_Trajectory_Rouchon(rh0,OM,L,eta,dt,dtsf,T)
        else:
            if method == 'Kraus':
                ctp, c_rho[n,:,:,:], ct, cr[n,:,:] = HOM_eta_traj(rh0,dt,dtsf,gam,T,the,vth,et3,et4,rro = True)
            elif method == 'Rouchon':
                ctp, c_rho[n,:,:,:], ct, cr[n,:,:] = sqt.LowMem_Trajectory_Rouchon(rh0,OM,L,eta,dt,dtsf,T,rro = True)
    if concplot == True: # are we plotting traces of the concurrence along trajectories?
        concur_plot(ctp, c_rho,avgconc,img_name+'_'+method)
    if qpanels == True:
        qpanels_plot(ctp, c_rho, img_name+'_'+method, cap_override)
    if rro == True:
        return ctp, c_rho, ct, cr # trajectories are passed back in case of further plotting, analysis, debugging, etc.
    else:
        return ctp, c_rho

''' ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
Concurrence Plotter 

    See Wooters Phys. Rev. Lett. 80, 2245 (1998) for the underlying definition
    of Concurrence (a two-qubit entanglement monotone).

'''

# concurrence between two qubits
@nb.njit(inline='always')
def concurrence(rho):
    sy2 = np.kron(sqt.sigY(),sqt.sigY())
    egv = np.linalg.eigvals(rho @ sy2 @ np.conjugate(rho) @ sy2).real
    for l in range(0,4):
        if -1.0e-3 < egv[l] < 0.0: # clean up numerical error causing issues in sqrt
            egv[l] = 0
    egv = np.sort(np.sqrt(egv))
    C = egv[3]-egv[2]-egv[1]-egv[0]
    if C < 0.0:
        C = 0.0 
    return C

def concur_plot(ct,c_rho_ens,avgconc,img_name):
    N_SQT = c_rho_ens.shape[0]
    Nt = ct.size
    c_conc = np.zeros((N_SQT,Nt))
    for n in range(0,N_SQT):
        for k in range(0,Nt):
            c_conc[n,k] = concurrence(c_rho_ens[n,k,:,:])
    fig0 = pl.figure(figsize=(5.0,3.5))
    ax0 = fig0.add_subplot(111)
    ax0.grid('on')
    ax0.set_xlim([ct[0],ct[-1]]); ax0.set_xlabel(r'$t$')
    ax0.set_ylim([-0.02,1.02]); ax0.set_ylabel(r'$\mathcal{C}$')
    ccycle_12 = cycler('color',['xkcd:purple','xkcd:lilac','xkcd:bright blue','xkcd:teal','xkcd:aquamarine','xkcd:chartreuse','xkcd:goldenrod','xkcd:pale orange','xkcd:coral','xkcd:deep pink','xkcd:wine','xkcd:charcoal'])
    ax0.set_prop_cycle(ccycle_12)
    if avgconc == True: # plot the average concurrence, and that of up to a dozen individual trajectories
        c_avg = np.sum(c_conc,axis=0)/N_SQT
        ax0.plot(ct,c_avg,'k')
        if N_SQT > 12:
            SQT_plotmax = 12 
        else:
            SQT_plotmax = N_SQT
        for n in range(0,SQT_plotmax):
            ax0.plot(ct,c_conc[n,:],alpha = 0.5)
    else: # just plot the concurrence of all individual trajectories
        for n in range(0,N_SQT): # note: for large N_SQT this will get cluttered, and a density plot is strongly recommended instead
            ax0.plot(ct,c_conc[n,:],alpha = 0.5)
    fig0.tight_layout()
    fig0.savefig('2QF_concplot_'+img_name+'.png',dpi = 300)
    ax0.cla(); fig0.clf()

''' ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
Density Matrix Visualizers '''

def qpanels_plot(ct, c_rho_ens, img_name, cap_override):
    N_SQT = c_rho_ens.shape[0]
    Nt = ct.size
    cq = sqt.qvec_NNQ(c_rho_ens,(N_SQT,Nt),2) # coordinate decomposition of trajectories
    figq, qaxs = pl.subplots(4,4,figsize=(15,10))
    ccycle_12 = cycler('color',['xkcd:purple','xkcd:lilac','xkcd:bright blue','xkcd:teal','xkcd:aquamarine','xkcd:chartreuse','xkcd:goldenrod','xkcd:pale orange','xkcd:coral','xkcd:deep pink','xkcd:wine','xkcd:charcoal'])
    qaxs[0,0].set_prop_cycle(ccycle_12); qaxs[1,0].set_prop_cycle(ccycle_12); qaxs[2,0].set_prop_cycle(ccycle_12); qaxs[3,0].set_prop_cycle(ccycle_12)
    qaxs[0,1].set_prop_cycle(ccycle_12); qaxs[1,1].set_prop_cycle(ccycle_12); qaxs[2,1].set_prop_cycle(ccycle_12); qaxs[3,1].set_prop_cycle(ccycle_12)
    qaxs[0,2].set_prop_cycle(ccycle_12); qaxs[1,2].set_prop_cycle(ccycle_12); qaxs[2,2].set_prop_cycle(ccycle_12); qaxs[3,2].set_prop_cycle(ccycle_12)
    qaxs[0,3].set_prop_cycle(ccycle_12); qaxs[1,3].set_prop_cycle(ccycle_12); qaxs[2,3].set_prop_cycle(ccycle_12); qaxs[3,3].set_prop_cycle(ccycle_12)
    qaxs[0,0].set_ylabel(r'$\langle Populations \rangle$'); qaxs[0,0].set_xlabel(r'$t$')
    qaxs[0,1].set_ylabel(r'$q_{IX}$'); qaxs[0,1].set_xlabel(r'$t$')
    qaxs[0,2].set_ylabel(r'$q_{IY}$'); qaxs[0,2].set_xlabel(r'$t$')
    qaxs[0,3].set_ylabel(r'$q_{IZ}$'); qaxs[0,3].set_xlabel(r'$t$')
    qaxs[1,0].set_ylabel(r'$q_{XI}$'); qaxs[1,0].set_xlabel(r'$t$')
    qaxs[1,1].set_ylabel(r'$q_{XX}$'); qaxs[1,1].set_xlabel(r'$t$')
    qaxs[1,2].set_ylabel(r'$q_{XY}$'); qaxs[1,2].set_xlabel(r'$t$')
    qaxs[1,3].set_ylabel(r'$q_{XZ}$'); qaxs[1,3].set_xlabel(r'$t$')
    qaxs[2,0].set_ylabel(r'$q_{YI}$'); qaxs[2,0].set_xlabel(r'$t$')
    qaxs[2,1].set_ylabel(r'$q_{YX}$'); qaxs[2,1].set_xlabel(r'$t$')
    qaxs[2,2].set_ylabel(r'$q_{YY}$'); qaxs[2,2].set_xlabel(r'$t$')
    qaxs[2,3].set_ylabel(r'$q_{YZ}$'); qaxs[2,3].set_xlabel(r'$t$')
    qaxs[3,0].set_ylabel(r'$q_{ZI}$'); qaxs[3,0].set_xlabel(r'$t$')
    qaxs[3,1].set_ylabel(r'$q_{ZX}$'); qaxs[3,1].set_xlabel(r'$t$')
    qaxs[3,2].set_ylabel(r'$q_{ZY}$'); qaxs[3,2].set_xlabel(r'$t$')
    qaxs[3,3].set_ylabel(r'$q_{ZZ}$'); qaxs[3,3].set_xlabel(r'$t$')
    if cap_override == False:
        max_sqt = 12 
    else:
        max_sqt = N_SQT
    crho_avg = np.sum(c_rho_ens,axis = 0)/N_SQT
    qaxs[0,0].plot(ct,np.real(crho_avg[:,0,0]),'k',label='ee')
    qaxs[0,0].plot(ct,np.real(crho_avg[:,1,1]),'b',label='eg')
    qaxs[0,0].plot(ct,np.real(crho_avg[:,2,2]),'m',label='ge')
    qaxs[0,0].plot(ct,np.real(crho_avg[:,3,3]),'r',label='gg')
    qaxs[0,0].legend(loc="right")
    for n in range(0,max_sqt):
        qaxs[0,1].plot(ct,cq[n,:,0],alpha = 0.5)
        qaxs[0,2].plot(ct,cq[n,:,1],alpha = 0.5)
        qaxs[0,3].plot(ct,cq[n,:,2],alpha = 0.5)
        qaxs[1,0].plot(ct,cq[n,:,3],alpha = 0.5)
        qaxs[1,1].plot(ct,cq[n,:,4],alpha = 0.5)
        qaxs[1,2].plot(ct,cq[n,:,5],alpha = 0.5)
        qaxs[1,3].plot(ct,cq[n,:,6],alpha = 0.5)
        qaxs[2,0].plot(ct,cq[n,:,7],alpha = 0.5)
        qaxs[2,1].plot(ct,cq[n,:,8],alpha = 0.5)
        qaxs[2,2].plot(ct,cq[n,:,9],alpha = 0.5)
        qaxs[2,3].plot(ct,cq[n,:,10],alpha = 0.5)
        qaxs[3,0].plot(ct,cq[n,:,11],alpha = 0.5)
        qaxs[3,1].plot(ct,cq[n,:,12],alpha = 0.5)
        qaxs[3,2].plot(ct,cq[n,:,13],alpha = 0.5)
        qaxs[3,3].plot(ct,cq[n,:,14],alpha = 0.5)
    figq.tight_layout()
    figq.savefig('2QF_qplot_'+img_name+'.png',dpi = 300)
    figq.clf()
