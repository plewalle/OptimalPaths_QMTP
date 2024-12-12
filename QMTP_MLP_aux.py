'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2014-2021)
KB Whaley Group at UC Berkeley (2021-current)

August & September 2022, with revision November & December 2024

Python 3 code update: Multipath Manifolds for Jordan and Siddiqi textbook on Quantum Measurement.
See Phys. Rev. A 96, 053807 (2017) for the original.

Python 3 code update: Stroboscopic phase portraits for the kicked XZ-measurement rotor.
See Phys. Rev. A 98, 012141 (2018) for the original.

This file contains the main computational methods, but does not execute them. 
It is paired with a script which loads and runs these methods for specific values.
'''
# code tested in python 3.9.20; package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
from scipy.integrate import solve_ivp #                            scipy 1.10.0
from matplotlib import pyplot as pl #                          matplotlib 3.9.2
import matplotlib.cm as cm # color map in plots
import numba as nb #                                               numba 0.60.0

'''
CONTENTS:

1. ODE Integrator Methods ................................................  045
    a. RK4: Integration Debugging / Batch Divergence Detection ...........  049
    b. RK4 x DOP 853: High-Order Fine Tuning .............................  126

2. Equations of Motion ...................................................  226
    a. Driven Decaying Qubit .............................................  230
    b. Kicked XZ Measurement .............................................  286

3. 2D/4D Manifold Integrator and Plotter .................................  383

4. 1D/2D Stroboscopic Phase Portrait Plotter .............................  458 
    a. Integration and Lyapunov Exponents ................................  462
    b. Plotting ..........................................................  598
    
'''

'''
1111111111111111111111111111111111111111111111111111111111111111111111111111111
Batch ODE Integration (Update / Upgrade for Python 3)
1111111111111111111111111111111111111111111111111111111111111111111111111111111
''' 

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Fourth-Order Runge-Kutta (fixed timestep)

    Primarily used for idenifying initial conditions leading to divergences, 
    and getting preliminary error estimates with a fixed, possibly crude, 
    timestep. This initial sorting allows us to avoid errors later, and pass
    promising but non-convergent solutions to a higher-order adaptive method
    later for more precise re-integration. We here use a fixed-timestep method
    because this allows us to batch integrate many initial conditions in 
    parallel. 

'''

# t0 is initial time
# ic is array of initial conditions at t0
# dt is the timestep
# eqmo specifies the equations of motion
# eqargs is a list of parameters used in eqmo
def RK4_STEP_fixed(t0,ic,dt,eqmo,eqargs):
    k1 = dt*eqmo(t0,ic,eqargs) 
    k2 = dt*eqmo(t0+dt/2.0,ic+k1/2.0,eqargs)
    k3 = dt*eqmo(t0+dt/2.0,ic+k2/2.0,eqargs)
    k4 = dt*eqmo(t0+dt,ic+k3,eqargs)
    fc = ic + (k1+k4)/6.0 + (k2+k3)/3.0
    return fc

# single path, not a batch integrator yet! 
def RK4_Fixed(ic,eqmo,eqargs,Ti,Tf,dt):
    dim = ic.size # dimension of system of eqs
    # number of timesteps to reach final time
    Ntime = np.rint((Tf-Ti)/dt).astype(np.int64)+1 
    ct = np.linspace(Ti,Tf,Ntime) # time array
    cx = np.zeros((dim,Ntime)) # will hold the solutions
    cx[:,0] = ic
    for i in range(1,Ntime):
        cx[:,i] = RK4_STEP_fixed(ct[i-1],cx[:,i-1],dt,eqmo,eqargs)
    return ct, cx

''' Now the batch integrator for fixed time-step. Low order but reliable.
Use for finding and flagging divergences in an ensemble. Error estimates are 
made by doing integration twice at different timesteps. '''
# Choose timestep such than an even number of them fits in the time interval.
# fet = final error tolerance (absolute and relative) 
def RK4_BatchFixedDiv(ic,eqmo,eqargs,Ti,Tf,dt):
    Num = ic[0,:].size # number of initial conditions
    dim = ic[:,0].size # dimension of ODE system 
    ErrMaxTot = np.zeros(Num)
    PassMask = [True]*Num
    # number of timesteps including initial and final
    Ntime = np.rint((Tf-Ti)/dt).astype(np.int64)+1
    ct = np.linspace(Ti,Tf,Ntime) # time array
    cx = np.zeros((dim,Ntime,Num)) # will hold the solutions
    cx[:,0,:] = ic
    Ntime2 = np.rint((Tf-Ti)/(2*dt)).astype(np.int64)+1
    ct2 = np.linspace(Ti,Tf,Ntime2)
    cx2 = np.zeros((dim,Ntime2,Num))
    cx2[:,0,:] = ic
    finerr = np.zeros((dim,Num))
    # integrate for keeps
    for i in range(1,Ntime):
        cx[:,i,:] = RK4_STEP_fixed(ct[i-1],cx[:,i-1,:],dt,eqmo,eqargs)
    # integrate for error check
    for i in range(1,Ntime2):
        cx2[:,i,:] = RK4_STEP_fixed(ct2[i-1],cx2[:,i-1,:],2*dt,eqmo,eqargs)
    # get the estimate on the final error
    finerr = np.fabs(cx[:,-1,:]-cx2[:,-1,:])
    for n in range(0,Num):
        if all(np.isfinite(finerr[:,n]))==True:
            ErrMaxTot[n] = np.sum(finerr[:,n])
            if any(np.fabs(cx[:,-1,n]) > np.fabs(ic[:,n])*1.0e10)==True and any(np.fabs(cx[:,-1,n]) > 1.0e10)==True:
                PassMask[n] = False
        else:
            PassMask[n] = False
            ErrMaxTot[n] = np.inf
    return ct, cx, ErrMaxTot, PassMask

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Composite Integrator: Join RK4 above with a higher order method --
Eighth-Order Runge-Kutta (Dormand-Prince 853 algorithm, adaptive timestep)

    This is a higher order integrator that we use to obtain precise solutions
    for specific intial conditions which the simple analysis suggest should
    lead to convergent solutions, but which did not meet error tolerances.
    Use Press et. al sec. 17.2.4 and http://www.nr.com/webnotes?20 as a guide
    to the inner workings of the algorithm. Python documentation can be found
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

'''

def Batch_Integrate(ic,eqmo,eqargs_array,Ti,Tf,aberr_step,relerr_step,timescale,dtplot,fullsol = True):
    # dim = ic[:,0].size # dimension of dynamical system
    Num = ic[0,:].size # number of initial conditions
    # do a first fixed-timestep integration
    ct, cx, ErrMaxTot, DivPassMask = RK4_BatchFixedDiv(ic,eqmo,eqargs_array,Ti,Tf,dtplot)
    Ntime = ct.size 
    # cap the total acceptable error: Decide which paths to pass to higher-order method
    AbErrTol = aberr_step*(Ntime/(Tf-Ti))
    ReErrTol = 0.0001*np.fabs(cx[0,-1,:])
    # which trajectories pass or fail? take stock, and do further analysis where needed
    if fullsol == True:
        cx_save = cx
    else:
        cx_save = cx[:,[0,-1],:]
    newdiverge = 0
    savecount = 0
    for n in range(0,Num):
        if DivPassMask[n] == True and ErrMaxTot[n] < AbErrTol+ReErrTol[n]: 
            savecount += 1 # no divergence, and agreeable error characteristics
        if DivPassMask[n] == True and ErrMaxTot[n] > AbErrTol+ReErrTol[n]: # no divergence problem, but need further attention
            if fullsol == True:
                sol_dop853_1 = solve_ivp(eqmo,[Ti,Tf],ic[:,n],t_eval = ct,args = [eqargs_array],method = 'DOP853',dense_output = True,atol = aberr_step,rtol = relerr_step)
                if sol_dop853_1.success == True: # did the integrator work?
                    cx_save[:,:,n] = sol_dop853_1.y # yes! 
                    savecount += 1
                else: 
                    newdiverge += 1; DivPassMask[n] = False # no!
            else:
                Ntimescale = np.rint((Tf-Ti)/timescale).astype(np.int64) + 2
                sol_dop853_1 = solve_ivp(eqmo,[Ti,Tf],ic[:,n],t_eval = np.linspace(Ti,Tf,Ntimescale),args = [eqargs_array],method = 'DOP853',dense_output = True,atol = aberr_step,rtol = relerr_step)
                if sol_dop853_1.success == True: # did the integrator work?
                    cx_save[:,0,n] = ic[:,n]    
                    cx_save[:,1,n] = sol_dop853_1.y[:,-1] # yes!
                    savecount += 1
                else: 
                    newdiverge += 1; DivPassMask[n] = False # no!
    print("Succesfully Integrated "+str(savecount)+"/"+str(Num)+" trajectories")
    if fullsol == True:
        return ct, cx_save, DivPassMask
    else: 
        return  cx_save, DivPassMask
    
# works like the above, just for a single initial condition
def Single_Integrate(ic,eqmo,eqargs_array,Ti,Tf,aberr_step,relerr_step,timescale,dtplot,fullsol = True):
    ''' Integrate RK4 at dt and 2*dt. Compare '''
    dim = ic.size # dimension of system of eqs
    # number of timesteps to reach final time
    Ntime = np.rint((Tf-Ti)/dtplot).astype(np.int64)+1 
    HalfTime = np.rint((Tf-Ti)/(2*dtplot)).astype(np.int64)+1
    ct = np.linspace(Ti,Tf,Ntime) # time array
    cx = np.zeros((dim,Ntime)) # will hold the solutions
    cx[:,0] = ic
    for i in range(1,Ntime):
        cx[:,i] = RK4_STEP_fixed(ct[i-1],cx[:,i-1],dtplot,eqmo,eqargs_array)
    cxref = np.zeros((dim,HalfTime)) # will hold the solutions
    cxref[:,0] = ic
    for i in range(1,HalfTime):
        cxref[:,i] = RK4_STEP_fixed(ct[i-1],cx[:,i-1],2*dtplot,eqmo,eqargs_array)
    finerr = np.abs(cx[:,-1]-cxref[:,-1])
    if any(np.isfinite(finerr)) == False: # throw a warning and terminate
        print("Integration Failure: Diverging RK4 Test Integration")
        if fullsol == True:
            return ct, cx
        else:
            return cx[:,[0,-1]]
    else: # continue with higher-order method
         if fullsol == True:
             sol_dop853_1 = solve_ivp(eqmo,[Ti,Tf],ic,method = 'DOP853',t_eval = ct, args = [eqargs_array], dense_output = True,atol = aberr_step,rtol = relerr_step)
             if sol_dop853_1.success == True: # did the integrator work?
                 cx = sol_dop853_1.y # yes! 
                 return ct, cx
             else: # no!
                 print("Integration Failure: DOP853 Warning, with RK4 Solution Returned")
                 return ct, cx
         else: 
             Ntimescale = np.rint((Tf-Ti)/timescale).astype(np.int64) + 2
             sol_dop853_1 = solve_ivp(eqmo,[Ti,Tf],ic,t_eval = np.linspace(Ti,Tf,Ntimescale),args = [eqargs_array],method = 'DOP853',dense_output = True,atol = aberr_step,rtol = relerr_step)
             if sol_dop853_1.success == True: # did the integrator work?
                 cx = sol_dop853_1.y[:,[0,-1]] # yes! 
                 return cx
             else: 
                 print("Integration Failure: DOP853 Warning, with RK4 Solution Returned")
                 return cx[:,[0,-1]]
    # note that we only return the RK4 result in case of error here
    # for a single trajectory, the time cost involved with re-computing via higher order method is just not that big

'''
2222222222222222222222222222222222222222222222222222222222222222222222222222222
Optimal Path Equations of Motion for Problems of Interest
2222222222222222222222222222222222222222222222222222222222222222222222222222222
'''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Driven and Decaying Qubit: XZ Bloch Plane '''

# inefficient homodyne fluorescence with Rabi Drive causing rotations
# everything is in xz plane of Bloch sphere 
def HomF_OP_EQMO(t,Q,eqargs):
    gam = eqargs[0]
    om = eqargs[1]
    eta = eqargs[2]
    if Q.size == 4:
        x = Q[0]
        z = Q[1]
        px = Q[2]
        pz = Q[3]
        r = (px*(1.0 - x**2 + z) + x*(1.0 - pz - pz*z))*np.sqrt(gam*eta)
        xdot = om*z - r*x**2*np.sqrt(gam*eta) + r*(z + 1.0)*np.sqrt(gam*eta) + 0.5*gam*x*(eta + eta*z-1.0)
        zdot = -om*x + (z + 1.0)*(0.5*gam*(eta + eta*z - 2.0) - r*x*np.sqrt(gam*eta))
        pxdt = 2*px*r*x*np.sqrt(gam*eta) +r*(-1.0 + pz + pz*z)*np.sqrt(eta*gam) - 0.5*px*gam*(-1.0 + eta + z*eta) + pz*om
        pzdt = -0.5*(-1.0 + px*x)*eta*gam + pz*r*x*np.sqrt(gam*eta) - pz*gam*(-1.0 + eta + z*eta) - px*(r*np.sqrt(eta*gam) + om)
        QDOT = np.asarray([xdot,zdot,pxdt,pzdt])
    else:
        x = Q[0,:]
        z = Q[1,:]
        px = Q[2,:]
        pz = Q[3,:]
        N = Q[0,:].size
        r = (px*(1.0*np.ones(N) - x**2 + z) + x*(1.0*np.ones(N) - pz - pz*z))*np.sqrt(gam*eta)
        QDOT = np.zeros((4,Q[0,:].size))
        QDOT[0,:] = om*z - r*x**2*np.sqrt(gam*eta) + r*(z + 1.0*np.ones(N))*np.sqrt(gam*eta) + 0.5*gam*x*(eta + eta*z-1.0*np.ones(N))
        QDOT[1,:] = -om*x + (z + 1.0*np.ones(N))*(0.5*gam*(eta + eta*z - 2.0*np.ones(N)) - r*x*np.sqrt(gam*eta))
        QDOT[2,:] = 2*px*r*x*np.sqrt(gam*eta) +r*(-1.0*np.ones(N) + pz + pz*z)*np.sqrt(eta*gam) - 0.5*px*gam*(-1.0*np.ones(N) + eta + z*eta) + pz*om
        QDOT[3,:] = -0.5*(-1.0*np.ones(N) + px*x)*eta*gam + pz*r*x*np.sqrt(gam*eta) - pz*gam*(-1.0*np.ones(N) + eta + z*eta) - px*(r*np.sqrt(eta*gam) + om*np.ones(N))
    return QDOT

# stochastic Hamiltonian / energy for the same
def HomF_OP_H(Q,eqargs):
    gam = eqargs[0]
    om = eqargs[1]
    eta = eqargs[2]
    if Q.size == 4:
        x = Q[0]
        z = Q[1]
        px = Q[2]
        pz = Q[3]
    else:
        x = Q[0,:]
        z = Q[1,:]
        px = Q[2,:]
        pz = Q[3,:]
    r = (px*(1.0 - x**2 + z) + x*(1.0 - pz - pz*z))*np.sqrt(gam*eta)
    xdot = om*z - r*x**2*np.sqrt(gam*eta) + r*(z + 1.0)*np.sqrt(gam*eta) + 0.5*gam*x*(eta + eta*z-1.0)
    zdot = -om*x + (z + 1.0)*(0.5*gam*(eta + eta*z - 2.0) - r*x*np.sqrt(gam*eta))
    G = -0.5*(r - x*np.sqrt(eta*gam))**2 + 0.5*eta*gam*(x**2 - z - 1.0)
    H = px*xdot + pz*zdot + G
    return H

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Pure State Joint XZ Measurement: XZ Bloch Great Circle '''

@nb.njit(inline='always')
def a_xz(th,tx,tz):
    return 0.5*(np.sin(th)**2/tz+np.cos(th)**2/tx)
@nb.njit(inline='always')
def ad_xz(th,tx,tz):
    return ((tx - tz)*np.cos(th)*np.sin(th))/(tx*tz)
@nb.njit(inline='always')
def b_xz(th,tx,tz):
    return (1.0/tx-1.0/tz)*np.sin(th)*np.cos(th)
@nb.njit(inline='always')
def bd_xz(th,tx,tz):
    return -(((tx - tz)*np.cos(2*th))/(tx*tz))

def tZ(t,off,amp,sig):
    N = np.ceil(t[-1]).astype(np.int64)
    g = np.zeros((t.size,N))
    for n in range(0,N):
        g[:,n] = amp*np.exp(-0.5*((t-(n+0.5)*np.ones(t.size)*off)/(sig))**2)
    return off*np.ones(t.size) - np.sum(g,axis=1) # amp must be between zero (weak) and one (projective)

def tauZ(t,off,amp,sig): # for a float t
    N = np.rint(np.ceil(t)).astype(np.int64)+3
    g = np.zeros(N)
    for n in range(0,N):
        g[n] = amp*np.exp(-0.5*((t-(n+0.5)*off)/(sig))**2)
    return off-np.sum(g) # amp must be between zero (weak) and one (projective)

# readout
@nb.njit(inline='always')
def roz(th,p):
    return np.cos(th)-p*np.sin(th)
@nb.njit(inline='always')
def rox(th,p):
    return np.sin(th)+p*np.cos(th)

# equations of motion
@nb.njit(inline='always')
def fxz1_arr(th,p,tx,tz): # theta dot
    a = a_xz(th,tx,tz)
    b = b_xz(th,tx,tz)
    return 2*a*p+b
@nb.njit(inline='always')
def fxz2_arr(th,p,tx,tz):
    ad = ad_xz(th,tx,tz)
    bd = bd_xz(th,tx,tz)
    return -ad*(p**2-1.0)-bd*p

# re-formatted equations of motion
def QDOT_XZ_EQMO(t,q,eqargs_xz):
    qdot = np.zeros(q.shape)
    # theta dot
    if q.size == 2:
        th = q[0]; p = q[1]
    else:
        th = q[0,:]; p = q[1,:]
    off = eqargs_xz[0]; amp = eqargs_xz[1]; sig = eqargs_xz[2]
    tx = off; tz = tauZ(t,off,amp,sig)
    a = a_xz(th,tx,tz)
    b = b_xz(th,tx,tz)
    if q.size == 2:
        qdot[0] = 2*a*p + b
    else:
        qdot[0,:] = 2*a*p+b
    # pdot
    ad = ad_xz(th,tx,tz)
    bd = bd_xz(th,tx,tz)
    if q.size == 2:
        qdot[1] = -ad*(p**2-1.0)-bd*p
    else:
        qdot[1,:] = -ad*(p**2-1.0)-bd*p
    return qdot

# stochastic energy
@nb.njit(inline='always')
def Earr(th,p,tx,tz):
    a = a_xz(th,tx,tz)
    b = b_xz(th,tx,tz)
    N = a.size  
    return a*(p**2-np.ones(N))+b*p
@nb.njit(inline='always')
def Emat(th,p,tx,tz):
    a = a_xz(th,tx,tz)
    b = b_xz(th,tx,tz)
    N = a[:,0].size
    M = a[0,:].size
    return a*(p**2-np.ones((N,M)))+b*p
    
# stochastic action integrand
@nb.njit(inline='always')
def Sd_arr(th,p,tx,tz):
    return Earr(th,p,tx,tz) - p*fxz1_arr(th,p,tx,tz)

'''
3333333333333333333333333333333333333333333333333333333333333333333333333333333
Manifold Integration and Plotting: XZ Bloch Plane with 4D OP Phase Space
3333333333333333333333333333333333333333333333333333333333333333333333333333333
'''

def LM_iPmesh(qi,px_arr,pz_arr):
    px_min = px_arr[0]; px_max = px_arr[1]; dpx = px_arr[2];
    pz_min = pz_arr[0]; pz_max = pz_arr[1]; dpz = pz_arr[2];
    xmesh = np.arange(px_min,px_max,dpx)
    zmesh = np.arange(pz_min,pz_max,dpz)
    Qi = np.zeros((4,xmesh.size*zmesh.size))
    for i in range(0,xmesh.size):
        for j in range(0,zmesh.size):
            Qi[:,i*j+j] = np.asarray(qi[0],qi[1],xmesh[i],zmesh[j])
    return Qi

# plot the sampled LM in the xz plane of the Bloch sphere.
# NO REFINEMENT is applied in this case. Initial Pmesh determines sampling.
def LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,img_name):
    # establish the grid of initial conditions in the p variables
    pxmin = Pmesh[0,0]
    pxmax = Pmesh[0,1]
    NX = np.rint(Pmesh[0,2]).astype(np.int64)
    pzmin = Pmesh[1,0]
    pzmax = Pmesh[1,1]
    NZ = np.rint(Pmesh[1,2]).astype(np.int64)
    N = np.rint(NX*NZ).astype(np.int64)
    ic = np.zeros((4,N))
    pxMESH = np.linspace(pxmin,pxmax,NX)
    pzMESH = np.linspace(pzmin,pzmax,NZ)
    for nx in range(0,NX):
        for nz in range(0,NZ):
            ic[:,nx + nz*NX] = np.asarray([Qi[0],Qi[1],pxMESH[nx],pzMESH[nz]])
    # do the integration
    aberr_step = 1.0e-12; relerr_step = 1.0e-12
    QF, fl = Batch_Integrate(ic,HomF_OP_EQMO,eqargs,0.0,Tf,aberr_step,relerr_step,1.0/eqargs[0],0.01/eqargs[0],fullsol = False)
    N = QF[0,0,:].size # reset size to only count successful integrations
    # get range of stochastic energies (sets bounds on colorbar)
    StochEnergyArr = HomF_OP_H(QF[:,0,:],eqargs)
    Emin = np.amin(StochEnergyArr)
    Emax = np.amax(StochEnergyArr)
    # color array
    col = np.zeros((N,4))
    for n in range(0,N):
        col[n,:] = cm.viridis((StochEnergyArr[n]-Emin)/(Emax-Emin))
    # now the plotting
    fig0 = pl.figure(figsize=(5.,5.))
    ax0 = fig0.add_subplot(111)
    ax0.set_xlim([-1.02,1.02])
    ax0.set_ylim([-1.02,1.02])
    for n in range(0,N):
        if QF[0,1,n]**2 + QF[1,1,n]**2 <= 1.0:
            ax0.scatter(QF[0,1,n],QF[1,1,n],color = col[n,:],s = 0.1,marker = 's', alpha = 0.5)
    ax0.plot(np.linspace(-1.0,1.0,1000),np.sqrt(1.0-np.linspace(-1.0,1.0,1000)**2),'r--')
    ax0.plot(np.linspace(-1.0,1.0,1000),-np.sqrt(1.0-np.linspace(-1.0,1.0,1000)**2),'r--')
    fig0.savefig('LM_4DPS_'+img_name+'.pdf')#,dpi = 300)
    fig0.savefig('LM_4DPS_'+img_name+'.png',dpi = 300)
    ax0.cla(); fig0.clf()
    # make a colorbar
    fig1 = pl.figure(figsize=(1.0,4))
    ax1 = fig1.add_subplot(111)
    cbar = np.zeros((2,100))
    cbar[0,:] = np.linspace(Emin,Emax,100)
    cbar[1,:] = np.linspace(Emin,Emax,100)
    ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax1.imshow(cbar.T,cmap='viridis',origin='lower',aspect=15.0/(Emax-Emin),extent=(0,1,Emin,Emax))
    fig1.savefig('Esto_CBAR_'+img_name+'.pdf')#,dpi = 300)
    ax1.cla(); fig1.clf()

'''
4444444444444444444444444444444444444444444444444444444444444444444444444444444
Stroboscopic Phase Portrait: Chaotic Kicked XZ Measurement
4444444444444444444444444444444444444444444444444444444444444444444444444444444
'''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Preparation: Lyapunov Exponents, etc. '''

def Batch_LYAP_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz):
    # initialize the ic mesh
    the_mesh = np.linspace(0,2*np.pi,Nth)
    pth_mesh = np.linspace(-prange,prange,Np)
    Qi = np.zeros((2,3*Np*Nth))
    for i in range(0,Nth):
        for j in range(0,Np):
            Qi[1,3*i*j:3*i*j+3] = np.ones(3)*pth_mesh[j]
            Qi[0,3*i*j] = the_mesh[i] - dth0
            Qi[0,3*i*j+1] = the_mesh[i] 
            Qi[0,3*i*j+2] = the_mesh[i] + dth0
    # integrate all the paths
    aberr_step = 1.0e-12; relerr_step = 1.0e-12
    ct, cQ, fl_dpm = Batch_Integrate(Qi,QDOT_XZ_EQMO,eqargs_xz,0.0,Tf,aberr_step,relerr_step,eqargs_xz[2],dt_plot/dt_comp_factor,fullsol = True)
    Nt_plot = np.rint((ct.size-1)/dt_comp_factor).astype(np.int64) + 1
    tl = np.linspace(dt_comp_factor,Tf,Nt_plot-1)
    # compute the Lyapunov exponents
    Qplot = []; DCO = []
    for i in range(0,Nth*Np):
        if all(fl_dpm[3*i:3*i+3]) == True:
            Qplot.append(cQ[:,::dt_comp_factor,3*i+1]) # save the sub-sampled trajectory
            dcp = np.absolute(cQ[0,::dt_comp_factor,3*i+1]-cQ[0,::dt_comp_factor,3*i+2])
            dcm = np.absolute(cQ[0,::dt_comp_factor,3*i+1]-cQ[0,::dt_comp_factor,3*i])
            DCO.append(0.5*dcp + 0.5*dcm) # average of distance to +/- offset trajectories
    cQ_plot = np.moveaxis(np.asarray(Qplot),0,-1); 
    del Qplot
    dco_plot = np.asarray(DCO) # np.moveaxis(np.asarray(DCO),0,-1); 
    del DCO
    LYAP = np.zeros((Nt_plot-1,cQ_plot.shape[2]))
    for k in range(0,Nt_plot-1):
        LYAP[k,:] = np.log(np.asarray(dco_plot[:,k+1])/dth0)/tl[k]
    return tl, cQ_plot, LYAP

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Plotting Stroboscopic Phase Portrait '''

def StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,imgname):
    tl, cQ_plot, LYAP = Batch_LYAP_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz)
    NT = LYAP.shape[0]; NL = LYAP.shape[1]
    c2 = np.zeros((NT,NL,4)) # holds the color values for the Lyapunov exponents.
    LydMax = 0.1
    LydMin = -0.1
    for m in range(0,NL):
        for n in range(0,NT): # assign the actual CMYK values to each point below
            if LYAP[n,m] <= 0.0: # converging trajectories
                if LYAP[n,m] >= LydMin:
                    c2[n,m,:] = cm.Blues_r(LYAP[n,m]/LydMin)
                else:
                    c2[n,m,:] = cm.Blues_r(1)
            else: # diverging trajectories
                if LYAP[n,m] <= LydMax:
                    c2[n,m,:] = cm.Reds_r(LYAP[n,m]/LydMax)
                else:
                    c2[n,m,:] = cm.Reds_r(1)
    # main plotting
    fig0 = pl.figure(figsize=(5.0,3.5))
    ax0 = fig0.add_subplot(111)
    ax0.set_xlim([0,np.pi])
    ax0.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax0.set_xticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'])
    ax0.set_yticks([-np.pi,-2*np.pi/3,-np.pi/2,0,np.pi/2,2*np.pi/3,np.pi])
    ax0.set_yticklabels([r'$-\pi$',r'$-2\pi/3$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$2\pi/3$',r'$\pi$'])
    ax0.set_ylim([-prange,prange])
    Navg = 0.5*(Np+Nth); msize = np.amax([0.1,5.0/Navg])
    for i in range(0,NL):
        for k in range(0,NT):
            ax0.scatter(np.remainder(cQ_plot[0,k,i],np.pi),cQ_plot[1,k,i],color = c2[k,i,:],s = msize,marker = 's',edgecolors = 'none')
            ax0.scatter(np.remainder(-cQ_plot[0,k,i],np.pi),-cQ_plot[1,k,i],color = c2[k,i,:],s = msize,marker = 's',edgecolors = 'none')
    fig0.savefig('XZK_StroboPS'+imgname+'.pdf')
    fig0.savefig('XZK_StroboPS'+imgname+'.png',dpi = 300)
    ax0.cla(); fig0.clf()
    # colorbar plotting
    fig1 = pl.figure(figsize=(1.3,4))
    ax1 = fig1.add_subplot(111)
    cbar = np.zeros((2,100))
    cbar[0,:] = np.linspace(LydMin,0.0,100)
    cbar[1,:] = np.linspace(LydMin,0.0,100)
    ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax1.imshow(cbar.T,cmap='Blues_r',origin='upper',aspect=7.5/(np.absolute(LydMin)),extent=(0,1,LydMin,0))
    fig1.tight_layout()
    fig1.savefig('LYAP-minus_CBAR_'+imgname+'.pdf')#,dpi = 300)
    ax1.cla(); fig1.clf()
    fig1 = pl.figure(figsize=(1.3,4))
    ax1 = fig1.add_subplot(111)
    cbar = np.zeros((2,100))
    cbar[0,:] = np.linspace(LydMax,0.0,100)
    cbar[1,:] = np.linspace(LydMax,0.0,100)
    ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax1.imshow(cbar.T,cmap='Reds',origin='lower',aspect=7.5/(np.absolute(LydMax)),extent=(0,1,0,LydMax))
    fig1.tight_layout()
    fig1.savefig('LYAP-plus_CBAR_'+imgname+'.pdf')#,dpi = 300)
    ax1.cla(); fig1.clf()