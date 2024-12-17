'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

code created December 2024, based on older codes (2015-2022)

Batch ODE Integration (Update / Upgrade for Python 3)
See Appendix D of http://hdl.handle.net/1802/36379 
(Philippe Lewalle's dissertation) for older versions (Py 2) and references. 
'''
# code tested in python 3.9.20 
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
from scipy.integrate import solve_ivp #                            scipy 1.10.0

''' CONTENTS

Sec. 1: Batch Integration via 4th order Runge Kutta ....................... 028
Sec. 2: Single and Batch Integration Submethods ........................... 105
    a. Combined RK4 and DOP853 Integration for Batched Initial Cond. ...... 117    
    b. Combined RK4 and DOP853 Integration for Single Initial Cond. ....... 162
    
'''

''' 111111111111111111111111111111111111111111111111111111111111111111111111111

Fourth-Order Runge-Kutta (fixed timestep)

    Primarily used for idenifying initial conditions leading to divergences, 
    and getting preliminary error estimates with a fixed, possibly crude, 
    timestep. This initial sorting allows us to avoid errors later, and pass
    promising but non-convergent solutions to a higher-order adaptive method
    later for more precise re-integration. We here use a fixed-timestep method
    because this allows us to batch integrate many initial conditions in 
    parallel. 

111111111111111111111111111111111111111111111111111111111111111111111111111 '''

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

''' 222222222222222222222222222222222222222222222222222222222222222222222222222

Composite Integrator: Join RK4 above with a higher order method --
Eighth-Order Runge-Kutta (Dormand-Prince 853 algorithm, adaptive timestep)

    This is a higher order integrator that we use to obtain precise solutions
    for specific intial conditions which the simple analysis suggest should
    lead to convergent solutions, but which did not meet error tolerances.
    Use Press et. al sec. 17.2.4 and http://www.nr.com/webnotes?20 as a guide
    to the inner workings of the algorithm. Python documentation can be found
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

222222222222222222222222222222222222222222222222222222222222222222222222222 '''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Integrate ODE system for batch of initial conditions '''

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
    
''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Apply the same methods to a single initial condition '''
    
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
