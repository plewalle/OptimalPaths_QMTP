'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

August & September 2022, with revision November & December 2024

Python 3 code update: Multipath Manifolds for Jordan and Siddiqi textbook on Quantum Measurement.
See Phys. Rev. A 96, 053807 (2017) for the original.

Python 3 code update: Stroboscopic phase portraits for the kicked XZ-measurement rotor.
See Phys. Rev. A 98, 012141 (2018) for the original.

This file contains the main computational methods, but does not execute them. 
It is paired with a script which loads and runs these methods for specific values.
'''
# code tested in python 3.12.3 
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
import numba as nb #                                               numba 0.60.0
from matplotlib import pyplot as pl #                          matplotlib 3.9.2
import matplotlib.cm as cm # color map in plots
from matplotlib.colors import LinearSegmentedColormap # assemble custom cmaps
import ODE_aux as ode
import os; cur_path = os.path.dirname(__file__)
'''
CONTENTS:

1. Equations of Motion ...................................................  043
    a. Driven Decaying Qubit .............................................  047
    b. Kicked XZ Measurement .............................................  103

2. 2D/4D Manifold Integrator and Plotter .................................  200

3. 1D/2D Stroboscopic Phase Portrait Plotter .............................  302 
    a. Integration and Lyapunov Exponents ................................  306
    b. Plotting ..........................................................  342
    
'''

'''
1111111111111111111111111111111111111111111111111111111111111111111111111111111
Optimal Path Equations of Motion for Problems of Interest
1111111111111111111111111111111111111111111111111111111111111111111111111111111
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
2222222222222222222222222222222222222222222222222222222222222222222222222222222
Manifold Integration and Plotting: XZ Bloch Plane with 4D OP Phase Space
2222222222222222222222222222222222222222222222222222222222222222222222222222222
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
def LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,img_name,animate = False,Nframe = None):
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
    aberr_step = 1.0e-4; relerr_step = 1.0e-4
    if animate == False:
        print('\nLagrangian Manifold, Final Snapshot Initialized')
        QF, fl = ode.Batch_Integrate(ic,HomF_OP_EQMO,eqargs,0.0,Tf,aberr_step,relerr_step,1.0/eqargs[0],0.002/eqargs[0],fullsol = False)
    else:
        print('\nLagrangian Manifold, Time Animation Initialized')
        ct, QF, fl = ode.Batch_Integrate(ic,HomF_OP_EQMO,eqargs,0.0,Tf,aberr_step,relerr_step,1.0/eqargs[0],Tf/Nframe,fullsol = True)
    N = QF[0,0,:].size # reset size to only count successful integrations
    # get range of stochastic energies (sets bounds on colorbar)
    StochEnergyArr = HomF_OP_H(QF[:,0,:],eqargs)
    Emin = np.amin(StochEnergyArr)
    Emax = np.amax(StochEnergyArr)
    # color array
    col = np.zeros((N,4))
    for n in range(0,N):
        col[n,:] = cm.viridis((StochEnergyArr[n]-Emin)/(Emax-Emin))
    # do another set of error filters
    plotlist = []
    for n in range(0,N): # did the trajectory stay in bloch sphere, and was the stochastic energy conserved?
        if np.all(QF[0,:,n]**2 + QF[1,:,n]**2 <= 1.0 + 1.0e-4) and np.isclose(StochEnergyArr[n], HomF_OP_H(QF[:,-1,n],eqargs), rtol=2.0e-03, atol=2.0e-03*(Emax-Emin)):
            plotlist.append(n)
    # now the plotting
    fig0 = pl.figure(figsize=(5.,5.))
    ax0 = fig0.add_subplot(111)
    ax0.axes.set_aspect('equal')
    ax0.set_xlim([-1.02,1.02])
    ax0.set_ylim([-1.02,1.02])
    if animate == False:
        ax0.set_xlabel(r'$x$'); ax0.set_ylabel(r'$z$')
        print('Plotting in progress ... ',end = '')
        ax0.scatter(QF[0,1,plotlist],QF[1,1,plotlist],color = col[plotlist,:],s = 0.1,marker = 's', alpha = 0.5)
        ax0.plot(np.cos(np.linspace(0.0,2.0*np.pi,1000)),np.sin(np.linspace(0.0,2.0*np.pi,1000)),'r--')
        fig0.tight_layout()
        fig0.savefig('LM_4DPS_'+img_name+'.pdf')#,dpi = 300)
        fig0.savefig('LM_4DPS_'+img_name+'.png',dpi = 300)
        ax0.cla(); fig0.clf()
    else:
        if Nframe == None: # enforce a default frame rate if none was provided
            Nframe = int(np.around(5*Tf/300))
        ax0.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
        if not os.path.exists(cur_path + r'\LM_4DPS_'+img_name+'_webani'): # plot the frames at both a low and high resolution
            os.makedirs(cur_path + r'\LM_4DPS_'+img_name+'_webani')
        if not os.path.exists(cur_path + r'\LM_4DPS_'+img_name+'_ani'):
            os.makedirs(cur_path + r'\LM_4DPS_'+img_name+'_ani')
        for k in range(0,ct.size):
            ax0.scatter(QF[0,k,plotlist],QF[1,k,plotlist],color = col[plotlist,:],s = 0.4,marker = 's', alpha = 0.5)
            ax0.plot(np.cos(np.linspace(0.0,2.0*np.pi,1000)),np.sin(np.linspace(0.0,2.0*np.pi,1000)),'r--')
            fig0.savefig(cur_path+r'\LM_4DPS_'+img_name+r'_webani\webani_'+str(k).zfill(3)+'.png',dpi = 150,bbox_inches='tight')
            fig0.savefig(cur_path+r'\LM_4DPS_'+img_name+r'_ani\ani_'+str(k).zfill(3)+'.png',dpi = 300,bbox_inches='tight')
            ax0.cla()
            printProgressBar(k, ct.size, suffix = 'Plotting Frames', length = 50)
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
    print('Complete!')

'''
3333333333333333333333333333333333333333333333333333333333333333333333333333333
Stroboscopic Phase Portrait: Chaotic Kicked XZ Measurement
3333333333333333333333333333333333333333333333333333333333333333333333333333333
'''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Preparation: Lyapunov Exponents, etc. '''

def Batch_LYAP_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,relerr_step = 1.0e-6,aberr_step = 1.0e-6):
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
    ct, cQ, fl_dpm = ode.Batch_Integrate(Qi,QDOT_XZ_EQMO,eqargs_xz,0.0,Tf,aberr_step,relerr_step,eqargs_xz[2],dt_plot/dt_comp_factor,fullsol = True)
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

def StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,imgname,webres = False,aniframe = None,relerr_step = 1.0e-6,aberr_step = 1.0e-6):
    if webres == False:
        print('\nStroboscopic Phase Portrait Initialized')
    else:
        print('\nStroboscopic Phase Portrait Animation, Frame '+str(aniframe).zfill(3)+', Initialized')
        if not os.path.exists(cur_path+r'\XZK_StroboPS_'+imgname+r'_webani'): # plot the frames at both a low and high resolution
            os.makedirs(cur_path+r'\XZK_StroboPS_'+imgname+r'_webani')
    tl, cQ_plot, LYAP = Batch_LYAP_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,relerr_step = relerr_step,aberr_step = aberr_step)
    NT = LYAP.shape[0]; NL = LYAP.shape[1]
    c2 = np.zeros((NT+1,NL,4)) # holds the color values for the Lyapunov exponents.
    LydMax = 0.25; LydMin = -LydMax
    colors = ['c','b','k','r']; #nodes = [0.0,0.5,1.0] 
    cmap_ly = LinearSegmentedColormap.from_list('cmap_ly',colors,gamma = 0.5) #list(zip(nodes, colors))
    for m in range(0,NL):
        c2[0,m,:] = cmap_ly(0.5)
        for n in range(1,NT+1): # assign CMYK values to each point below
            if LydMin <= LYAP[n-1,m] <= LydMax:
                c2[n,m,:] = cmap_ly(0.5*(LYAP[n-1,m]+LydMax)/LydMax)
            else: # outside of lyap cutoff
                if LYAP[n-1,m] <= LydMin:
                    c2[n,m,:] = cmap_ly(0)
                elif LYAP[n-1,m] >= LydMax:
                    c2[n,m,:] = cmap_ly(1)
    # main plotting
    fig0 = pl.figure(figsize=(5.0,3.5))
    ax0 = fig0.add_subplot(111)
    ax0.set_xlim([0,np.pi])
    ax0.set_ylim([-prange,prange])
    if webres == False:
        Navg = 0.5*(Np+Nth) 
        msize = np.amax([0.1,5.0/Navg])
        ax0.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
        ax0.set_xticklabels([r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'])
        ax0.set_yticks([-np.pi,-2*np.pi/3,-np.pi/2,-np.pi/3,0,np.pi/3,np.pi/2,2*np.pi/3,np.pi])
        ax0.set_yticklabels([r'$-\pi$',r'$-2\pi/3$',r'$-\pi/2$',r'$-\pi/3$',r'$0$',r'$\pi/3$',r'$\pi/2$',r'$2\pi/3$',r'$\pi$'])
    else: # need the points to appear on lower-resolution images for web display
        msize = 0.4
        ax0.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    for i in range(0,NL):
        ax0.scatter(np.remainder(cQ_plot[0,:,i],np.pi),cQ_plot[1,:,i],color = c2[:,i,:],s = msize,marker = 's',edgecolors = 'none')
        ax0.scatter(np.remainder(-cQ_plot[0,:,i],np.pi),-cQ_plot[1,:,i],color = c2[:,i,:],s = msize,marker = 's',edgecolors = 'none')
        printProgressBar(i, NL, suffix = 'Plotting', length = 50)
    if webres == False:
        fig0.savefig('XZK_StroboPS_'+imgname+'.pdf')
        fig0.savefig('XZK_StroboPS_'+imgname+'.png',dpi = 300)
    else:
        fig0.savefig(cur_path+r'\XZK_StroboPS_'+imgname+r'_webani\webani_'+str(aniframe).zfill(3)+'.png',dpi = 150,bbox_inches='tight')
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
    ax1.imshow(cbar.T,cmap=cmap_ly,origin='lower',aspect=15/(np.absolute(LydMax)),extent=(0,1,LydMin,LydMax))
    fig1.tight_layout()
    fig1.savefig('LYAP_CBAR_'+imgname+'.pdf')#,dpi = 300)
    ax1.cla(); fig1.clf()
    print('Complete!')

# Print iterations progress
# adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration+1) / float(total)))
    filledLength = int(length * (iteration+1) // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total-1: 
        print('')
