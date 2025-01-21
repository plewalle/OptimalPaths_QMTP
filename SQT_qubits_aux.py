'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

code created December 2024

Stochastic Quantum Trajectory Simulations for Qubit Systems.
Included here is:
    1. A general implementation of Rouchon & Ralph's trajectory simulation 
        method. See Phys. Rev. A 91, 012118 (2015) or arXiv:1410.5345
    2. Coordinate decomposition of density matrices for systems composed 
        entirely of qubits.
These tools give one methodological option for implementing the two-qubit
entanglement via fluorescence example from QMTP. The methods here are easy to
adapt to a much wider range of problems. 

Sec. 1 is completely general, while Sec. 2 contains an assumption that the
system is composed exclusively of qubits. We apply this to a two-qubit problem
in the associated example code, but Sec. 2 is suitable for N-qubit systems. 
Generalization of Sec. 2 for N qudits instead of qubits could be achieved by 
adapting e.g. the methods of https://arxiv.org/abs/0806.1174 ; This is not done 
here simply because it is not necessary for the two-qubit example of interest 
that appears in QMTP.
'''
# code tested in python 3.12.3
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
import numba as nb #                                               numba 0.60.0
from numpy.random import normal as gauss # draw a random number from Gaussian
import functools as ft
import string

''' CONTENTS

Sec. 1. General Rouchon/Ralph Diffusive Trajectory Simulation ............. 047 
    a. Rouchon/Ralph Steppers ............................................. 050
    b. Rouchon/Ralph Trajectory Integration ............................... 093
    
Sec. 2. Qubit System State Parameterization ............................... 176
    a. Single Qubit Parameterization and Operations ....................... 183
    b. Multi-Qubit Coordinate Generalization .............................. 246

'''

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

''' 111111111111111111111111111111111111111111111111111111111111111111111111111
SEC 1: General SQT Simulation Methods
111111111111111111111111111111111111111111111111111111111111111111111111111 '''

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Rouchon/Ralph Steppers; see arXiv:1410.5345 '''

# stepper for a single measurement channel
# see arXiv:1410.5345 by Rouchon + Ralph
# expects a single L as measurement input
@nb.njit(inline='always')
def rho_Rouchon_Update_single(rho,OM,L,Ld,eta,dt,dW): # dW = dWt(Nmeas,dt)
    # Nmeas = 1
    Ld = DAG(L)
    N = rho.shape[0] # size of state
    # compute the readout (signal + noise)
    dr = np.sqrt(eta)*np.trace(L @ rho + rho @ Ld)*dt + dW[0]
    # compute the effective dynamical operator
    Mop = np.eye(N) - 1.0j*OM*dt - 0.5*(Ld @ L)*dt + np.sqrt(eta)*dr*L + 0.5*eta*L@L*(dr*dr-dt) 
    # compute the actual dynamical update
    numer = Mop @ rho @ DAG(Mop) + (1.0-eta)*(L @ rho @ Ld)*dt
    return numer/np.trace(numer)

# stepper for several simultaneous measurements
# see arXiv:1410.5345 by Rouchon + Ralph
# expects a a list of L as measurement input
def rho_Rouchon_Update_multi(rho,OM,L,eta,dt,dW): # dW = dWt(Nmeas,dt)
    Nmeas = L.shape[0]
    Ld = np.zeros(L.shape) + 1.0j*np.zeros(L.shape); # om = np.zeros(L.shape) + 1.0j*np.zeros(L.shape)
    for nm in range(0,Nmeas):
        Ld[nm,:,:] = DAG(L[nm,:,:])
    N = rho.shape[0] # size of state
    Mop = np.eye(N) - 1.0j*OM*dt
    dr = np.zeros(Nmeas)
    for nm in range(0,Nmeas): # compute the main dynamical operator
        dr[nm] = np.real(np.sqrt(eta[nm])*np.trace(L[nm,:,:] @ rho + rho @ Ld[nm,:,:])*dt) + dW[nm]
        Mop += - 0.5*(Ld[nm,:,:] @ L[nm,:,:])*dt + np.sqrt(eta[nm])*dr[nm]*L[nm,:,:] 
    kdel = np.eye(Nmeas) # compute the cross-term numerical correction
    for j in range(0,Nmeas):
        for k in range(0,Nmeas):
            Mop += 0.5*np.sqrt(eta[j]*eta[k])*(L[j,:,:] @ L[k,:,:])*(dr[j]*dr[k] - kdel[j,k]*dt)
    # compute the actual dynamical update
    numer = Mop @ rho @ DAG(Mop)
    for nm in range(0,Nmeas):
        numer += (1.0-eta[nm])*(L[nm,:,:] @ rho @ Ld[nm,:,:])*dt
    return numer/np.trace(numer)

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Rouchon/Ralph Integration '''

def dWt(nshape,dt): # Wiener noise
    return gauss(loc = 0.0, scale = np.sqrt(dt), size = nshape)

def Trajectory_Rouchon(rh0,OM,L,eta,dt,T,return_q = False,dW = None):
    ct = np.arange(0,T+dt,dt); NT = ct.size
    if L.ndim == 2: # L is a matrix
        Nmeas = 1
        Ld = DAG(L)
    elif L.ndim == 3: # L is a list of matrices, and eta should also be a list/array of this length
        Nmeas = L.shape[0]
        Ld = np.zeros(L.shape); # om = np.zeros(L.shape) + 1.0j*np.zeros(L.shape)
        for nm in range(0,Nmeas):
            Ld[nm,:,:] = DAG(L[nm,:,:])
    if dW == None: # if dW was not given as input, create the noise realization now
        dW = dWt((Nmeas,NT),dt)
    NR = rh0.shape[0]
    c_rho = np.zeros((NT,NR,NR)) + 1.0j*np.zeros((NT,NR,NR))
    c_rho[0,:,:] = rh0 # initialize density matrix time slices
    for k in range(1,NT):
        if Nmeas == 1:
            c_rho[k,:,:] = rho_Rouchon_Update_single(c_rho[k-1,:,:],OM,L,Ld,eta,dt,dW[:,k])
        else:
            c_rho[k,:,:] = rho_Rouchon_Update_multi(c_rho[k-1,:,:],OM,L,Ld,eta,dt,dW[:,k])
        printProgressBar(k, NT, prefix = 'Trajectory Integration:')
    if return_q == False: # return time-sampled density matrix
        return ct, c_rho, dW
    else: # return time-sampled coordinates instead
        NQ = np.rint(np.log2(NR)).astype(np.int32)
        cq = qvec_NNQ(c_rho,NT,NQ)
        return ct, cq, dW

def LowMem_Trajectory_Rouchon(rh0,OM,L,eta,dt,dtsf,T,return_q = False,dW = None,rro = False):
    ct = np.arange(0,T+dt,dt); NT = ct.size
    dtp = dt*dtsf
    Ntp = np.rint(T/dtp).astype(np.int64)+1 # number of timesteps saved
    ctp = np.arange(0,T+dtp,dtp)
    if L.ndim == 2: # L is a matrix
        Nmeas = 1
        Ld = DAG(L)
        if rro == True:
            cr = np.zeros(NT-1)
    elif L.ndim == 3: # L is a list of matrices, and eta should also be a list/array of this length
        Nmeas = L.shape[0]
        Ld = np.zeros(L.shape) + 1.0j*np.zeros(L.shape); # om = np.zeros(L.shape) + 1.0j*np.zeros(L.shape)
        for nm in range(0,Nmeas):
            Ld[nm,:,:] = DAG(L[nm,:,:])
        if rro == True: 
            cr = np.zeros((NT-1,Nmeas))
    if dW == None: # if dW was not given as input, create the noise realization now
        dW = dWt((Nmeas,NT-1),dt)
    NR = rh0.shape[0]
    c_rho = np.zeros((Ntp,NR,NR)) + 1.0j*np.zeros((Ntp,NR,NR)) # initialize array for the state coordinates
    c_rho[0,:,:] = rh0 # initialize density matrix time slices
    for k in range(1,NT):
        if Nmeas == 1:
            rh0 = rho_Rouchon_Update_single(rh0,OM,L,Ld,eta,dt,dW[:,k-1])
            if rro == True:
                cr[k] = np.real(np.sqrt(eta)*np.trace(L @ rh0 + rh0 @ Ld)*dt) + dW[:,k-1]
        else:
            rh0 = rho_Rouchon_Update_multi(rh0,OM,L,eta,dt,dW[:,k-1])
            if rro == True:
                for n in range(0,Nmeas):
                    cr[k-1,n] = np.sqrt(eta[n])*np.trace(L[n,:,:] @ rh0 + rh0 @ Ld[n,:,:])*dt + dW[n,k-1]
        if k%dtsf == 0: # save a point
            c_rho[np.rint(k/dtsf).astype(np.int64),:,:] = rh0
        printProgressBar(k, NT, prefix = 'Trajectory Integration:')
    if rro == False:
        if return_q == False: # return time-sampled density matrix
            return ctp, c_rho
        else: # return time-sampled coordinates instead
            NQ = np.rint(np.log2(NR)).astype(np.int32)
            cq = qvec_NNQ(c_rho,Ntp,NQ)
            return ctp, cq
    else:
        if return_q == False: # return time-sampled density matrix
            return ctp, c_rho, ct, cr
        else: # return time-sampled coordinates instead
            NQ = np.rint(np.log2(NR)).astype(np.int32)
            cq = qvec_NNQ(c_rho,Ntp,NQ)
            return ctp, cq, ct, cr

''' 222222222222222222222222222222222222222222222222222222222222222222222222222
SEC 2: Coordinate Parameterization for Qubits + Basic Functions
222222222222222222222222222222222222222222222222222222222222222222222222222 '''

@nb.njit(inline='always')
def DAG(op):
    return np.conjugate(np.transpose(op))

''' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Single Qubit Operations '''

''' Basic Operators '''

# N x N Identity
@nb.njit(inline = 'always')
def ID(N):
    return np.eye(N) + 1.0j*np.zeros((N,N))

# Pauli X
@nb.njit(inline = 'always')
def sigX():
    mat = np.zeros((2,2)) + 1.0j*np.zeros((2,2))
    mat[0,1] = 1.0
    mat[1,0] = 1.0
    return mat
# Pauli Y
@nb.njit(inline = 'always')
def sigY():
    mat = np.zeros((2,2)) + 1.0j*np.zeros((2,2))
    mat[0,1] = -1.0j
    mat[1,0] = 1.0j
    return mat
# Pauli Z
@nb.njit(inline = 'always')
def sigZ():
    mat = np.zeros((2,2)) + 1.0j*np.zeros((2,2))
    mat[0,0] = 1.0
    mat[1,1] = -1.0
    return mat

# Pauli Ladders
@nb.njit(inline = 'always')
def sigm():
    mat = np.zeros((2,2)) + 1.0j*np.zeros((2,2))
    mat[1,0] = 1.0
    return mat
@nb.njit(inline = 'always')
def sigp():
    return np.transpose(sigm())

''' Bloch Parameterization '''

# Convert Qubit Density Matrix to Bloch Vector
def Rho2_to_Qvec(rho):
    x = np.trace(np.dot(sigX(),rho))
    y = np.trace(np.dot(sigY(),rho))
    z = np.trace(np.dot(sigZ(),rho))
    qvec = np.real_if_close(np.asarray([x,y,z]),tol = 1000)
    # cast to real values assuming imaginary parts are on the order expected
    # from machine-precision limitations / rounding errors. 
    # If they are not and output is genuinely complex, this is likely to 
    # result in errors further along. 
    return qvec

# Convert Bloch Vector to Qubit Density Matrix
def Rho2_from_Qvec(qvec):
    return 0.5*(ID(2)+qvec[0]*sigX()+qvec[1]*sigY()+qvec[2]*sigZ())

def Rho2_th(theta): # xz Bloch plane
    return np.asarray([[np.cos(0.5*theta)**2,np.cos(0.5*theta)*np.sin(0.5*theta)],[np.cos(0.5*theta)*np.sin(0.5*theta),np.sin(0.5*theta)**2]])

''' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Multi-Qubit Generalization '''

# N-Qubit Gell--Mann Matrix
# plist e.g. = ['X','I','X','Y',...]
def GM_NQ(plist):
    NQ = len(plist)
    sig_list = []
    for n in range(0,NQ):
        if plist[n] == "I":
            sig_list.append(ID(2))
        if plist[n] == "X":
            sig_list.append(sigX())
        if plist[n] == "Y":
            sig_list.append(sigY())
        if plist[n] == "Z":
            sig_list.append(sigZ())
    GAM = ft.reduce(np.kron,sig_list)
    return GAM

digs = string.digits + string.ascii_letters

# https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base
# printing numbers base 4 maps onto permuting combinations of I, X, Y, Z
def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1
    x *= sign
    digits = []
    while x:
        digits.append(digs[x % base])
        x = x // base
    if sign < 0:
        digits.append('-')
    digits.reverse()
    return ''.join(digits)

def strb4_to_Pauli(base_4_string):
    intlist = [char for char in base_4_string]
    pauli_list = []
    for intstring in intlist:
        if intstring == '0':
            pauli_list.append('I')
        if intstring == '1':
            pauli_list.append('X')
        if intstring == '2':
            pauli_list.append('Y')
        if intstring == '3':
            pauli_list.append('Z')
    return pauli_list 

def QGam_Ordering(NQ):
    Gam_list = []
    for n in range(1,(2**NQ)**2):
        Gam_list.append(strb4_to_Pauli(int2base(n,4).zfill(NQ)))
    return Gam_list

# array of Gell-Mann Matrices for NQ-qubit state
def GM_ARRAY(NQ):
    GM = np.zeros(((2**NQ)**2-1,2**NQ,2**NQ)) + 1.0j*np.zeros(((2**NQ)**2-1,2**NQ,2**NQ))
    Gam_List = QGam_Ordering(NQ)
    for n in range(0,(2**NQ)**2-1):
        GM[n,:,:] = GM_NQ(Gam_List[n])
    return GM

# assemble density matrix from the NQ coordinates
def rho_NQ(qvec,NQ):
    rho = ID(2**NQ)
    gma = GM_ARRAY(NQ)
    for j in range(0,(2**NQ)**2-1):
        rho += gma[j]*qvec[j]
    return rho/(2**NQ)
# do the reverse: extract qvec given the NQ density matrix
def qvec_NQ(rho,NQ): 
    qvec = np.zeros((2**NQ)**2-1)
    GMA = GM_ARRAY(NQ)
    for n in range((2**NQ)**2-1):
        qvec[n] = np.real_if_close(np.trace(GMA[n,:,:] @ rho),1.0e3)
    return qvec

# batched versions of the above pair (faster for repeated use on same-sized arrays)
# a pair of helper functions for batching first
@nb.njit(inline = 'always') # then compile the repeated part locally
def construct_rho(qvec,gma,nq):
    rho = ID(2**nq)
    for j in range(0,(2**nq)**2-1):
        rho += gma[j,:,:]*qvec[j]
    return rho/(2**nq)
@nb.njit(inline = 'always') # then compile the repeated part locally
def construct_q(rho,gma,nq):
    qvec = np.zeros((2**nq)**2-1)
    for j in range(0,(2**nq)**2-1):
        qvec[j] = np.real(np.trace(gma[j,:,:] @ rho))
    return qvec
# call these functions for batch conversion
def rho_NNQ(qvecs,N,NQ):
    GMA = GM_ARRAY(NQ) # run this part only once
    if type(N) == int:
        RHO = np.zeros((N,2**NQ,2**NQ)) + 1.0j*np.zeros((N,2**NQ,2**NQ))
        for n in range(0,N):
            RHO[n,:,:] = construct_rho(qvecs[n,:],GMA,NQ)
    elif type(N) == tuple: # this assumes N has length 2, because we currently have no use-case beyond this for this code
        RHO = np.zeros((*N,2**NQ,2**NQ)) + 1.0j*np.zeros((N[0],N[1],2**NQ,2**NQ))
        # ND = 2 # len(N); for ND != 2 this function should be modified! Brute-force generalization requires an additional nested loop for each dimension.
        for m in range(0,N[0]):
            for n in range(0,N[1]):
                RHO[m,n,:,:] = construct_rho(qvecs[m,n,:],GMA,NQ)
    return RHO
''' Regarding the len(N) == 1 or len(N) == 2 choices above and below:
The typical use cases we have here are i) N = Ntime, where we have Ntime 
time-slices of a single stochastic trajectory, or ii) N = (N_SQT,Ntime), where 
we have Ntime time-slices of an N_SQT-sized ensemble of trajectories. These 
codes will not work as is on cases outside of these shape options.'''
def qvec_NNQ(rhos,N,NQ):
    GMA = GM_ARRAY(NQ) # run this part only once
    if type(N) == int:
        QVEC = np.zeros((N,(2**NQ)**2-1))
        for n in range(0,N):
            QVEC[n,:] = construct_q(rhos[n,:,:],GMA,NQ)
    elif type(N) == tuple: # this assumes N has length 2, because we currently have no use-case beyond this for this code
        QVEC = np.zeros((*N,(2**NQ)**2-1))
        # ND = 2 # len(N); for ND != 2 this function should be modified! Brute-force generalization requires an additional nested loop for each dimension.
        for m in range(0,N[0]):
            for n in range(0,N[1]):
                QVEC[m,n,:] = construct_q(rhos[m,n,:,:],GMA,NQ)
    return QVEC