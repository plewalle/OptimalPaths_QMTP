'''
Philippe Lewalle
AN Jordan Group at University of Rochester (2015-2021)
KB Whaley Group at UC Berkeley (2021-current)

Running MLP Plots for Jordan & Siddiqi Quantum Measurement Book.
This script creates figures like Fig. 9.2 of the textbook.

Code Written August & September 2022, Revised November & December 2024,
based on earlier codes from projects completed as a graduate student in AN Jordan's group: 
    arXiv:1612.07861, arXiv:1612.03189, arXiv:1803.07615

'''
# code tested in python 3.12.3
# package versions this code was last debugged in are also listed below
import numpy as np #                                               numpy 1.26.4
from matplotlib import pyplot as pl #                          matplotlib 3.9.2
import matplotlib.animation as animation
from PIL import Image
import os; cur_path = os.path.dirname(__file__)
import QMTP_MLP_aux as AUX

# Divergence of the co-states for extremely rare readout sequences / post-selection boundary conditions is typical.
# The integrator used here is built to identify and remove such diverging paths from the final calculations / plots.  
np.seterr(over='ignore',invalid='ignore',divide='ignore') # consequently the internal generation of NaN is not of interest here.

''' 
Langrangian Manifold: Driven Homodyne Flourescence
'''

x0 = 0.0; z0 = 1.0; Qi = np.asarray([x0,z0]) # initial state
gam = 1.0; om = 2.0; eta = 0.5; eqargs = np.asarray([gam,om,eta]) # constants: decay rate, Rabi drive rate, measurement efficiency

Tf = 2.5

# range of initial "momenta", and momentum mesh resolution
pxmin = -5.0; pxmax = 5.0; Nx = 150 
pzmin = -5.0; pzmax = 5.0; Nz = 150 
Pmesh = np.asarray([[pxmin,pxmax,Nx],[pzmin,pzmax,Nz]])
# final-time plot with labeled axes
imgs_dir = 'sparse_QMTheoryPractice'
AUX.LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,imgs_dir)

# frames for animation (with un-labeled axes)
pxmin = -5.0; pxmax = 5.0; Nx = 200 
pzmin = -5.0; pzmax = 5.0; Nz = 200 
Pmesh = np.asarray([[pxmin,pxmax,Nx],[pzmin,pzmax,Nz]])
NF = 350
Tf = 3.5
AUX.LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,imgs_dir,animate = True,Nframe = NF)

#Nx = 400; Nz = 400 # suggested for dense sampling
#Pmesh = np.asarray([[pxmin,pxmax,Nx],[pzmin,pzmax,Nz]])
#AUX.LM_4DPS_BlochPlotter(Qi,Tf,Pmesh,eqargs,'_dense_QMTheoryPractice')

# collect animation frames into .gif image
print('Assembling frames into .gif animations ... ',end = '')
# low resolution for web use first
img_arr = []
for n in range(0,NF,2):
    imgfr = Image.open(cur_path+r'\LM_4DPS_'+imgs_dir+r'_webani\webani_'+str(n).zfill(3)+'.png')
    img_arr.append(imgfr)

figLM, axLM = pl.subplots(figsize=(5.,5.),frameon=False)
axLM.set_axis_off()
imax = axLM.imshow(img_arr[0],animated = True)
def update_frames(i):
    imax.set_array(img_arr[i])
    return imax,
animation_LM = animation.FuncAnimation(figLM, update_frames, frames=len(img_arr), interval=10, blit=True,repeat_delay=500,)
animation_LM.save(cur_path+r'\LM_4DPS_'+imgs_dir+r'_webani\webani.gif', writer = animation.PillowWriter(fps=24,metadata=dict(artist='plewalle'),bitrate=600))
figLM.clf()

# high resolution version of the same
img_arr = []
for n in range(0,NF):
    imgfr = Image.open(cur_path+r'\LM_4DPS_'+imgs_dir+r'_ani\ani_'+str(n).zfill(3)+'.png')
    img_arr.append(imgfr)

figLM, axLM = pl.subplots(figsize=(5.,5.),frameon=False)
axLM.set_axis_off()
imax = axLM.imshow(img_arr[0],animated = True)
animation_LM = animation.FuncAnimation(figLM, update_frames, frames=len(img_arr), interval=10, blit=True,repeat_delay=500,)
animation_LM.save(cur_path+r'\LM_4DPS_'+imgs_dir+r'_ani\ani.gif', writer = animation.PillowWriter(fps=24,metadata=dict(artist='plewalle'),bitrate=1200))
figLM.clf()
print('Done!')

'''
Stroboscopic Phase Portrait: X Measurement with kicked Z Measurement
'''

prange = np.pi+1.0 # cut the mesh a bit above the first large resonance
Np = 40
Nth = 40

off = 1.0 # base measurement strengths
amp = 0.97 # amplitude of the z-measurement kick (tau ~ off - amp). 
# Use any value in [0,1), where 0 is trivially integrable, and a transition to chaos occurs as amp increases
sig = 0.01 # width of the z-measurement kick
eqargs_xz = np.asarray([off, amp, sig]) 
Tf = 100.0 # final integration time
dt_plot = 1.0
dt_comp_factor = 1000
dth0 = 0.01

imgnm = 'QMTheoryPractice_k97_T100_N50'
AUX.StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,imgnm,relerr_step = 5.0e-7,aberr_step = 5.0e-7)

# now the animated version, scanning through kick-strength
NF = 200
prange = np.pi/2 # cut the mesh to highlight central islands of stability
Np = 40
Nth = 40
imgs_dir = 'QMTheoryPractice_hikick_T100_N40'
amp_arr = np.linspace(0.9,0.995*off,NF)
for n in range(0,NF):
    eqargs_xz = np.asarray([off, amp_arr[n], sig])
    AUX.StroboPS_XZK(Nth,Np,prange,Tf,dt_plot,dt_comp_factor,dth0,eqargs_xz,imgs_dir,webres = True,aniframe = n,relerr_step = 1.0e-6,aberr_step = 1.0e-6)

# collect animation frames into .gif image
print('Assembling frames into .gif animations ... ',end = '')
# low resolution for web use first
img_arr = []
for n in range(0,NF,2):
    imgfr = Image.open(cur_path+r'\XZK_StroboPS_'+imgs_dir+r'_webani\webani_'+str(n).zfill(3)+'.png')
    img_arr.append(imgfr)
    
figSP, axSP = pl.subplots(figsize=(5.0,3.5),frameon=False)
axSP.set_axis_off()
imax = axSP.imshow(img_arr[0],animated = True)
animation_SP = animation.FuncAnimation(figSP, update_frames, frames=len(img_arr), interval=40, blit=True,repeat_delay=500,)
animation_SP.save(cur_path+r'\XZK_StroboPS_'+imgs_dir+r'_webani\webani.gif', writer = animation.PillowWriter(fps=24,metadata=dict(artist='plewalle'),bitrate=1200))
figSP.clf()
print('Done!')
