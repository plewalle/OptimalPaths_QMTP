# OptimalPaths_QMTP
This is a code sample detailing the creation of certain figures appearing in Jordan and Siddiqi's recent textbook on Quantum Measurement. 
In particular, these codes execute simulation of quantum trajectories, and batch integration and illustration of quantum trajectories conditioned on extremal-probability sequences of continuous readout events. 
The current version (v1, uploaded November 2024) has been debugged in Python 3.9.7, with dependencies on numpy, scipy, and matplotlib.pyplot.

Profs. Andrew Jordan and Irfan Siddiqi recently published a textbook, Quantum Measurement: Theory and Practice (QMTP, Cambridge University Press, 2024). 

https://doi.org/10.1017/9781009103909

The codes included herein can be used to reproduce Figs. 9.2 and 9.10 of the text, or variants on them. 

Specifically: QMTP_MLP_aux.py and QMTP_MLP_run.py are associated with Fig. 9.2, and QMTP_SQT_aux.py and QMTP_SQT_run.py are associated with Fig. 9.10.  

The current v1 of this code has Kraus Operators and equations of motion hard-coded for the examples shown in the textbook and in prior published works on the topic. Look for an update soon with added functionality to apply the basic integration tools to a wider range of examples, along with some refinements for speed and ease of use.

The original published research works on which Fig. 9.2 and the associated codes are based are:

[1] Philippe Lewalle, Areeya Chantasri, and Andrew N. Jordan. Prediction and characterization of multiple extremal paths in continuously monitored qubits. Phys. Rev. A 95 042126 (2017) or arxiv:1612.07861.

[2] M. Naghiloo, D. Tan, P. M. Harrington, P. Lewalle, A. N. Jordan, and K. W. Murch. Quantum Caustics in Resonance Fluorescence Trajectories. Phys. Rev. A 96 053807 (2017) or arxiv:1612.03189.

[3] Philippe Lewalle, John Steinmetz, Andrew N. Jordan. Chaos in Continuously Monitored Quantum Systems: An Optimal Path Approach. Phys. Rev. A 98, 012141 (2018) or arxiv:1803.07615.

The original publisehd research work on which Fig. 9.10 and the associated codes are based is:

[4] Philippe Lewalle, Cyril Elouard, Sreenath K. Manikandan, Xiao-Feng Qian, Joseph H. Eberly, Andrew N. Jordan. Entanglement of a pair of quantum emitters via continuous fluorescence measurements: a tutorial. Advances in Optics and Photonics, 13 (3), 517-583 (2021), or arxiv:1910.01206.

Subsequent works reviewing this or related material include:

[5] Philippe Lewalle. Quantum Trajectories and their Extremal-Probability Paths: New Phenomena and Applications. Ph.D. Dissertation, University of Rochester (2021).

[6] Andrew N. Jordan and Irfan A. Siddiqi. Quantum Measurement: Theory and Practice. Cambridge University Press, 2024.
