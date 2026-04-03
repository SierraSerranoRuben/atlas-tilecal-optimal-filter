# Optimal Filtering for ATLAS Tile Calorimeter (HL-LHC)



This project was developed as part of the evaluation test for admission into Google Summer of Code (GSoC) in collaboration with CERN (HSF).



## Overview



The ATLAS Tile Calorimeter is used to reconstruct hadronic energy in high-energy physics experiments. Under High-Luminosity LHC (HL-LHC) conditions, signal reconstruction becomes extremely challenging due to:

- Electronic noise  
- Out-of-time pileup  
- Overlapping signals from multiple interactions  

The objective is to accurately reconstruct:

- Energy (signal amplitude)  
- Time (phase)  

from digitized detector samples.

---

## Methodology
The reconstruction is based on the Optimal Filtering (OF) algorithm:

- Signal modeled using a first-order Taylor expansion  
- Constrained optimization using Lagrange multipliers:
- Amplitude unbiasedness  
- Phase invariance  
- Pedestal suppression  
- Noise modeled through a covariance matrix  

The optimal weights are obtained analytically:

$$w = R^{-1}C(C^{T}R^{-1}C)^{-1}$$

---
## Challenges

- Strongly correlated noise due to pileup  
- Ill-conditioned covariance matrix (R)  
- Numerical instability in matrix inversion  
- Breakdown of timing reconstruction under high luminosity  
---
## Regularization Strategy

To stabilize the system, Tikhonov regularization was applied:

$$R\_reg = R + \lambda I$$

- Grid search used to determine optimal $\lambda$  
- Trade-off between numerical stability and physical accuracy  

---

## Results

- Standard Optimal Filtering shows large reconstruction errors under HL-LHC conditions  
- Regularization improves conditioning but does not fully recover performance  
- Timing reconstruction becomes unstable due to near-zero amplitudes  

---
## Conclusion
This work highlights the limitations of linear filtering methods under extreme pileup conditions and motivates the use of machine learning approaches (e.g., CNNs) for signal reconstruction.

---
## Technologies

- Python  
- NumPy 
- PyTorch  
- Matplotlib  

---
## Author
Rubén Sierra Serra
