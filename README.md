# Focused Sampling for Low-Cost and Accurate Ehrenfest Modeling of Cavity Quantum Electrodynamics
Authors: Ming-Hsiu Hsieh, Alex Krotz, Roel Tempelaar

Publication: [*J. Chem. Theory Comput.* **2025** (accepted)](https://arxiv.org/abs/2506.21702)


## Usage of the code
The code has been run under Python 3.10.15 with the following packages of the version specified in parentheses.

Required packages: Numpy (1.26.4), Numba (0.60.0), Ray (2.38.0), matplotlib (3.9.2)

To calculate DC-MF and MF dynamics, go to line 15 in `main.py`. For DC-MF dynamics, keep `from mixQC_DCMF_Foc import runCalc`. For MF dynamics, change it to `from mixQC_MF_Foc import runCalc`.

To simulate the small half-wavelength cavity (Section 3 in the paper), rename `input_ShortCavity.py` to `input.py`. Alternatively, to simulate the long cavity (Section 4 in the paper), rename `input_LongCavity.py` to `input.py`

In the function `Wigner()` in both `mixQC_DCMF_Foc.py` and `mixQC_MF_Foc.py`, you can choose to use Wigner sampling (first 5 lines) or Focused sampling (last 4 lines).

To start the run, execute 
```
python main.py
```


Alternatively, to run a simulation with quantum exact reference (CISD), and execute:
```
python CISD.py
```
The input file is the same as the ones for DC-MF and MF calculations.
