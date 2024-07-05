## Code repository 

This is a code repository containing Python scripts used to compute results in
the manuscript [*A Stability Analysis of Neural Networks and Its Application to
Tsunami Early Warning*](https://doi.org/10.31223/X5D954) (DOI:10.31223/X5D954).

Following is a BibTeX example for citing this code repository.
```
@misc{RimSuri2024,
  doi = {10.5281/zenodo.12663552},
  url = {https://zenodo.org/doi/10.5281/zenodo.12663552},
  author = {Donsub Rim and Sanah Suri},
  title = {Code repository},
  year = {2024}
}
```
[![DOI](https://zenodo.org/badge/824469379.svg)](https://zenodo.org/doi/10.5281/zenodo.12663552)

## Requirements

The code was tested with Python version ``3.8.5`` with the following packages

- [``pytorch``](https://pytorch.org) version ``1.4.0`` (no GPU support required)
- [``matplotlib``](https://matplotlib.org) version ``3.8.0``

## Download data

Download all the necessary data (960MB storage required), stored in the Box
folder
```
https://wustl.box.com/v/gnss-adverse
```
Required files are in the sub-directories
```
data/_sjdf
scripts/_output
```
Alternatively, these files can be reproduced using the code in the repo [ML_GNSS_SJdF_2022](https://github.com/dsrim/ML_GNSS_SJdF_2022) associated with the GRL paper
[*Tsunami Early Warning from Global Navigation Satellite System Data using Convolutional Neural Networks*](https://doi.org/10.1029/2022GL099511).


## Toy example
Run the script that illustrates the low-rank expansion in an untrained toy NN model
```
cd scripts && python toy_example.py
```

## Tsunami model example

Run the script that computes the low-rank expansion and analyzes adversarial
examples obtained from PGD
```
cd scripts && python cnn_gnss_lxp_adv.py
```
(Optional) Run the script to compute all terms in the Householder expansion to
compute an SVD of the nonlinear part in the low-rank expansion. This is a
computational expensive operation, so a pre-computed result is made available in
the Box folder.
```
cd scripts && python cnn_gnss_lxp_ortho.py
```
Next, run the script to perform stability analysis against additive noise
```
cd scripts && python cnn_gnss_lxp_noise.py
```
(The last script requires outputs from the second.)

## Contact

For any questions regarding the code repository, contact
Donsub Rim (rim@wustl.edu) or Sanah Suri (s.sanah@wustl.edu)