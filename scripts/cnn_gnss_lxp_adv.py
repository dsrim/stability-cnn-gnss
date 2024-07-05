"""
1. Pick ensemble model index in range(25), test example index in range(192)

2. Compute adversarial examples using PGD

Compute and analyze the low-rank expansion

3. Compute basis (without the orthonormal basis)

4. Try basis functions as adversarial perturbations

5. Try projecting and filtering PGD adv expl, make plots

"""

import utils
from itertools import product
import numpy as np


if __name__ == "__main__":
    nens = 25
    ntest = 192

    # 1
    i_ens = 1
    i_testno = 0

    # 2
    utils.pgd_adv(i_ens=i_ens,
                i_testno=i_testno)

    # 3-5
    utils.lxp_analysis(i_ens=i_ens,
                    i_testno=i_testno)
