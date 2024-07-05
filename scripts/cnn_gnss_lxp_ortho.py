"""
Compute orthogonal basis for the low-rank matrix $F_\sigma$

"""

import cnn_lxp
import torch
import utils
from itertools import product
import numpy as np
import os


if __name__ == "__main__":
    nens = 25
    ntest = 192

    # 1
    i_ens = 1
    i_testno = 0

    lxp = cnn_lxp.CNNLinearExpansion(i_ens = i_ens)
    lxp.load_cnn()

    # load test input-output pair
    i_testno = 0

    test_input = lxp.Model._get_data1_tensor(i_testno, inout='in')
    test_output = lxp.Model._get_data1_tensor(i_testno, inout='out')

    test_input1 = torch.unsqueeze(test_input, 0)
    net_output = lxp.net(test_input1)

    # compute linear expansion
    out = lxp.compute_lxp(test_input1)
    lxp.compute_basis(i_testno)

    ## compute projection to computed basis
    lxp.compute_coeff_stats()

    suffix = "_n{:02d}_i{:04d}".format(i_ens, i_testno)
    suffix1 = "_n{:02d}_i{:04d}".format(i_ens, i_testno)

    psi = np.load("_output/UX{:s}.npy".format(suffix))
    phi = np.load("_output/UY{:s}.npy".format(suffix))

    phi_singvals = np.load("_output/sX{:s}.npy".format(suffix))
    psi_singvals = np.load("_output/sY{:s}.npy".format(suffix))


    fig, ax = utils.plot_singvals(phi_singvals, psi_singvals)
    fig.savefig("_plots/basis_sampling{:s}.pdf".format(suffix), dpi=200)

    # plot coefficient svd

    fname = "_output/corr{:s}.npy".format(suffix)
    nrelus = 17

    if os.path.exists(fname):
        corr = np.load(fname)
    else:
        coeffX = np.memmap("_output/coeffX{:s}.dat".format(suffix),
                            mode="r",
                            shape=(nrelus, 2**nrelus - 1), dtype=float)
        coeffY = np.memmap("_output/coeffY{:s}.dat".format(suffix),
                            mode="r",
                            shape=(nrelus, 2**nrelus - 1), dtype=float)
        corr = coeffX.dot(coeffY.T)
        np.save(fname, corr)

    fig, axs = utils.plot_corr(corr)
    fig.savefig("_plots/corr{:s}.pdf".format(suffix), dpi=200)

    u, s, v0 = np.linalg.svd(corr)

    psin = psi[:, :nrelus].dot(v0.T)
    phin = phi[:, :nrelus].dot(u)

    fname = "_output/psin{:s}.npy".format(suffix)
    np.save(fname, psin)

    fname = "_output/phin{:s}.npy".format(suffix)
    np.save(fname, phin)
    
