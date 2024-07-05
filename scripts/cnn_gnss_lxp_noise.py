"""
Compute input perturbations made from generic additive noise

"""

import cnn_lxp
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "monospace"
import matplotlib.pyplot as plt
import utils
import os, sys


if __name__ == "__main__": 

    ens_index = 1
    test_index = 0

    lxp = cnn_lxp.CNNLinearExpansion()
    lxp.load_cnn()

    UX, UY = utils.load_basis(ens_index, test_index)
    x_adv, y_adv = utils.load_adv(ens_index, test_index)
    phin, psin = utils.load_psi_phi(ens_index, test_index)

    x, y = utils.load_inout_npy(test_index, lxp)

    x_adv = torch.tensor(x, dtype=torch.float32)
    torch.manual_seed(1)
    x_adv += torch.cumsum(torch.randn(x_adv.shape), axis=-1)
    x_adv = x_adv.reshape(1, 180, 512)
    y_adv = lxp.net(x_adv)

    suffix = utils.get_suffix(ens_index, test_index)

    dx = (x_adv - x).detach().numpy()

    # compute projection and filtered perturbation
    dx_proj = UX @ (UX.T @ dx.flatten())
    dx_filter = dx.flatten() - dx_proj

    nst, ndir, npts = 60, 3, 512

    ylim = 15
    scale = 0.05

    fig, axs, = utils.plot_adv_expl(x, dx_proj, lxp, scale=scale, ylim=ylim)
    fig.savefig("_plots/dx_noise_proj{:s}.pdf".format(suffix), dpi=200)

    fig, axs, = utils.plot_adv_expl(x, dx, lxp, scale=scale, ylim=ylim)
    fig.savefig("_plots/dx_noise_{:s}.pdf".format(suffix), dpi=200)

    fig, axs, = utils.plot_adv_expl(x, dx_filter, lxp, scale=scale, ylim=ylim)
    fig.savefig("_plots/dx_noise_filter{:s}.pdf".format(suffix), dpi=200)


    ## statistics over white noise

    nsamples = 100

    UX, UY = utils.load_basis(ens_index, test_index)
    phin, psin = utils.load_psi_phi(ens_index, test_index)

    torch.manual_seed(1)

    R_list = []
    R_proj_list = []
    R_filter_list = []

    scale = 0.08
    for i in range(nsamples):
        dx = scale*torch.randn(x_adv.shape)

        x_adv = utils.perturb(x, dx)
        y_adv = lxp.net(x_adv)

        R_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx = utils.toarray(dx)
        dx_proj = UX @ (UX.T @ dx.flatten())
        dx_filter = dx.flatten() - dx_proj

        x_adv = utils.perturb(x, dx_filter)
        y_adv = lxp.net(x_adv)

        R_filter_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx_proj = UX @ (UX.T @ dx.flatten())
        x_adv = utils.perturb(x, dx_proj)
        y_adv = lxp.net(x_adv)

        R_proj_list.append(utils.compute_R(x, x_adv, y, y_adv))

    print(np.mean(R_list), np.mean(R_proj_list), np.mean(R_filter_list))
    print(np.linalg.norm(dx.flatten())/ np.linalg.norm(x))

    R_white = (R_list, R_proj_list, R_filter_list)

    ## statistics over Brownian motion

    UX, UY = utils.load_basis(ens_index, test_index)
    x_adv, y_adv = utils.load_adv(ens_index, test_index)
    phin, psin = utils.load_psi_phi(ens_index, test_index)

    torch.manual_seed(1)

    R_list = []
    R_proj_list = []
    R_filter_list = []

    scale = 1.0
    for i in range(nsamples):
        dx = scale*torch.cumsum(torch.randn(x_adv.shape), axis=-1).detach().numpy().flatten()/200

        x_adv = utils.perturb(x, dx)
        y_adv = lxp.net(x_adv)

        R_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx_proj = UX @ (UX.T @ dx.flatten())
        dx_filter = dx.flatten() - dx_proj

        x_adv = utils.perturb(x, dx_filter)
        y_adv = lxp.net(x_adv)

        R_filter_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx_proj = UX @ (UX.T @ dx.flatten())
        x_adv = utils.perturb(x, dx_proj)
        y_adv = lxp.net(x_adv)

        R_proj_list.append(utils.compute_R(x, x_adv, y, y_adv))

    print(np.mean(R_list), np.mean(R_proj_list), np.mean(R_filter_list))
    print(np.linalg.norm(dx.flatten())/ np.linalg.norm(x))

    R_brownian = (R_list, R_proj_list, R_filter_list)

    ## statistics over power law correlated noise

    UX, UY = utils.load_basis(ens_index, test_index)
    phin, psin = utils.load_psi_phi(ens_index, test_index)

    torch.manual_seed(1)

    R_power_list = []
    R_power_proj_list = []
    R_power_filter_list = []

    #c0 = 1/2000
    c0 = 1/10
    #c0 = 1/400
    #c0 = 1/200
    scale = 25.0

    tt = np.arange(512//2)
    power = np.exp(-c0*tt**2)
    power = np.hstack((power, power[::-1]))

    for i in range(nsamples):
        dx =  np.fft.ifft(scale*np.random.randn(180, 512) * power).real

        x_adv = utils.perturb(x, dx)
        y_adv = lxp.net(x_adv)

        R_power_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx_proj = UX @ (UX.T @ dx.flatten())
        dx_filter = dx.flatten() - dx_proj
        x_adv = utils.perturb(x, dx_filter)
        y_adv = lxp.net(x_adv)

        R_power_filter_list.append(utils.compute_R(x, x_adv, y, y_adv))

        dx_proj = UX @ (UX.T @ dx.flatten())
        x_adv = utils.perturb(x, dx_proj)
        y_adv = lxp.net(x_adv)

        R_power_proj_list.append(utils.compute_R(x, x_adv, y, y_adv))

    print(np.mean(R_power_list), np.mean(R_power_proj_list), np.mean(R_power_filter_list))

    print(np.linalg.norm(dx.flatten())/ np.linalg.norm(x))

    R_power = (R_power_list, R_power_proj_list, R_power_filter_list)

    sc = 0.7
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=sc*np.array([12, 5]))

    fig.suptitle("$||\delta x ||_2/||x||_2 = 0.03$")

    ax = axs[0]
    ax.boxplot([R_white[0], R_white[2]])
    ax.set_xticklabels(["white", "filter"])
    ax.set_ylabel("$||\delta y ||_2/||\delta x||_2$")
    ax = axs[1]
    ax.boxplot([R_brownian[0], R_brownian[2]])
    ax.set_xticklabels(["brownian", "filter"])
    ax = axs[2]
    ax.boxplot([R_power[0], R_power[2]])
    ax.set_xticklabels(["power", "filter"])

    fig.savefig("_plots/noise_boxplot.pdf")

    fig.show()