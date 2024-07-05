
import cnn_lxp
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "monospace"
import matplotlib.pyplot as plt
import os, sys 

station_all_list = [
                'P316', 'albh', 'bamf', 'bend', 'bils', 'cabl', 'chzz', 'cski',
                'ddsn', 'eliz', 'elsr', 'grmd', 'holb', 'lsig', 'lwck', 'mkah',
                'neah', 'nint', 'ntka', 'ocen', 'onab', 'p154', 'p156', 'p157',
                'p160', 'p162', 'p329', 'p343', 'p362', 'p364', 'p365', 'p366',
                'p380', 'p387', 'p395', 'p396', 'p397', 'p398', 'p401', 'p403',
                'p407', 'p441', 'p733', 'p734', 'pabh', 'ptrf', 'ptsg', 'reed',
                'sc02', 'sc03', 'seas', 'seat', 'tfno', 'thun', 'till', 'trnd',
                'uclu', 'ufda', 'wdcb', 'ybhb',
                ]

station_part_list = [
                "bamf",
                "lsig",
                ]

station_name_list = []
station_bool_array = np.zeros(len(station_all_list), dtype=bool)
for i, station in enumerate(station_all_list):
    if station in station_part_list:
        station_name_list.append(station)
        station_bool_array[i] = True


def plot_output(y, ax, ptype="adv", max_amp=None):

    if type(y) == torch.Tensor:
        y = y.detach().numpy()

    y = reshape_output(y)

    t = np.linspace(0, 6, 256)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel("$time$ (hrs)")
    ax.set_ylabel("surf elev (m)")

    gauge_no = 901
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    i = 1
    i1 = 4
    if type(max_amp) == type(None):
        max_amp = np.abs(y[i, :]).max()
    if ptype=="adv":
        ax.plot(t, y[i, :], color=color_list[i1], label="pred gauge {:d}".format(gauge_no))
    elif ptype=="basis":
        ax.plot(t, y[i, :], color=color_list[i1], label="gauge {:d}".format(gauge_no))
    elif ptype=="test":
        ax.plot(t, y[i, :], "k--", label="orig pred")
    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.set_ylim(np.array([-1.01, 1.01])*max_amp)
    ax.set_xlim([0, 6])
    ax.grid("on")


def toarray(x):
    if type(x) == np.ndarray:
        return x
    else:
        return x.detach().numpy()

def reshape_input(x):

    nst = 60
    ndir = 3
    npts = 512
    new_shape = (nst, ndir, npts)
    return np.reshape(x, new_shape)

def reshape_input1(x):

    nst = 60
    ndir = 3
    npts = 512
    new_shape = (1, nst*ndir, npts)
    return np.reshape(x, new_shape)

def reshape_output(y):

    ngauges = 3
    npts = 256
    new_shape = (ngauges, npts)

    return np.reshape(y, new_shape)

def plot_input(x, axs, max_amp=None, ptype="adv"):

    if type(x) == torch.Tensor:
        x = x.detach().numpy()
    x = reshape_input(x)

    t = np.arange(512) / 60
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(max_amp) == type(None):
        max_amp = np.abs(x[station_bool_array, :]).max()

    x_part = x[station_bool_array, -1, :].T
    for i in range(len(axs)):
        if ptype=="adv":
            axs[i].plot(t, x_part[:, i], label=station_part_list[i], color=color_list[i])
        elif ptype=="test":
            axs[i].plot(t, x_part[:, i], label="orig input", color="k", linestyle="dashed")
        elif ptype=="diff":
            axs[i].plot(t, x_part[:, i], label="$\delta x$ " + station_part_list[i], color=color_list[i])
        axs[i].set_ylim(np.array([-1.05, 1.05])*max_amp)
        axs[i].grid("on")
        axs[i].set_ylabel("disp Z (m)")
        axs[i].tick_params(axis='y', which='major', labelsize=8)
        axs[i].set_xlim([0, 8])
    axs[0].sharex(axs[1])
    axs[1].set_xlabel("$time$ (mins)")



def prepare_adv(dx, x, sc=1.0, ord=2):

    if not type(dx) == torch.Tensor:
        dx = torch.tensor(dx, dtype=torch.float32)
    if not type(x) == torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32)

    x = reshape_input1(x)
    dx = reshape_input1(dx)

    if ord == 2:
        dx_l2norm = torch.sqrt(torch.sum(torch.abs(dx)**2))

        dx = dx / dx_l2norm
        sc = sc * torch.sqrt(torch.sum(torch.abs(x)**2)) 
    elif ord == np.inf:
        dx = dx / torch.abs(dx).max()
    else:
        raise ValueError

    x_adv = x + sc*dx

    return x_adv, sc*dx

def load_psi_phi(ens_index, test_index):

    suffix = "_n{:02d}_i{:04d}".format(ens_index, test_index)

    fname1 = "_output/phin{:s}.npy".format(suffix)
    fname2 = "_output/psin{:s}.npy".format(suffix)

    return np.load(fname1), np.load(fname2)

def load_basis(ens_index, test_index):

    UX = np.load("_output/UX_n{:02d}_i{:04d}.npy".format(ens_index, test_index))
    UY = np.load("_output/UY_n{:02d}_i{:04d}.npy".format(ens_index, test_index))

    return UX, UY

def load_inout_npy(test_index, lxp, return_npy=True):

    test_input = lxp.Model._get_data1_tensor(test_index, inout='in')
    test_output = lxp.Model._get_data1_tensor(test_index, inout='out')

    if return_npy:
        return test_input.detach().numpy(), test_output.detach().numpy()
    else:
        return test_input, test_output

def load_adv(ens_index, test_index, proj_type="l2"):

    x_adv = np.load("_output/x_adv_{:s}_n{:02d}_i{:04d}.npy".format(proj_type, ens_index, test_index))
    y_adv = np.load("_output/y_adv_{:s}_n{:02d}_i{:04d}.npy".format(proj_type, ens_index, test_index))

    return x_adv, y_adv


def plot_adv_expl(x, dx, lxp, scale=0.05, ylim=3.0):

    x_adv_tensor1, dx_scaled_tensor = prepare_adv(dx, x, sc=scale)
    dx_scaled_array = toarray(dx_scaled_tensor)

    dx_l2norm = np.linalg.norm(dx_scaled_array.flatten()) / np.sqrt(60*3*512)
    dx_linfnorm = np.linalg.norm(dx_scaled_array.flatten(), ord=np.inf)
    print("l2norm = {:1.4f}, linf = {:1.4f}".format(dx_l2norm, dx_linfnorm))

    x = reshape_input1(x)

    x = torch.tensor(x, dtype=torch.float32)

    y = lxp.net(x)
    y_adv_tensor1 = lxp.net(x_adv_tensor1)

    fig1, axs1 = create_plot6()

    axs_input_base = np.array([axs1["A"], axs1["B"]])
    axs_input_diff = np.array([axs1["C"], axs1["D"]])
    ax_output = axs1["H"]

    ax_output.set_ylim([-ylim, ylim])

    # adv input plot
    plot_input(x_adv_tensor1, axs_input_base, max_amp=5)
    plot_input(x, axs_input_base, ptype="test", max_amp=5)
    for ax in axs_input_base:
        legend = ax.legend(loc="upper right", fontsize=9)
        legend.get_frame().set_alpha(None)
    #axs_input_base[-1].set_yticks([])
    #axs_input_base[-1].set_yticklabels([])

    # adv output plot
    plot_output(y_adv_tensor1, ax_output)
    plot_output(y, ax_output, ptype="test", max_amp=ylim)

    plot_input(dx_scaled_tensor, axs_input_diff, ptype="diff")
    for ax in axs_input_diff:
        legend = ax.legend(loc="lower right", fontsize=9)
        legend.get_frame().set_alpha(None)
    y = toarray(y)
    y_adv_array1 = toarray(y_adv_tensor1)
    dy = y_adv_array1 - y

    axs_input_base[0].set_title("(a)", loc="left")
    axs_input_diff[0].set_title("(b)", loc="left")
    ax_output.set_title("(c)", loc="left")

    input_rel_l2norm = np.linalg.norm(dx_scaled_array.flatten()) \
                     / np.linalg.norm(toarray(x).flatten())

    output_rel_l2norm = np.linalg.norm(dy.flatten()) \
                      / np.linalg.norm(y.flatten())

    axs_input_diff[0].set_title(
        r"$|| \delta x ||_2 / ||x||_2$ = {:1.3f}".format(input_rel_l2norm), loc="right", fontsize=9)
    ax_output.set_title(r"$|| \delta y ||_2 / ||y||_2$ = {:1.2f}".format(output_rel_l2norm), loc="right", fontsize=9)
    ax_output.legend(loc="upper right", fontsize=8)
    fig1.show()

    return fig1, axs1


def plot_adv_expl_output(x, dx, lxp, scale=0.05, ylim=3.0):

    x_adv_tensor1, dx_scaled_tensor = prepare_adv(dx, x, sc=scale)
    dx_scaled_array = toarray(dx_scaled_tensor)

    x = reshape_input1(x)
    x = torch.tensor(x, dtype=torch.float32)

    y = lxp.net(x)
    y_adv_tensor1 = lxp.net(x_adv_tensor1)
    fig, ax = create_plot7()

    # adv output
    plot_output(y_adv_tensor1, ax)
    plot_output(y, ax, ptype="test")

    y = toarray(y)
    y_adv_array1 = toarray(y_adv_tensor1)
    dy = y_adv_array1 - y

    input_rel_l2norm = np.linalg.norm(dx_scaled_array.flatten()) \
                     / np.linalg.norm(toarray(x).flatten())

    output_rel_l2norm = np.linalg.norm(dy.flatten()) \
                      / np.linalg.norm(y.flatten())

    ax.set_ylim([-ylim, ylim])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(r"$|| \delta y ||_2 / ||y||_2$ = {:1.2f}".format(output_rel_l2norm), fontsize=9, loc="right")

    fig.tight_layout()
    fig.show()
    return fig, ax



def create_plot1(sc = 1.3):
    return plt.subplots(ncols = 3,
                        nrows = 2,
                        figsize = sc*np.array([4, 8]),
                        gridspec_kw = {"height_ratios": (5, 1)})

def create_plot2(sc = 0.65):
    return plt.subplots(figsize = sc*np.array([9, 4]))

def create_plot3(sc = 0.65):
    return plt.subplots(ncols = 2, figsize = sc*np.array([10, 4]))

def create_plot4(sc = 0.65):
    return plt.subplots(ncols = 3, nrows = 2, figsize = sc*np.array([10, 6]))

def create_plot5(sc = 0.8):
    return plt.subplot_mosaic(mosaic = """
                                       ...
                                       AAA
                                       BBB
                                       ...
                                       CCC
                                       """,
                              figsize = sc*np.array([4, 7.0]),
                              gridspec_kw = {"height_ratios": (0.05, 2.0, 2.0, 0.05, 2.5,)})

def create_plot6(sc = 1.0):
    return plt.subplot_mosaic(mosaic = """
                                       A.C
                                       ...
                                       B.D
                                       ...
                                       HHH
                                       """,
                              figsize = sc*np.array([7, 6]),
                              gridspec_kw = {"height_ratios": 
                                             (2.0, 0.1, 2.0, 0.9, 3),
                            "width_ratios": (1, 0.1, 1)})

def create_plot7(sc = 0.65):
    return plt.subplots(figsize = sc*np.array([7, 4]))


def get_suffix(ens_index, test_index):
    return "_n{:02d}_i{:04d}".format(ens_index, test_index)


def plot_singvals(s_phi, s_psi):

    mach_eps = 1.1920e-7

    fig, ax = create_plot2()
    ax.semilogy(np.arange(len(s_psi)) + 1, s_psi / s_psi[0], marker=".", label="$\psi$-space")
    ax.semilogy(np.arange(len(s_phi)) + 1, s_phi / s_phi[0], marker="x", label="$\phi$-space")
    ax.grid("on")
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    ax.vlines([17,], [1e-17,], [10,], linestyle="dotted", color="k", label="n relus")
    ax.hlines([mach_eps,], [-0.5,], [len(s_phi),], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([0.5, 30])
    ax.set_ylim([mach_eps*1e-3, 10])
    ax.legend()
    fig.tight_layout()
    fig.show()

    return fig, ax

def plot_singvals_toy(s_phi, s_psi):
    mach_eps = 1.1920e-7

    fig, ax = create_plot2()
    ax.semilogy(np.arange(len(s_psi)) + 1, s_psi / s_psi[0], marker=".", label="$\psi$-space")
    ax.semilogy(np.arange(len(s_phi)) + 1, s_phi / s_phi[0], marker="x", label="$\phi$-space")
    ax.grid("on")
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    ax.vlines([6,], [1e-17,], [10,], linestyle="dotted", color="k", label="n relus")
    ax.hlines([mach_eps,], [-0.5,], [len(s_phi),], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([0.5, len(s_phi)//2])
    ax.set_ylim([mach_eps*1e-3, 10])
    ax.legend()
    fig.tight_layout()
    fig.show()
    return fig, ax

def plot_corr(corr):

    mach_eps = 1.1920e-7

    from matplotlib import colors

    u, s, v = np.linalg.svd(corr, full_matrices=False)

    fig, axs = create_plot3()

    ax = axs[0]
    im = ax.imshow(np.abs(corr).T, norm=colors.LogNorm(), extent = (1, corr.shape[0], corr.shape[1], 1))
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("$\ell'$")
    ax.set_ylabel("$\ell$")

    ax = axs[1]
    ax.semilogy(np.arange(len(s)) + 1, s/s[0], marker=".")
    ax.grid("on")
    ax.hlines([mach_eps,], [-0.5,], [len(s)+1,], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([0.5, 17.5])
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    ax.legend()

    fig.tight_layout()
    fig.show()

    return fig, axs


def plot_psi_phi(psi, phi):

    ylim = max([np.abs(psi).max(), np.abs(phi).max()])*1.05

    fig1, axs1 = create_plot5()

    # adv input plot
    axs_input = np.array([axs1["A"], axs1["B"]])
    plot_input(psi, axs_input, max_amp = 0.04)
 
    for ax in axs_input:
        legend = ax.legend(loc="upper right", fontsize=9)
        legend.get_frame().set_alpha(None)

    ax_output = axs1["C"]
    # adv output
    plot_output(phi, ax_output, ptype="basis", max_amp = 0.2)

    legend = ax_output.legend(loc="lower right", fontsize=9, fancybox=False)
    legend.get_frame().set_alpha(None)

    fig1.tight_layout()
    fig1.show()
    return fig1, axs1


def plot_singvals1(s_phi):

    mach_eps = 1.1920e-7

    fig, ax = create_plot2()
    ax.semilogy(s_phi / s_phi[0], marker="s", color="b", label="$F_0$ singular values")
    ax.grid("on")
    #ax.semilogy(s_psi / s_psi[0], marker=".", label="$\phi$-space")
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    #ax.vlines([16,], [1e-17,], [10,], linestyle="dotted", color="k", label="n relus")
    ax.hlines([mach_eps,], [-0.5,], [len(s_phi),], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([-0.5, 31.5])
    ax.set_ylim([mach_eps*1e-3, 10])
    ax.legend()
    fig.tight_layout()
    fig.show()

    return fig, ax

def plot_singvals2(s_phi):

    mach_eps = 1.1920e-7

    fig, ax = create_plot2()
    ax.semilogy(s_phi / s_phi[0], marker=">", color="g", label="$F_{\sigma}(x_0)$ singular values")
    ax.grid("on")
    #ax.semilogy(s_psi / s_psi[0], marker=".", label="$\phi$-space")
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    #ax.vlines([16,], [1e-17,], [10,], linestyle="dotted", color="k", label="n relus")
    ax.hlines([mach_eps,], [-0.5,], [len(s_phi),], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([-0.5, 31.5])
    ax.set_ylim([mach_eps*1e-3, 10])
    ax.legend()
    fig.tight_layout()
    fig.show()

    return fig, ax

def pdfcrop():

    fname_list = os.listdir("_plots")

    cmd = "cd _plots"
    for fname in fname_list:
        if "pdf" in fname[-4:] and "crop" not in fname:
            cmd +=  " && pdfcrop " + fname
    os.system(cmd)


def plot_singvals_toy_f0fb(s0, s1):
    mach_eps = 1.1920e-7

    fig, ax = create_plot2(sc = 0.65)
    ax.semilogy(np.arange(len(s0)) + 1, s0 / s0[0], marker="s", label="sing vals $F_0$")
    ax.semilogy(np.arange(len(s1)) + 1, s1 / s1[0], marker=".", label="sing vals $F_{\sigma}(x_0)$")
    ax.grid("on")
    ax.set_ylabel("$\sigma_i / \sigma_1$ ")
    ax.set_xlabel("index $i$")
    ax.vlines([6,], [1e-17,], [10,], linestyle="dotted", color="k", label="n relus")
    ax.hlines([mach_eps,], [-0.5,], [len(s0),], linestyle="dashed", color="k", label="mach eps")
    ax.set_xlim([0.5, len(s0)//2])
    ax.set_ylim([mach_eps*1e-3, 10])
    leg = ax.legend(fancybox=False)
    leg.get_frame().set_alpha(None)
    fig.tight_layout()
    fig.show()
    return fig, ax


def compute_R(x, x_adv, y, y_adv):

    dx = toarray(x_adv).flatten() - toarray(x).flatten()
    dy = toarray(y_adv).flatten() - toarray(y).flatten()

    dx_rel = np.linalg.norm(dx) / np.linalg.norm(toarray(x).flatten())
    dy_rel = np.linalg.norm(dy) / np.linalg.norm(toarray(y).flatten())

    return dy_rel / dx_rel


def perturb(x, dx):

    x_adv = torch.tensor(x, dtype=torch.float32).reshape(1, 180, 512) \
          + torch.tensor(dx, dtype=torch.float32).reshape(1, 180, 512)

    return x_adv.clone().detach()


def plot_ratio(ratio_array, sc = 0.5):

    fig, axs = plt.subplots(ncols=2, figsize=sc*np.array([12, 5]), 
                            gridspec_kw={"width_ratios": (6, 4)},
                            sharey=True)

    ax = axs[0]
    ax.boxplot(ratio_array[3, :, :].T)

    xticklabels_ = np.linspace(0.0, 2.0, 11)
    ax.set_xticklabels(["{:1.1f}".format(i) for i in xticklabels_])
    ax.set_ylabel(r"$|| F_{\sigma} (x_0) ||_2 / || F_0 ||_2$")
    ax.set_xlabel("bias level (dim 30)")

    ax = axs[1]
    ax.boxplot(ratio_array[:, 4, :].T)
    xticklabels_ = [10, 20, 30, 40, 50]
    ax.set_xticklabels(["{:d}".format(i) for i in xticklabels_])
    ax.set_xlabel("dim (bias lvl 0.6)")
    fig.tight_layout()

    fig.show()

    return fig, axs


def _l2_projection(x_base, epsilon, x_adv):
    delta = x_adv - x_base

    # consider the batch run
    mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= epsilon

    # compute the scaling factor
    scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
    scaling_factor[mask] = epsilon

    # scale delta based on the factor
    delta *= epsilon / scaling_factor.view(-1, 1, 1)
    return (x_base + delta)


def generate_example(x, y, model, loss_func, T, eps, alpha, proj_type="l2"):
    """

    Parameters
    ----------
        x: input
        y: model output
        model: convolutional neural network
        loss_func: loss function
        T: number of iterations
        eps: constraint for perturbation size
        alpha: step size
        proj_type: {"l2", "linf"} specify norm to measure perturbation size
    
    Adapted from code from Sanghyun Hong, see repository:
    https://github.com/Sanghyun-Hong/DeepSloth/

    MIT License
    
    Copyright (c) 2020 Sanghyun Hong (Albert)
    
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """

    model.eval()

    # perturbation
    p = torch.zeros(1, 180, 512)

    # random perturbation
    x_adv = x + eps*torch.rand((1, 180, 512))
    x_adv.requires_grad = True

    for i in range(T):
        sys.stdout.write("\r PGD iteration {:4d}".format(i))

        # evaluating pertrubed x
        output = model(x_adv)

        # loss between true value and perturbed evaluation
        loss = loss_func(output, y)

        # gradient
        grad = torch.autograd.grad(loss, x_adv)[0]

        # updating perturbation
        if proj_type=="l2":
            x_adv = x_adv + alpha*grad
            x_adv = _l2_projection(x, eps, x_adv)
        elif proj_type=="linf":
            p = torch.clamp(p + alpha*grad.sign(), -eps, eps)
            x_adv = x_adv + p
        else:
            raise NotImplementedError

    return x_adv

def pgd_adv(proj_type="l2", i_ens=1, i_testno=0, seed=0):

    torch.random.manual_seed(seed)

    lxp = cnn_lxp.CNNLinearExpansion(i_ens = i_ens)
    lxp.load_cnn()

    x = lxp.Model._get_data1_tensor(i_testno, inout="in")
    y = lxp.Model._get_data1_tensor(i_testno, inout="out")
    y = torch.unsqueeze(y, dim=0)

    model = lxp.net.eval()
    loss_func = nn.L1Loss()

    T = 1000
    if proj_type == "linf":
        alpha = 1.0
        eps = 0.5
    elif proj_type == "l2":
        alpha = 1.0
        eps = float(0.01*torch.norm(x.flatten(), p=2))
    else:
        raise NotImplementedError
    
    x_adv = generate_example(x, y, model, loss_func, T, eps, alpha, proj_type=proj_type)

    y_adv = model(x_adv)

    x_adv_array = toarray(x_adv)
    y_adv_array = toarray(y_adv)
    y_array = toarray(y)

    suffix = "_{:s}_n{:02d}_i{:04d}".format(proj_type, i_ens, i_testno)

    fname = "_output/x_adv{:s}.npy".format(suffix)
    np.save(fname, x_adv_array)

    fname = "_output/y_adv{:s}.npy".format(suffix)
    np.save(fname, y_adv_array)


def lxp_analysis(i_ens=1, i_testno=0):

    scale = 0.005

    index_nos = "_n{:02d}_i{:04d}".format(i_ens, i_testno)

    lxp = cnn_lxp.CNNLinearExpansion(i_ens = i_ens)
    lxp.load_cnn()

    x, y = load_inout_npy(i_testno, lxp)

    x1 = reshape_input1(x)
    x1 = torch.tensor(x1, dtype=torch.float32)
    y1 = lxp.net(x1)
    y1 = toarray(y1)
    ylim = np.abs(y1.flatten()).max()*1.1

    lxp.compute_basis(i_testno)

    UX = lxp.basis[0]

    n = UX.shape[1]

    alphabet_str = "abcdefghijklmnopqrstuvwxyz"

    for p in range(n):
        fig, axs = plot_adv_expl_output(x, -UX[:, p], lxp, scale=scale, ylim=ylim*1.25)
        fig.suptitle("({:s})".format(alphabet_str[p]), x=0.05)
        fig.savefig("_plots/dpsi{:s}_{:02d}.pdf".format(index_nos, p),
                    bbox_inches="tight")
        plt.close(fig)

    x_adv, y_adv = load_adv(i_ens, i_testno)
    dx = x_adv - x
    dx_proj = UX @ (UX.T @ dx.flatten())
    dx_filter = dx.flatten() - dx_proj

    for label, diff in \
        [("dx", dx), ("dx_proj", dx_proj), ("dx_filter", dx_filter)]:
        fig, axs  = \
            plot_adv_expl(x, diff, lxp, scale=scale, ylim=ylim*1.25)
        fig.savefig("_plots/{:s}{:s}.pdf".format(label, index_nos), 
                    bbox_inches="tight")
        plt.close(fig)

    # clean up
    del lxp 

