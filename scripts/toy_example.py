
import torch
import torch.nn as nn
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import os

def mkdir(path):
    if not os.path.exists("_plots"): os.mkdir("_plots")

class ToyNN(nn.Module):
    def __init__(self, bias_level=0.0):

        super().__init__()

        self.dim = dim

        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)
        self.lin3 = nn.Linear(dim, dim, bias=False)
        self.lin4 = nn.Linear(dim, dim, bias=False)
        self.lin5 = nn.Linear(dim, dim, bias=False)
        self.lin6 = nn.Linear(dim, dim, bias=False)
        self.lin7 = nn.Linear(dim, dim, bias=False)

        self.lin_list = [
                          self.lin1,
                          self.lin2,
                          self.lin3,
                          self.lin4,
                          self.lin5,
                          self.lin6,
                          self.lin7,
                        ]

        self.t_lin1 = nn.Linear(dim, dim, bias=False)
        self.t_lin2 = nn.Linear(dim, dim, bias=False)
        self.t_lin3 = nn.Linear(dim, dim, bias=False)
        self.t_lin4 = nn.Linear(dim, dim, bias=False)
        self.t_lin5 = nn.Linear(dim, dim, bias=False)
        self.t_lin6 = nn.Linear(dim, dim, bias=False)
        self.t_lin7 = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            self.lin1.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin2.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin3.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin4.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin5.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin6.weight[...] = torch.randn(dim, dim) - bias_level
            self.lin7.weight[...] = torch.randn(dim, dim) - bias_level

            for i in range(dim):
                self.lin1.weight[i, (i+1):] = 0.0
                self.lin2.weight[i, (i+1):] = 0.0
                self.lin3.weight[i, (i+1):] = 0.0
                self.lin4.weight[i, (i+1):] = 0.0
                self.lin5.weight[i, (i+1):] = 0.0
                self.lin6.weight[i, (i+1):] = 0.0
                self.lin7.weight[i, (i+1):] = 0.0


        self.lin_list = [
                          self.lin1,
                          self.lin2,
                          self.lin3,
                          self.lin4,
                          self.lin5,
                          self.lin6,
                          self.lin7,
                        ]


        self.t_lin_list = [
                            self.t_lin1,
                            self.t_lin2,
                            self.t_lin3,
                            self.t_lin4,
                            self.t_lin5,
                            self.t_lin6,
                            self.t_lin7,
                          ]

        with torch.no_grad():
            self.t_lin1.weight[...] = self.lin1.weight.T
            self.t_lin2.weight[...] = self.lin2.weight.T
            self.t_lin3.weight[...] = self.lin3.weight.T
            self.t_lin4.weight[...] = self.lin4.weight.T
            self.t_lin5.weight[...] = self.lin5.weight.T
            self.t_lin6.weight[...] = self.lin6.weight.T
            self.t_lin7.weight[...] = self.lin7.weight.T

        self.relu = nn.ReLU()
        self.householder_list = []

        self.nrelus = len(self.lin_list) - 1


    def f0(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.lin6(x)
        x = self.lin7(x)
        return x

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.relu(x)
        x = self.lin5(x)
        x = self.relu(x)
        x = self.lin6(x)
        x = self.relu(x)
        x = self.lin7(x)
        return x

    def compute_fb(self, b):


        f0_wgt_list = [toarray(mat.weight) for mat in self.lin_list]
        self.f0_wgt_list = f0_wgt_list
        nrelus = net.nrelus
    
        b_str = np.binary_repr(b, width=nrelus)
        b_array = np.array([int(s) for s in b_str])

        mat = np.eye(dim)
        mat = f0_wgt_list[0].dot(mat)

        for i in range(nrelus):
            if b_array[i] == 1:
                v0 = toarray(self.householder_list[i]).reshape(-1, 1)
                v0_mat = v0.dot(v0.T)
                mat = v0_mat.dot(mat)
            mat = f0_wgt_list[i+1].dot(mat)

        return mat

    def compute_fb_mat(self):

        nrelus = self.nrelus
        dim = self.dim

        mat = np.zeros((dim, dim))
        for b in range(1, 2**nrelus):
            mat += self.compute_fb(b)

        return mat





    def reverse(self, y):

        nrelus = len(self.householder_list)
        self.reverse_list = []

        vl = self.householder_list[-1]
        vl0 = torch.zeros_like(vl)
        vl0[...] = vl

        vl0 = vl0.reshape(-1, len(vl0))
        t_lin_fctn = self.t_lin_list[-1]
        vl2 = t_lin_fctn(y.reshape(1, -1))
        vl0 = torch.cat((vl0, vl2), 0)

        for i in range(nrelus-2, -1, -1) :
            t_lin_fctn = self.t_lin_list[i]
            vl0 = t_lin_fctn(vl0)

            if i > -1:
                vl1 = self.householder_list[i]
                vl2 = torch.einsum("ji,i,k->jk",vl0, vl1, vl1)
                vl0 = torch.cat((vl0, vl2), 0)

        return vl0


    def relu_special(self, x_all):

        x0 = torch.sum(x_all, 1)
        y0 = x0 - torch.abs(x0)

        a = y0.norm(2)

        if a > 1e-8:
            y0 /= a
        else:
            y0 = y0*0.0

        x_all_new = - torch.einsum("l,j,jk->lk",y0, y0, x_all)
        x_all_out = torch.cat((x_all, x_all_new), 1)

        self.householder_list.append(y0)

        return x_all_out

    def lin_special(self, x, linear):

        x = x.permute(1, 0)    # use batch dimension
        x = linear(x)
        x = x.permute(1, 0)    # use batch dimension
        return x

    def forward_special(self, x):

        self.householder_list = []

        x = torch.unsqueeze(x, -1)
        x = self.lin_special(x, self.lin1)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin2)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin3)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin4)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin5)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin6)
        x = self.relu_special(x)
        x = self.lin_special(x, self.lin7)

        return x

    def compute_outmat(self, b):


        nrelus = len(self.householder_list)
        bn = np.binary_repr(b, width=nrelus)
        bn_array = np.array([int(i) for i in bn])

        mat = np.eye(self.dim, dtype=np.float32)

        lin_mat = toarray(self.lin_list[0].weight)
        mat = lin_mat.dot(mat)
        for i in range(len(bn_array)):

            if bn_array[i] == 1:
                v0_npy = toarray(self.householder_list[i]).reshape(-1, 1)
                relu_mat = v0_npy.dot(v0_npy.T)
                mat = relu_mat.dot(mat)

            lin_mat = toarray(self.lin_list[i+1].weight)
            mat = lin_mat.dot(mat)
        return mat

def toarray(x):
    return x.detach().numpy()


if __name__ == "__main__":

    # Comput singular value ratios for various hyperparameters
    ratio_all = []
    for dim in [10, 20, 30, 40, 50]:
        ratio_all_list = []
        for bias_level in np.linspace(0.0, 2.0, 11):
            ratio_list = []
            for i in range(100):

                net = ToyNN(bias_level=bias_level)
                net = net.eval()

                x = torch.rand(dim)
                out = net.forward_special(x)
                net_out = net.forward(x)

                f0_wgt_list = [toarray(mat.weight) for mat in net.lin_list]

                f0_mat = np.eye(dim)
                for mat in f0_wgt_list:
                    f0_mat = mat.dot(f0_mat)

                u0, s0, v0 = np.linalg.svd(f0_mat)

                fb_mat = net.compute_fb_mat()
                u1, s1, v1 = np.linalg.svd(fb_mat)

                ratio_list.append(s1[0]/s0[0])
                print(dim, bias_level, i)

            ratio_all_list.append(ratio_list)
        ratio_all.append(ratio_all_list)

    ratio_array = np.array(ratio_all)
    np.save("_output/singval_rat.npy", ratio_array)

    print("maximum ratio computed: ", ratio_array.max(axis=-1))


    # Compute Low-Rank Linear expansion for the toy model

    dim = 30
    torch.random.manual_seed(0)
    net = ToyNN()
    net = net.eval()
    # check
    x_test = torch.rand(dim) - 0.5
    x_test1 = torch.zeros(dim, 1)
    x_test1[:, 0] = x_test
    z_test1 = net.relu_special(x_test1)
    z_test = net.relu(x_test)
    print((z_test - torch.sum(z_test1, 1)).norm())

    linear = nn.Linear(dim, dim)
    x_test = torch.rand(2, dim)
    x_test1 = torch.zeros(dim, 2)
    x_test1[...] = x_test.permute(1, 0)
    z_test1 = net.lin_special(x_test1, linear)
    z_test = linear(x_test)
    print((z_test1 - z_test.permute(1,0)).norm())

    x = torch.rand(dim)
    out = net.forward_special(x)
    net_out = net.forward(x)

    f0_out = net.f0(x)
    print((out[:, 0] - f0_out).norm())
    print((net_out - torch.sum(out, 1)).norm())

    in_npy = toarray(net.reverse(net_out))
    out_npy = toarray(out)
    net_out_npy = toarray(net_out)
    _, s_phi, _ = np.linalg.svd(out_npy[:, 1:])
    _, s_psi, _ = np.linalg.svd(in_npy.T[:, 1:])

    import matplotlib
    matplotlib.rcParams["font.family"] = "monospace"
    import matplotlib.pyplot as plt
    import utils

    fig, ax = utils.plot_singvals_toy(s_phi, s_psi)
    mkdir("_plots")
    fig.savefig("_plots/singvals_toy.pdf")

    f0_wgt_list = [toarray(mat.weight) for mat in net.lin_list]

    f0_mat = np.eye(dim)
    for mat in f0_wgt_list:
        f0_mat = mat.dot(f0_mat)

    u0, s0, v0 = np.linalg.svd(f0_mat)

    fb_mat = net.compute_fb_mat()
    u1, s1, v1 = np.linalg.svd(fb_mat)


    fig, ax = utils.plot_singvals_toy_f0fb(s0, s1)
    fig.savefig("_plots/singvals_toy_f0fs.pdf")
    fig.show()

    import utils

    mkdir("_output")
    ratio_array = np.load("_output/singval_rat.npy")
    fig, axs = utils.plot_ratio(ratio_array)
    fig.savefig("_plots/boxplot_ratios.pdf")
