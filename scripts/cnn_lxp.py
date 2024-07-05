r"""
Class containing methods for Low Rank Expansion that linearizes a CNN about an
input
"""

import torch
import torch.nn as nn
import numpy as np
import os, sys
import copy


class CNNLinearExpansion():

    def __init__(self, i_ens=1):
        self.download = False
        self.i_ens = i_ens

    
    def _tensor_to_array(self, tensor):
        r"""
        Helper function that converts input torch tensor to a numpy array

        """
        return tensor.detach().numpy()


    def _array_to_tensor(self, array, dtype = torch.float32):
        r"""
        Helper function that converts input numpy array to a torch tensor 
        
        """
        return torch.tensor(array, dtype=dtype)


    def load_cnn(self):
        r"""
        Load one neural network in the ensemble.

        """
    
        import cnn_gnss

        Model = cnn_gnss.GNSS_gauge_model()
        Model.load_data()
        Model.load_model('sjdf')

        self.Model = Model

        epoch_list = Model.stop_epoch
        #nensemble = Model.nensemble

        i_ens = self.i_ens
        net = Model.get_eval_model(i_ens, epoch_list[i_ens])

        self.net = net

        # a list of modules in the NN
        self.eval_sequence = [('conv1',  net.conv1),
                              ('relu',   net.relu),       # 1
                              ('pool',   net.pool),
                              ('conv2',  net.conv2),
                              ('relu',   net.relu),       # 2
                              ('pool',   net.pool),
                              ('conv3',  net.conv3),
                              ('relu',   net.relu),       # 3
                              ('pool',   net.pool),
                              ('conv4',  net.conv4),
                              ('relu',   net.relu),       # 4
                              ('pool',   net.pool),
                              ('conv5',  net.conv5),
                              ('relu',   net.relu),       # 5
                              ('pool',   net.pool),
                              ('conv6',  net.conv6),
                              ('relu',   net.relu),       # 6
                              ('pool',   net.pool),
                              ('conv7',  net.conv7),
                              ('relu',   net.relu),       # 7
                              ('pool',   net.pool),
                              ('conv8',  net.conv8),
                              ('relu',   net.relu),       # 8
                              ('pool',   net.pool),
                              ('conv9',  net.conv9),
                              ('relu',   net.relu),       # 9
                              ('pool',   net.pool),
                              ('t_conv1',net.t_conv1),
                              ('relu',   net.relu),       # 10
                              ('t_conv2',net.t_conv2),
                              ('relu',   net.relu),       # 11
                              ('t_conv3',net.t_conv3),
                              ('relu',   net.relu),       # 12
                              ('t_conv4',net.t_conv4),
                              ('relu',   net.relu),       # 13
                              ('t_conv5',net.t_conv5),
                              ('relu',   net.relu),       # 14
                              ('t_conv6',net.t_conv6),
                              ('relu',   net.relu),       # 15
                              ('t_conv7',net.t_conv7),
                              ('relu',   net.relu),       # 16
                              ('t_conv8',net.t_conv8),
                              ('relu',   net.relu),       # 17
                             ]

        self.nrelus = 17

        relu_pos = []
        for i, nn_module in enumerate(self.eval_sequence):

            if nn_module[0] == "relu":
                relu_pos.append(i)

        self.relu_pos = np.array(relu_pos)


    def _print_msg(self, msg):
        sys.stdout.write("\r {:s} ".format(msg))


    def _get_ref_vector(self, x):
        r"""
        Get Householder reflector
        
        """
        v = (torch.abs(x) - x).flatten()
        if torch.dot(v, v) > 1e-14:
            v /= torch.sqrt(torch.dot(v, v))
        else:
            v *= 0.0
        v *= np.sqrt(2)
        return v


    def _lrelu_split(self, z, beta=-0.5):
        r"""
        Convert leaky ReLU into a sum of two matrices
        
        """

        get_ref_vector = self._get_ref_vector
        
        #z1 = 0.5*(1 - beta)*z
        z1 = z

        v = get_ref_vector(z)
        v = np.sqrt(0.5*(1 + beta))*v

        # Householder reflector
        z2 = - v*(v.dot(z1.flatten()))
        z2 = z2.reshape(z1.shape)

        return z1, z2, v


    def _pool(self, x):
        tensor_to_array = self._tensor_to_array
        array_to_tensor = self._array_to_tensor

        x1 = tensor_to_array(x)
        old_shape = x1.shape
        new_shape = old_shape[:-1] + (old_shape[-1]//2, 2)

        x2 = x1.reshape(new_shape)
        index_array = np.argmax(x2, axis=-1).reshape(-1, 2)

        m = np.prod(index_array.shape)
        out = np.zeros(m)

        index_array = index_array.flatten()

        x2 = x2.reshape(-1, 2)
        for i in range(m):
            out[i] = x2[i, index_array[i]]

        out = out.reshape(new_shape[:-1])
        out = array_to_tensor(out)

        return out, index_array


    def compute_lxp(self, x):
        r"""

        Parameters
        ----------
            x : torch tensor (?, 3*60, 512)
        
        """

        print_msg = self._print_msg
        tensor_to_array = self._tensor_to_array
        array_to_tensor = self._array_to_tensor
        lrelu_split = self._lrelu_split
        pool = self._pool

        eval_sequence = self.eval_sequence

        self.sequence = []

        for nn_module in eval_sequence:

            func_name, func = nn_module
            print_msg(func_name)

            # nonlinear activation
            if func_name == 'relu':
                z1, z2, v = lrelu_split(x)

                x = z1 + z2
                self.sequence.append((func_name, v))
            
            # compute index for max pool layer
            elif func_name == 'pool':
                x, index_array = pool(x)
                self.sequence.append((func_name, index_array))

            else:
                in_shape = x.shape
                x = func(x)
                out_shape = x.shape
                self.sequence.append(
                    (func_name, func.weight, func.bias, func.in_channels, func.out_channels, func.kernel_size[0])
                    )


        return x


    def _btconv(self, x, weight, bias, in_channels, out_channels, kernel_size, padding=1):
        """
        Compute transpose convolution layer using provided weights
        """

        x = x.reshape(1, out_channels, -1)

        tconv = nn.ConvTranspose1d(out_channels,
                                   in_channels,
                                   kernel_size,
                                   padding=padding)
        tconv = tconv.eval()
        
        tconv.weight.data[:] = weight
        tconv.bias.data[:] = 0.0

        out = tconv(x).flatten()

        out_bias = torch.einsum("ijk,j->ik", x, bias)
        out_bias = out_bias.flatten().sum()

        return out, out_bias


    def _bconv(self, x, weight, bias, in_channels, out_channels, kernel_size, padding=1):
        """
        Compute convolution layer using provided weights
        """

        x = x.reshape(1, out_channels, -1)

        conv = nn.Conv1d(out_channels,
                         in_channels,
                         kernel_size,
                         stride=kernel_size)

        conv = conv.eval()

        conv.weight.data[:] = weight
        conv.bias.data[:] = 0.0

        out = conv(x).flatten()

        out_bias = torch.einsum("ijk,j->ik", x, bias)
        out_bias = out_bias.flatten().sum()

        return out, out_bias


    def _btpool(self, x, index_array):
        """
        Compute tranpose of a frozen max pooling layer
        """
        tensor_to_array = self._tensor_to_array
        array_to_tensor = self._array_to_tensor

        x1 = tensor_to_array(x)

        m = len(index_array)

        old_shape = x1.shape
        new_shape = old_shape + (2,)

        out = np.zeros(new_shape)

        for i in range(m):
            out[i, index_array[i]] = x1[i]

        out = out.flatten()
        out = array_to_tensor(out)

        return out

    def _vprint(self, *msg):

        if self.verbose == True:
            print(msg)


    def lxp1(self, b, verbose=False):
        """
        Compute a single term in the expanded NN for given binary number b

        """

        self.verbose = verbose

        vprint = self._vprint

        nrelus = self.nrelus
        eval_sequence = self.eval_sequence
        n = len(eval_sequence)
        sequence = self.sequence
        relu_pos = self.relu_pos

        b_str = np.binary_repr(b, width=nrelus)
        b_array = np.array([int(bit) for bit in b_str])
        vprint(b_array)
        k0 = relu_pos[np.argmax(b_array[::-1])]
        k1 = relu_pos[nrelus - 1 - np.argmax(b_array)]

        nn_module = sequence[k0]     # sequence[k] should always be a relu
        func_name = nn_module[0]
        v0 = nn_module[1]

        nn_module = sequence[k1]     # sequence[k] should always be a relu
        func_name = nn_module[0]
        v1 = nn_module[1]

        self.split = [(k0, v0, b_array),(k1, v1, b_array)]

        btconv = self._btconv
        bconv = self._bconv
        btpool = self._btpool

        x = copy.copy(v0)
        y = copy.copy(v1)

        if b > 0:

            self.bias_list = [[], []]
            bias_list = self.bias_list[0]

            for i in range(k0-1, -1, -1):
                nn_module = sequence[i]  
                func_name = nn_module[0]

                if func_name == "relu":
                    #func_name, v = nn_module
                    if b_array[relu_pos[::-1] == i]:
                        v_new = copy.copy(nn_module[1])
                        x = -x.dot(v_new)*v_new
                        print(b_array, i, relu_pos)
                    else:
                        pass
                elif func_name[:-1] == "conv":
                    func_name, wgts, bias, \
                        in_channels, out_channels, kernel_size = nn_module
                    x, out_bias = btconv(x, wgts, bias,
                                      in_channels, out_channels, kernel_size)
                    bias_list.append((x, out_bias))
                elif func_name[:-1] == "t_conv":
                    func_name, wgts, bias, \
                        in_channels, out_channels, kernel_size = nn_module
                    x, out_bias = bconv(x, wgts, bias,
                                       in_channels, out_channels, kernel_size)
                    bias_list.append((x, out_bias))
                elif func_name == "pool":
                    index_array = nn_module[1]
                    x = btpool(x, index_array)

                    vprint("m = ", len(index_array))

                vprint("-- after", func_name, x.shape)

            bias_list = self.bias_list[1]

            ## add the forward evaluated tensor
            for i in range(k1+1, n):
                nn_module = sequence[i]  
                func_name = nn_module[0]
                eval_func_name, eval_nn_module = eval_sequence[i]
                vprint(func_name, eval_func_name, eval_nn_module)

                if func_name == "relu":
                    if b_array[relu_pos[::-1] == i]:
                        v_new = copy.copy(nn_module[1])
                        y = - y.dot(v_new)*v_new
                    else:
                        pass

                elif (func_name[:-1] == "conv") or (func_name[:-1] == "t_conv"):
                    func_name, wgts, bias, \
                        in_channel, out_channel, kernel_size = nn_module

                    y = y.reshape(1, in_channel, -1)
                    #eval_nn_module.bias = nn.Parameter(torch.zeros_like(eval_nn_module.bias.data))
                    y = eval_nn_module(y)
                    y = y.flatten()

                elif func_name == "pool":
                    index_array = nn_module[1]
                    y = y.reshape(-1, 2)
                    m = y.shape[0]
                    z = torch.zeros(m)
                    if m != len(index_array):
                        print("warning: dimension mismatch")
                    for j in range(m):
                        z[j] = y[j, index_array[j]]
                    y = copy.copy(z)

            self.lr_vectors = (copy.copy(x), copy.copy(y))


    def compute_basis(self, i_test):
        """
        Compute input and output basis psi and phi
        """

        test_input = self.Model._get_data1_tensor(i_test, inout='in')
        test_output = self.Model._get_data1_tensor(i_test, inout='out')        

        test_input1 = torch.unsqueeze(test_input, 0)
        net_output = self.net(test_input1)

        out = self.compute_lxp(test_input1)

        n = len(self.relu_pos)

        b_list = [2**i for i in range(n)] \
               + [1 + 2**i for i in range(1, n)] \
               + [2**(n-1) + 2**i for i in range(n-2)]

        ninput = 60*3*512
        noutput = 3*256

        nsamples = len(b_list)

        X = np.zeros((ninput, nsamples))
        Y = np.zeros((noutput, nsamples))

        for k, b in enumerate(b_list):
            self.lxp1(b)

            x, y = self.lr_vectors
        
            x_array = x.detach().numpy().flatten()
            y_array = y.detach().numpy().flatten()

            X[:, k] = x_array / np.linalg.norm(x_array)
            Y[:, k] = y_array / np.linalg.norm(y_array)

        UX, sX, VX = np.linalg.svd(X, full_matrices=False)
        UY, sY, VY = np.linalg.svd(Y, full_matrices=False)

        self.basis = [UX[:, :n], UY[:, :n]]
        self.singvals = [sX, sY]
        self.test_no = i_test

        suffix = "_n{:02d}_i{:04d}".format(self.i_ens, i_test)

        np.save("_output/UX{:s}.npy".format(suffix), UX)
        np.save("_output/UY{:s}.npy".format(suffix), UY)
        np.save("_output/sX{:s}.npy".format(suffix), sX)
        np.save("_output/sY{:s}.npy".format(suffix), sY)
        np.save("_output/VX{:s}.npy".format(suffix), VX)
        np.save("_output/VY{:s}.npy".format(suffix), VY)


    def compute_coeff_stats(self):
        """
        Compute full projections to computed phi/psi basis for all
        2^n - 1 terms in the raw (uncollected) expansion

        Run after compute_basis()
        """

        n = len(self.relu_pos)

        UX, UY = self.basis

        ninput = UX.shape[0]
        noutput = UY.shape[0]

        suffix = "_n{:02d}_i{:04d}".format(self.i_ens, self.test_no)
        self.coeffX_fname = "_output/coeffX{:s}.dat".format(suffix)

        coeffX = np.memmap(self.coeffX_fname,
                           mode="w+",
                           shape=(n,2**n-1),
                           dtype=float)
        
        self.coeffY_fname = "_output/coeffY{:s}.dat".format(suffix)

        coeffY = np.memmap(self.coeffY_fname,
                           mode="w+",
                           shape=(n,2**n-1),
                           dtype=float)

        for b in range(1, 2**n):

            msg = "\n{:10d} {:s}".format(b, np.binary_repr(b))
            spc = 80 - len(msg) - 1
            sys.stdout.write(msg + " "*spc)

            self.lxp1(b)
            x, y = self.lr_vectors

            x_array = x.detach().numpy().flatten()
            y_array = y.detach().numpy().flatten()

            coeffX[:, b-1] = UX.T.dot(x_array)
            coeffY[:, b-1] = UY.T.dot(y_array)
        
        self.coeffX = coeffX
        self.coeffY = coeffY





