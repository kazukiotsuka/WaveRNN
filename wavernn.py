#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.libaudio.encodes import split_signal, combine_signal


class WaveRNN(nn.Module):
    """WaveRNN.

    https://arxiv.org/abs/1802.08435

    math::
        xt = [ct-1, ft-1, ct]  # input
        ut = σ(Ru ht-1 + Iu*xt + bu)  # update gate
        rt = σ(Rr ht-1 + Ir*xt + br)  # reset gate
        et = tanh(rt∘(Re ht-1) + Ie*xt + be)  # recurrent unit
        ht = ut∘ht-1 + (1-u)∘et  # next hidden state
        yc, yf = split(ht)  # coarse, fine
        P(ct) = softmax(O2 relu(O1 yc))  # coarse distribution
        P(ft) = softmax(O4 relu(O3 yf))  # fine distribution
    """

    def __init__(self, hidden_size=896, n_class=256, batch_size=1, sample_rate=24000, disable_cuda=False, device=None):
        super(WaveRNN, self).__init__()

        self.__hidden_size__ = hidden_size
        self.__n_class__ = n_class
        self.__batch_size__ = batch_size
        self.__sample_rate__ = sample_rate
        self.__disable_cuda__ = disable_cuda
        self.__device__ = device

        # gating unit R (U)
        self.R = nn.Linear(hidden_size, hidden_size*3, bias=False)

        # hidden state I (W) (coarse, fine)
        self.Ic = nn.Linear(2, 3*hidden_size//2, bias=False)
        self.If = nn.Linear(3, 3*hidden_size//2, bias=False)

        # transform matrices into categorical distributions
        self.O1 = nn.Linear(hidden_size//2, hidden_size//2)
        self.O2 = nn.Linear(hidden_size//2, hidden_size//2)
        self.O3 = nn.Linear(hidden_size//2, n_class)
        self.O4 = nn.Linear(hidden_size//2, n_class)

        # bias terms
        self.bu = nn.Parameter(torch.zeros(hidden_size))
        self.br = nn.Parameter(torch.zeros(hidden_size))
        self.be = nn.Parameter(torch.zeros(hidden_size))

        # print summary
        self.print_summary()

        # to specific device
        if device: self.to_device(device)

    def init_state(self):
        """Initialize hidden state.

        returns:
            hidden_state (torch.FloatTensor): initialized hidden state with zeros
        """
        return torch.zeros(self.__batch_size__, self.__hidden_size__) if self.__disable_cuda__\
            else torch.zeros(self.__batch_size__, self.__hidden_size__).cuda(device=self.__device__)

    def to_device(self, device=None, cpu=False):
        if cpu:
            device = torch.device('cpu')
            self.__device__ = device
            self.__disable_cuda__ = True
        _, _, _ = self.R.to(device), self.Ic.to(device), self.If.to(device)
        _, _, _, _, = self.O1.to(device), self.O2.to(device), self.O3.to(device), self.O4.to(device)
        _, _, _ = self.bu.to(device), self.br.to(device), self.be.to(device)

    def forward(self, ht_1, ct_1, ft_1, ct):
        """Forward step.

        Params:
            (ht_1 (torch.FloatTensor): previous hidden state (1, hidden_size))
            ct_1 (torch.FloatTensor): previous coarse out (1, 1)
            ft_1 (torch.FloatTensor): previous fine out (1, 1)
            ct (torch.FloatTensor): current coarse out (1, 1)
        """
        hidden_size = self.__hidden_size__

        # input xt = [ct-1, ft-1, ct]
        xt = [ct_1, ft_1, ct]

        # fully connected (previous hidden ht-1) x (gating unit R (U))
        Rht_1 = self.R(ht_1)

        # fully connected (xt) x (hidden state Ic (W)) (coarse part)
        Icxt = self.Ic(torch.cat((xt[0], xt[1]), dim=1))

        # fully connected (xt) x (hidden state If (W)) (fine part)
        Ifxt = self.If(torch.cat((xt[0], xt[1], xt[2]), dim=1))

        # split I
        Iuxt = torch.cat(
            (Icxt[:, :hidden_size // 2],
             Ifxt[:, :hidden_size // 2]), dim=1)
        Irxt = torch.cat(
            (Icxt[:, hidden_size // 2:2 * hidden_size // 2],
             Ifxt[:, hidden_size // 2:2 * hidden_size // 2]), dim=1)
        Iext = torch.cat(
            (Icxt[:, 2 * hidden_size // 2:3 * hidden_size // 2],
             Ifxt[:, 2 * hidden_size // 2:3 * hidden_size // 2]),
            dim=1)

        # ut (update gate)
        ut = torch.sigmoid(Rht_1[:, :hidden_size] + Iuxt + self.bu)

        # rt (reset gate)
        rt = torch.sigmoid(Rht_1[:, hidden_size:hidden_size*2] + Irxt + self.br)

        # et (recurrent unit)
        et = torch.tanh(rt * (Rht_1[:, hidden_size*2:hidden_size*3]) + Iext + self.be)

        # ht (next hidden state)
        ht = ut * ht_1 + (1 - ut) * et

        # yc (coarse out), yf (fine out)
        yc, yf = torch.split(ht, hidden_size // 2, dim=1)

        # P(ct) coarse distribution (1, n_class)
        Pct = self.O3(torch.relu(self.O1(yc)))

        # P(ft) fine distribution (1, n_class)
        Pft = self.O4(torch.relu(self.O2(yf)))

        # Pct (1, n_class), Pft (1, n_class), ht (1, hidden_size)
        return ht, Pct, Pft

    # def generate(self, duration_ms: int, parallel=4, device='cpu'):
    #     """Generate a signal.
    #     """
    #     #import pdb; pdb.set_trace()
    #     self.__device__ = 'cpu'

    #     seq_len = (duration_ms // 1000) * self.__sample_rate__

    #     fold_len = seq_len//parallel
    #     overlap = 100
    #     mel = self.fold_with_overlap(torch.zeros(parallel, 10, seq_len), fold_len, overlap)
    #     N = mel.shape[0]


    #     # initialize hidden state
    #     ht_1 = torch.zeros(N, self.__hidden_size__).to(device)


    #     # starting samples
    #     ct_1, ft_1 = torch.zeros(N, 1), torch.zeros(N, 1)


    #     bcu, bfu = torch.split(self.bu, self.__hidden_size__//2)
    #     bcr, bfr = torch.split(self.br, self.__hidden_size__//2)
    #     bce, bfe = torch.split(self.be, self.__hidden_size__//2)

    #     # generated samples
    #     c_samples, f_samples = [], []

    #     start = time.time()
    #     speed = None

    #     with torch.no_grad():

    #         for t in range(fold_len):

    #             # ht-1 x gating unit R (U)
    #             Rht_1 = self.R(ht_1)

    #             Rcuht_1, Rfuht_1, \
    #             Rcrht_1, Rfrht_1,\
    #             Rceht_1, Rfeht_1 = torch.split(Rht_1, self.__hidden_size__//2, dim=1)

    #             # xt x Ic (W) (coarse)
    #             Icxt = self.Ic(torch.cat([ct_1, ft_1], dim=1))

    #             # split Icxt for u, r, e
    #             Icuxt, Icrxt, Icext = torch.split(Icxt, self.__hidden_size__//2, dim=1)

    #             # split ht-1 to coarse, fine
    #             hct_1, hft_1 = torch.split(ht_1, self.__hidden_size__//2, dim=1)

    #             # ut, rt, et (coarse)
    #             uct = torch.sigmoid(Rcuht_1 + Icuxt + bcu)
    #             rct = torch.sigmoid(Rcrht_1 + Icrxt + bcr)
    #             ect = torch.tanh(rct*Rceht_1 + Icext + bce)

    #             # next hidden state (coarse)
    #             yc = hc = uct*hct_1 + (1-uct)*ect

    #             # distribution (coarse)
    #             pc = self.O3(torch.relu(self.O1(yc)))

    #             # sample (coarse)
    #             ct = torch.distributions.Categorical(pc.exp()).sample()
    #             print(ct_1.shape)
    #             print(ct.view(0, 1).shape)
    #             print(ft_1.shape)
    #             c_samples += [ct]

    #             # xt x If (W) (coarse)
    #             print(torch.cat([ct_1, ft_1, ct.view(0, 1).float()], dim=1).shape)
    #             Ifxt = self.If(torch.cat([ct_1, ft_1, ct.float()], dim=1))

    #             # split If for u,r,e
    #             Ifuxt, Ifrxt, Ifext = torch.split(Ifxt, self.__hidden_size__//2)

    #             # ut, rt, et (fine)
    #             uft = torch.sigmoid(Rfuht_1 + Ifuxt + bfu)
    #             rft = torch.sigmoid(Rfrht_1 + Ifrxt + bfr)
    #             eft = torch.tanh(rft*Rfeht_1 + Ifext + bfe)

    #             # next hidden state (fine)
    #             yf = hf = uft*hft_1 + (1-uft)*eft

    #             # distribution (fine)
    #             pf = self.O4(torch.relu(self.O2(yf)))

    #             ft = torch.distributions.Categorical(pf.exp()).sample()
    #             f_samples += [ft]

    #             # update hidden state
    #             ht_1 = torch.cat([hc, hf], dim=1)

    #             # update ct_1, ft_1
    #             ct_1, ft_1 = ct.float(), ft.float()

    #             # time check
    #             if t % 1000 == 0:
    #                 speed = (t + 1) / (time.time() - start)
    #                 print(f'generate {t+1}/{seq_len}, Speed: {speed:.2f} samples/sec')

    #     coarse = torch.stack(c_samples).squeeze(1).cpu().numpy()
    #     fine = torch.stack(f_samples).squeeze(1).cpu().numpy()
    #     combined

    #     wav = self.xfade_and_unfold(combined, fold_len, overlap)

    #     samples_per_sec = round(speed, 3)
    #     batch = N
    #     total_samples = fold_len * batch
    #     x_realtime = round(speed/self.sample_rate, 3)
    #     return wav, samples_per_sec, batch, total_samples, x_realtime


    def generate(self, duration_ms: int):
        """Generate a signal.
        """
        #import pdb; pdb.set_trace()

        seq_len = (duration_ms // 1000) * self.__sample_rate__

        # initialize hidden state
        ht_1 = self.init_state()

        # starting samples
        ct_1, ft_1 = torch.zeros(1), torch.zeros(1)

        # bias terms
        bcu, bfu = torch.split(self.bu, self.__hidden_size__//2)
        bcr, bfr = torch.split(self.br, self.__hidden_size__//2)
        bce, bfe = torch.split(self.be, self.__hidden_size__//2)

        # generated samples
        c_samples, f_samples = [], []

        start = time.time()
        speed = None

        with torch.no_grad():

            for t in range(seq_len):

                # ht-1 x gating unit R (U)
                Rht_1 = self.R(ht_1)
                Rcuht_1, Rfuht_1,\
                Rcrht_1, Rfrht_1,\
                Rceht_1, Rfeht_1 = torch.split(Rht_1, self.__hidden_size__//2, dim=1)

                # xt x Ic (W) (coarse)
                Icxt = self.Ic(torch.cat([ct_1, ft_1]))

                # split Icxt for u, r, e
                Icuxt, Icrxt, Icext = torch.split(Icxt, self.__hidden_size__//2)

                # split ht-1 to coarse, fine
                hct_1, hft_1 = torch.split(ht_1, self.__hidden_size__//2, dim=1)

                # ut, rt, et (coarse)
                uct = torch.sigmoid(Rcuht_1 + Icuxt + bcu)
                rct = torch.sigmoid(Rcrht_1 + Icrxt + bcr)
                ect = torch.tanh(rct*Rceht_1 + Icext + bce)

                # next hidden state (coarse)
                yc = hc = uct*hct_1 + (1-uct)*ect

                # distribution (coarse)
                pc = self.O3(torch.relu(self.O1(yc)))

                # sample (coarse)
                ct = torch.distributions.Categorical(pc.exp()).sample()
                c_samples += [ct]

                # xt x If (W) (coarse)
                Ifxt = self.If(torch.cat([ct_1, ft_1, ct.float()]))

                # split If for u,r,e
                Ifuxt, Ifrxt, Ifext = torch.split(Ifxt, self.__hidden_size__//2)

                # ut, rt, et (fine)
                uft = torch.sigmoid(Rfuht_1 + Ifuxt + bfu)
                rft = torch.sigmoid(Rfrht_1 + Ifrxt + bfr)
                eft = torch.tanh(rft*Rfeht_1 + Ifext + bfe)

                # next hidden state (fine)
                yf = hf = uft*hft_1 + (1-uft)*eft

                # distribution (fine)
                pf = self.O4(torch.relu(self.O2(yf)))

                ft = torch.distributions.Categorical(pf.exp()).sample()
                f_samples += [ft]

                # update hidden state
                ht_1 = torch.cat([hc, hf], dim=1)

                # update ct_1, ft_1
                ct_1, ft_1 = ct.float(), ft.float()

                # time check
                if t % 1000 == 0:
                    speed = (t + 1) / (time.time() - start)
                    print(f'generate {t+1}/{seq_len}, Speed: {speed:.2f} samples/sec')

        coarse = torch.stack(c_samples).squeeze(1).cpu().numpy()
        fine = torch.stack(f_samples).squeeze(1).cpu().numpy()
        combined = combine_signal(coarse, fine)

        # normalize
        wav = combined / np.max(combined)

        samples_per_sec = round(speed, 3)
        batch = 1
        total_samples = len(wav) #fold_len * batch
        x_realtime = round(speed/24000, 3)
        return wav, samples_per_sec, batch, total_samples, x_realtime


    def print_summary(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def settings(self) -> dict:
        return {
            'hidden_size': self.__hidden_size__,
            'n_class': self.__n_class__,
        }

    def save_model(self, save_model_path: str):
        try:
            print(f'trying to save model parameters {self.state_dict().keys()} to {save_model_path} ..')
            torch.save(self.state_dict(), save_model_path)
            # torch.save(self, save_model_path)  # * this fails when data parallel
        except Exception as e:
            print(e)

    def load_model(self, model_file_path: str):
        try:
            self.load_state_dict(
                torch.load(model_file_path, map_location=lambda storage, loc: storage))
            # torch.load(model_file_path)  # * this fails if trained on multiple GPU. use state dict.
        except Exception as e:
            print(e)

    # https://github.com/fatchord/WaveRNN/
    def fold_with_overlap(self, x, target, overlap):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features).to(self.__device__)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    # https://github.com/fatchord/WaveRNN/
    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    # https://github.com/fatchord/WaveRNN
    def pad_tensor(self, x, pad, side='both') :
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c).to(self.__device__)
        if side == 'before' or side == 'both' :
            padded[:, pad:pad+t, :] = x
        elif side == 'after' :
            padded[:, :t, :] = x
        return padded
