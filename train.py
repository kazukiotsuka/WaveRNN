#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# core/tts/WaveRNN/train.py
#

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tools.libaudio.encodes import split_signal, combine_signal
from tools.libaudio.tensors import reshape_with_window
from mlutils.utils import time_since, update_lr
from models.mlmodeldic import record_model

def train(
        model, dataset, model_name='wavernn', lr=1e-3, n_epoch=1, n_iter=1000, seq_len=1000, batch_size=1,
        max_sampling=9999999, disable_cuda=False, device=None, check_inference=True, verbose=False):
    """Train WaveRNN.

    params:
        model (WaveRNN): instance of WaveRNN
        dataset (list or DataLoader): array of ndarray or DataLoader instance
        model_name (str): model name to be saved
        lr (float): initial learnin rate (default. 1e-4)
        n_epoch (int): num train epoch (default. 1)
        n_iter (int): num iteration for a signal
        seq_len (int): input length to the recurrent unit
        max_sampling (int): maximum sampling number (optional)
        batch_size (int): minibatch size (only when dataset is list. optional)
        disable_cuda (bool): if True use cuda
        device (torch.device): cuda device (optional)
        verbose (bool): if True print a lot

    returns:
        losses (list): losses
        loss_aves (list): loss averages
        model (WaveRNN) : trained model
        coarse (list): training output samples (coarse)
        fine (list): training output samples (coarse)
    """

    optimizer = torch.optim.Adam(model.parameters())

    # learning rate
    for p in optimizer.param_groups:
        p['lr'] = lr

    # criterion
    if not disable_cuda:
        criterion = nn.CrossEntropyLoss().cuda(device=device)
    else:
        criterion = nn.CrossEntropyLoss()

    start = time.time()
    losses, loss_aves = [], []
    total_iter, total_step = 0, 0
    infer_c, infer_f = [], []

    # dataset from list
    if isinstance(dataset, list):
        n_wavs = len(dataset)
        loader = []
        for wav in dataset:
            wav_c, wav_f = split_signal(wav)
            wav_c, wav_f = torch.FloatTensor(wav_c).view(1, -1), torch.FloatTensor(wav_f).view(1, -1)
            wav_c = reshape_with_window(wav_c, batch_size, 100)
            wav_f = reshape_with_window(wav_f, batch_size, 100)
            target_c, target_f = wav_c[:, 1:].long(), wav_f[:, 1:].long()  # shift 1 index upon right
            loader.append((wav_c, wav_f, target_c, target_f))
    # dataset from DataLoader
    elif isinstance(dataset, DataLoader):
        n_wavs = len(dataset.dataset)
        loader = dataset
    else:
        print(f'type {type(dataset)} of {dataset} is invalid.')

    # epoch
    for epoch in range(n_epoch):
        # train all wavs
        for index, (wav_c, wav_f, target_c, target_f) in enumerate(loader):
            # skip if sample is too short
            if wav_c.shape[1] < seq_len:
                print(f'wav length {wav_c.shape[1]} is too short. skip')
                continue
            # exceeds max sampling num
            if index > max_sampling - 1:
                break

            # iter for a signal
            for i in range(n_iter):

                optimizer.zero_grad()
                loss, seq_loss = 0, 0
                r = np.random.randint(0, wav_c.shape[1]-seq_len-1)

                # recurrent step
                for t in range(seq_len-1):

                    # start position
                    pos = r+t

                    # ct-1, ft-1, ct, yc, yf
                    ct_1, ft_1 = wav_c[:, pos:pos + 1], wav_f[:, pos:pos + 1]
                    ct = target_c[:, pos:pos + 1].type(torch.FloatTensor)
                    yc, yf = target_c[:, pos:pos + 1].view(-1), target_f[:, pos:pos + 1].view(-1)
                    #if is_conditioned:
                    #    _f0 = f0[:, pos:(pos + 1)].unsqueeze(1)
                    #    _phonemes = phonemes[:, pos:(pos + 1)].unsqueeze(1)


                    # sanity check
                    assert torch.all(torch.eq(wav_c[:, 1:].long(), target_c[:, :]))
                    assert torch.all(torch.eq(wav_c[:, pos+1:pos+2], ct)) and\
                           torch.all(torch.eq(ct.long().squeeze(), yc)),\
                            f'not equal either {wav_c[:, pos+1:pos+2].squeeze()}, {ct.long()} or {yc}'

                    # cuda
                    if not disable_cuda:
                        ct_1, ft_1, ct, yc, yf = ct_1.cuda(device), ft_1.cuda(device),\
                            ct.cuda(device), yc.cuda(device), yf.cuda(device)
                        #if is_conditioned:
                        #    _f0 = _f0.cuda()

                    # check inputs
                    if verbose:
                        print('-'*100)
                        print(
                            (f'ct_1 {ct_1} {ct_1.shape}\nft_1 {ft_1} {ft_1.shape}\nct {ct} {ct.shape}\n'
                             f'yc {yc} {yc.shape}\nyf {yf} {yf.shape}'))
                        print('-'*100)

                    # forward
                    if t == 0: ht_1 = model.init_state()
                    ht_1, pct, pft = model(ht_1, ct_1, ft_1, ct)

                    # check outputs
                    if verbose:
                        print(f'pct {pct} {pct.shape} \nyc {yc} {yc.shape}')
                        print(f'pft {pft} {pft.shape} \nyf {yf} {yf.shape}')

                    # calcurate loss
                    loss_c, loss_f = criterion(pct, yc), criterion(pft, yf)
                    loss += loss_c + loss_f

                    # training sample
                    if check_inference:
                        infer_c += [torch.distributions.Categorical(pct.exp()).sample().cpu()]
                        infer_f += [torch.distributions.Categorical(pft.exp()).sample().cpu()]

                # back propergation
                loss.backward()
                optimizer.step()

                total_iter += 1

                # calc loss for the iteration
                seq_loss += round(float(loss.item()), 3) / seq_len

                # append to loss record
                losses += [seq_loss]
                loss_ave = np.average(losses)
                loss_aves += [loss_ave]

                if i % 1000 == 0 or i == n_iter-1:
                    print(f'epoch {epoch}/{n_epoch-1} index {index}/{n_wavs-1} iter: {i}/{n_iter} '
                          f'-- loss ave: {loss_ave:.4f} loss: {seq_loss:.2f} '
                          f'-- elapse: {time_since(start)} speed {((time.time() - start) / total_iter) * seq_len:.1f} steps/sec')

                # annealing
                update_lr(i, optimizer, annealing_rate=0.98, interval=1000)


    # record model
    model_path = record_model(model, key_name=model_name, loss_aves=loss_aves, n_iter=len(losses),
        settings={'lr': lr, 'n_epoch': n_epoch, 'n_iter': n_iter, 'seq_len': seq_len},
        model_path=(f'/diskB/6/out/models/wavernn/wavernn_epoch{n_epoch}_n_iter{n_iter}_seq_len{seq_len}_lr{lr}'
                    f'_loss{str(round(np.average(loss_aves), 3)).replace(".", "-")}'))
    # save model
    model.save_model(model_path)

    if check_inference:
        # coarse, fine
        coarse = torch.stack(infer_c).squeeze(1).cpu().numpy()
        fine = torch.stack(infer_f).squeeze(1).cpu().numpy()
        return losses, loss_aves, model, coarse, fine
    else:
        return losses, loss_aves, model
