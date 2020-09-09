#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# tts/WaveRNN/dataset.py
#
# WaveRNN dataset
#

import sys
sys.path.append('../')
import os
import numpy as np
import torch
from datasets.voice_dataset import VoiceDataset, normalize
from tools.libaudio.utils import normalize, reshape_with_window
from tools.libaudio.encodes import bit_encode, split_signal, combine_signal


class WaveRNNDataset(VoiceDataset):

    def __init__(
            self, sample_rate=24000, key_name='jsut_ver1.1',
            with_conditions=False, remove_silence=True, batch_size=1, window_size=500):

        self.__root_dir__ = f'/diskB/6/Datasets/VoiceData/{key_name}/preprocessed'
        self.__wav_dir__ = f'/diskB/6/Datasets/VoiceData/{key_name}/wav24kHz'
        self.__f0_dir__ = f'{self.__root_dir__}/f0'
        self.__phoneme_dir__ = f'{self.__root_dir__}/phoneme'
        self.__sample_rate__ = sample_rate
        self.__with_conditions__ = with_conditions
        self.__remove_silence__ = remove_silence
        self.__batch_size__ = batch_size
        self.__window_size__ = window_size

        self.wav_file_names = os.listdir(self.__wav_dir__)
        self.f0_file_names = os.listdir(self.__f0_dir__)
        self.phonemes_file_names = os.listdir(self.__phoneme_dir__)

    def collate_fn(self, items):
        """

        important:
        `Whatever the batch_size is, returns only the first item.`

        returns:
            wav_c (torch.FloatTensor): [c0, c1, c2,..,cM] (B, max_len)
            wav_f (torch.FloatTensor): [f0, f1, f2,..,fM] (B, max_len)
            target_c (torch.LongTensor): [c1, c2, c3,..,cM-1] (B, max_len)
            target_f (torch.LongTensor): [f1, f2, f3,..,fM-1] (B, max_len)
            f0 (ndarray): [f0_0, f0_1, ....,f0_M] (B, max_len)
            phonemes (ndarray): [f0_0, f0_1, ....,f0_M] (B, max_len)
        """
        for i, item in enumerate(items):
            start, end = self.utterance_edge_indices(item.get('wav')) if self.__remove_silence__ else (0, len(c)-1)

            wav_c = torch.FloatTensor(c[start:end]).view(1, -1)
            wav_f = torch.FloatTensor(f[start:end]).view(1, -1)

            # reshape
            wav_c = reshape_with_window(wav_c, self.__batch_size__, self.__window_size__)
            wav_f = reshape_with_window(wav_f, self.__batch_size__, self.__window_size__)

            target_c = wav_c[:, 1:].long()
            target_f = wav_f[:, 1:].long()
            if self.__with_conditions__:
                f0 = torch.FloatTensor(item.get('f0')).view(1, -1)
                phonemes = torch.FloatTensor(item.get('phonemes')).view(1, -1)
                return wav_c, wav_f, target_c, target_f, f0, phonemes
            else:
                return wav_c, wav_f, target_c, target_f

    def utterance_edge_indices(self, c: np.array):
        """Find the index where utterance begins and ends.

        returns:
            s (int): The index at which utterance begins.
            e (int): The index at which utterance ends.
        """
        s, e, r = 0, len(c), 8
        for x in c:
            if not (x in range(128 - r, 128 + r)): break
            s += 1
        for x in reversed(c):
            if not (x in range(128 - r, 128 + r)): break
            e -= 1
        return s, e



