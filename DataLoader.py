#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import time
import csv
import collections
import cPickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn


class DataLoader(object):
    
    def __init__(self, is_training, utils_dir, data_path, batch_size, seq_length, labels, encoding='utf8'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        if is_training:
            self.utils_dir = utils_dir
            label_file = os.path.join(utils_dir, 'labels.pkl')
            #parse the Label file, it is pickle 
            print "read the label file"
            with open(label_file, 'r') as f:
                self.labels = pickle.load(f)
            self.label_size = len(self.labels)

        elif labels is not None:
            self.labels = labels
            self.label_size = len(self.labels)
            
        print "read the data file"
        self.preprocess(data_path)
        self.reset_batch_pointer()

    #transform the data format: 223|434|6767
    def transform(self, d):
        print d
        if pd.isnull(d):
             d='0'
        new_d=d.split('|')
        if len(new_d) >= self.seq_length:
            new_d = new_d[:self.seq_length]
        else:
            new_d = new_d + [0] * (self.seq_length - len(new_d))
        return new_d

    def preprocess(self, data_path):
        data = pd.read_csv(data_path,dtype={'code':str})
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        print("the total:%s"%(self.tensor.shape[0]))
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'

        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)


    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        print len(self.x_batches)
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0
