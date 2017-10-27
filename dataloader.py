# -*- coding: utf-8 -*-

import numpy as np

class Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.in_token_stream = []
        self.out_token_stream = []
        self.weights = []
        self.in_sequence_batch = []
        self.out_sequence_batch = []
        self.weights_batch = []

        self.num_batch = 0 # 持ってるバッチ数
        self.pointer = 0

    def create_batches(self, in_data_file, out_data_file, in_seq_length, out_seq_length):
        self.in_token_stream = []
        self.weights = []
        with open(in_data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == in_seq_length:
                    self.in_token_stream.append(parse_line)

        self.out_token_stream = []
        with open(out_data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == out_seq_length:
                    self.out_token_stream.append(parse_line)
                    self.weights.append([1.0]*len(parse_line) + [1.0] + [0.0]*(out_seq_length-len(parse_line))) # このseq_lengthはoutputのseq_length

        self.num_batch = int(len(self.in_token_stream) / self.batch_size)

        self.num_samples = self.num_batch * self.batch_size
        self.in_token_stream = self.in_token_stream[:(self.num_batch * self.batch_size)]
        self.in_sequence_batch = np.split(np.array(self.in_token_stream), self.num_batch, 0)
        self.out_token_stream = self.out_token_stream[:(self.num_batch * self.batch_size)]
        self.out_sequence_batch = np.split(np.array(self.out_token_stream), self.num_batch, 0)

        self.weights = self.weights[:(self.num_batch * self.batch_size)]
        self.weights_batch = np.split(np.array(self.weights), self.num_batch, 0)
        self.pointer = 0
        print('BATCH_SIZE   :', self.batch_size)
        print('NUM_BATCH    :', self.num_batch)


    def next_batch(self):
        inp = self.in_sequence_batch[self.pointer]
        out = self.out_sequence_batch[self.pointer]
        w = self.weights_batch[self.pointer]

        inp = list(np.array(inp).T)
        out = list(np.array(out).T)
        w = list(np.array(w).T)

        self.pointer = (self.pointer + 1) % self.num_batch
        return inp, out, w

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
