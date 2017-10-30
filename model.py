# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

class Seq2Seq(object):
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_size,\
        enc_seq_length, dec_seq_length, start_token, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.start_token = start_token
        self.learning_rate = learning_rate

    def construct_graph(self, Test=False):
        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()
        weights = list()

        #print(self.enc_seq_length)
        for _ in range(self.enc_seq_length):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=(None, )))
        for _ in range(self.dec_seq_length):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
            labels.append(tf.placeholder(tf.int32, shape=(None,)))
            weights.append(tf.placeholder(tf.float32, shape=(None, )))

        #outputs, states = self.return_seq2seq(encoder_inputs, decoder_inputs, TEST=Test)

        #return encoder_inputs, decoder_inputs, labels, weights, outputs
        return encoder_inputs, decoder_inputs, labels, weights,

    def output(self, encoder_inputs, decoder_inputs, Test=False):
        #outputs, states = self.return_seq2seq(encoder_inputs, decoder_inputs, TEST=Test)
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(
        outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                            encoder_inputs,
                            decoder_inputs,
                            cell,
                            num_encoder_symbols = self.vocab_size,
                            num_decoder_symbols = self.vocab_size,
                            embedding_size = self.embedding_dim,
                            feed_previous = Test
                            )
        return outputs, states


    def _pre_loss(self, output, label, weight):
        #print(output)
        #return tf.contrib.seq2seq.sequence_loss(output, label, weight)
        return tf.contrib.legacy_seq2seq.sequence_loss(output, label, weight)

    def _sample_loss(self, logits, targets, weights):
        log_perp_list = []

        for logit, target, weight in zip(logits, targets, weights):
            target = array_ops.reshape(target, [-1])
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logit)

            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        loss = math_ops.reduce_sum(log_perps)
        return loss

    def _ad_loss(self, rewards, logits, targets, weights):
        # get from rollout policy and discriminator
        log_perp_list = []
        rewards = tf.transpose(rewards) # 20, 64

        for i, (logit, target, weight) in enumerate(zip(logits, targets, weights)):
            reward = rewards[i]
            target = array_ops.reshape(target, [-1])
            reward = array_ops.reshape(reward, [-1])

            # これがlabelsに対するloss??
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logit)

            # これがdiscriminatorに対するloss??
            reward = tf.cast(reward, tf.float32)
            log_perp_list.append(crossent * weight * reward)

        log_perps = math_ops.add_n(log_perp_list)
        loss = math_ops.reduce_sum(log_perps)

        return loss

    def _optimizer(self, loss):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def generate(self,outputs):
        predictions = tf.stack([tf.nn.softmax(output) for output in outputs])
        return predictions

    def pre_train(self, labels, weights, outputs):
        loss = self._pre_loss(outputs, labels, weights)
        opt = self._optimizer(loss)
        return loss, opt

    def return_saver(self):
        return tf.train.Saver()


    def return_rewards(self):
        #return tf.placeholder(tf.float32, shape=[self.batch_size, self.dec_seq_length])
        return tf.placeholder(tf.float32, shape=[self.batch_size, self.dec_seq_length])


    def ad_train(self, rewards, labels, weights, outputs):
        loss = self._ad_loss(rewards, outputs, labels, weights)
        opt = self._optimizer(loss)
        return loss, opt

#end
