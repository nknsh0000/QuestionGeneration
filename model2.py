# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

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



        #def construct_graph(self, Test=False):
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


        #def return_seq2seq(self, encoder_inputs, decoder_inputs, TEST):
        TEST = False
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols = self.vocab_size,
                                num_decoder_symbols = self.vocab_size,
                                embedding_size = self.embedding_dim,
                                feed_previous = TEST
                                )
        #return outputs, states

        #def loss(self, output, label, weight):
        #print(output)
        #return tf.contrib.seq2seq.sequence_loss(output, label, weight)
        #return tf.contrib.legacy_seq2seq.sequence_loss(output, label, weight)
        loss_op = tf.contrib.legacy_seq2seq.sequence_loss(outputs, labels, weights)

    #def optimizer(self, loss):
        #return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        opt_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_op)
        predictions_op = predictions = tf.stack([tf.nn.softmax(output) for output in outputs])

        saver_op = tf.train.Saver()

    def train_step(self, encoder_inputs, decoder_inputs, labels, weights, outputs):
        """
        lossとupdate_opを返す.
        """

        #encoder_inputs, decoder_inputs, labels, weights, outputs = self._construct_graph()
        predictions = tf.stack([tf.nn.softmax(output) for output in outputs])

        #losses = []
        #for output, label, weight in zip(outputs, labels, weights):
        loss = self.loss(outputs, labels, weights)
            #losses.append(l)
        #loss = np.mean(losses)

        opt = self.optimizer(loss)
        saver = tf.train.Saver()

        return loss, opt, predictions, saver

    def test_step(self, encoder_inputs, decoder_inputs, labels, weights, outputs):
        """
        lossを返す.
        """

        #encoder_inputs, decoder_inputs, labels, weights, outputs = self._construct_graph()
        predictions = tf.stack([tf.nn.softmax(output) for output in outputs])

        #losses = []
        #for output, label, weight in zip(outputs, labels, weights):
        loss = self.loss(outputs, labels, weights)

        return loss,predictions



    def pre_train_epoch(self, sess, model, in_seq_length, out_seq_length, data_loader):
        ### 1epoch分 pre-train
        #graph = tf.Graph()
        #with graph.as_default():
            #model = Seq2Seq(self.vocab_size, self.batch_size, embedding_dim=128, hidden_size=100,
            #            enc_seq_length=in_seq_length, dec_seq_length=out_seq_length, start_token=start_token,
            #            learning_rate=learning_rate)
        #encoder_inputs, decoder_inputs, labels, weights, outputs = model.construct_graph()

        #loss_op, opt_op, predictions_op, saver_op = model.train_step(encoder_inputs, decoder_inputs, labels, weights, outputs)
        #test_loss_op, test_predictions_op = model.test_step(encoder_inputs, decoder_inputs, labels, weights, outputs)

        losses = []
        for it in range(data_loader.num_batch):
            #print('epoch ', epoch, ' batch ', it)
            X_batch, y_batch, w_batch = data_loader.next_batch()

            # feed_dict 構築
            feed_dict = {}
            feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(in_seq_length)}
            feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(out_seq_length)})
            feed_dict.update({labels[i]:y_batch[i] for i in range(out_seq_length)}) # 正解データ
            feed_dict.update({weights[i]:w_batch[i] for i in range(out_seq_length)})

            #l, w, o = sess.run([labels, weights, outputs], feed_dict=feed_dict)

            l, _, predictions = sess.run([loss_op, opt_op, predictions_op], feed_dict=feed_dict)

            #loss = sess.run([loss_op], feed_dict=feed_dict)
            losses.append(l)
        loss = np.mean(losses)
        #outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x})

        return loss, predictions
