# -*- coding: utf-8 -*-

"""
QuestionGeneratorがlstm
それの学習とテストを実行するプログラム
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
"""

import os
import numpy as np
import tensorflow as tf
import sys
import json
import pickle

from dataloader import Data_loader
from model import *
from squad import Squad

DATA_RESET = False

IN_SEQ_LENGTH = 500
OUT_SEQ_LENGTH = 20

LEARNING_RATE=0.01

START_TOKEN = 0

EPOCH_NUM = 20000
BATCH_SIZE = 64
VOCAB_SIZE = 5000
MAX_DATA_NUM = 10000

SQUAD_DATA = '/home/nakanishi/data/squad/train-v1.1.json'
OUTPUT_DIR = '/home/nakanishi/my_works/qGenerator/save'

# real_text file (index)
in_train_data_filename = 'in_train.txt'
in_eval_data_filename = 'in_eval.txt'
out_train_data_filename = 'out_train.txt'
out_eval_data_filename = 'out_eval.txt'

def make_datafile(datafile_names):
    DAT_NAME = 'squad.dat'
    in_train_data_file, in_eval_data_file, out_train_data_file, out_eval_data_file = datafile_names

    if not DATA_RESET:
        if os.path.exists(in_train_data_file) and os.path.exists(in_eval_data_file):
            if os.path.exists(os.path.join(OUTPUT_DIR, DAT_NAME)):
                print('Already exists datafiles.')
                with open(os.path.join(OUTPUT_DIR, DAT_NAME), 'rb') as f:
                    return pickle.load(f)

    print('Building Datafiles...')

    # input data (encode data)
    data_path = SQUAD_DATA
    with open(data_path) as f:
        data = json.load(f)

	Loader = Squad(data, vocab_size=VOCAB_SIZE, in_seq_length=IN_SEQ_LENGTH,
                out_seq_length=OUT_SEQ_LENGTH, max_data_num=MAX_DATA_NUM)

	Loader.save_textfile(in_train_data_file, out_train_data_file, limit=0.8)
	Loader.save_textfile(in_eval_data_file, out_eval_data_file, limit=0.2)

	with open(os.path.join(OUTPUT_DIR, DAT_NAME), 'wb') as f:
		pickle.dump(Loader, f)
	print('save ', (os.path.join(OUTPUT_DIR, DAT_NAME)))

    return Loader

def save_predictions(predictions, Xs, ys, filepath, S):
    predictions = list(np.array(predictions).transpose(1, 0, 2))
    if X != None and y != None:
        Xs = list(np.array(Xs).T)
        ys = list(np.array(ys).T)
        with open(filepath, 'w') as f:
            f.write('='*30)
            for line, X, y in zip(predictions, Xs, ys):
                f.write('-'*10+'input'+'-'*10)
                for w in X:
                    index = int(np.argmax(w))
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'label'+'-'*10)
                for w in y:
                    index = int(np.argmax(w))
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'prediction'+'-'*10)
                for w in line:
                    index = int(np.argmax(w))
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')

    else:
        with open(filepath, 'w') as f:
            for line in predictions:
                #p = p[0] # 最初のバッチだけ取り出し.
                for w in line:
                    index = int(np.argmax(w))
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
    print('save ', filepath)

def main():
    assert START_TOKEN == 0 # START_TOKENが0出ない時に例外?

    # log file
    log = open(os.path.join(OUTPUT_DIR, 'log2.txt'), 'w')

    in_train_data_file = os.path.join(OUTPUT_DIR, in_train_data_filename)
    in_eval_data_file = os.path.join(OUTPUT_DIR, in_eval_data_filename)
    out_train_data_file = os.path.join(OUTPUT_DIR, out_train_data_filename)
    out_eval_data_file = os.path.join(OUTPUT_DIR, out_eval_data_filename)

    Squad = make_datafile([in_train_data_file, in_eval_data_file, out_train_data_file, out_eval_data_file])

    log.write('vocab size '+str(Squad.vocab_size)+'\n')
    log.write('in seq length '+str(Squad.in_seq_length)+'\n')
    log.write('out seq length '+str(Squad.out_seq_length)+'\n')
    log.write('data num '+str(Squad.data_num)+'\n')

    print('Loading Data...')
    data_loader = Data_loader(BATCH_SIZE)
    test_data_loader = Data_loader(BATCH_SIZE)

    graph = tf.Graph()
    with graph.as_default():
        model = Seq2Seq(VOCAB_SIZE, BATCH_SIZE, embedding_dim=128, hidden_size=100,
                    enc_seq_length=IN_SEQ_LENGTH, dec_seq_length=OUT_SEQ_LENGTH, start_token=START_TOKEN,
                    learning_rate=LEARNING_RATE)

        encoder_inputs, decoder_inputs, labels, weights, outputs = model.construct_graph()
        #test_encoder_inputs, test_decoder_inputs, test_labels, test_weights, test_outputs = model.construct_graph(Test=True)
        #test_encoder_inputs, test_decoder_inputs, test_labels, test_weights, test_outputs = model.construct_graph()

        loss_op, opt_op, predictions_op, saver_op = model.train_step(encoder_inputs, decoder_inputs, labels, weights, outputs)
        #test_loss_op, test_predictions_op = model.test_step(test_encoder_inputs, test_decoder_inputs, test_labels, test_weights, test_outputs)
        test_loss_op, test_predictions_op = model.test_step(encoder_inputs, decoder_inputs, labels, weights, outputs)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())

        data_loader.create_batches(in_train_data_file, out_train_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)
        test_data_loader.create_batches(in_eval_data_file, out_eval_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)

        print('Stert Training...')
        log.write('Training')

        for epoch in xrange(EPOCH_NUM):
            losses = []
            data_loader.reset_pointer()

            for it in range(data_loader.num_batch):
                #print('epoch ', epoch, ' batch ', it)
                X_batch, y_batch, w_batch = data_loader.next_batch()

                # feed_dict 構築
                feed_dict = {}
                feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
                feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)})
                feed_dict.update({labels[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
                feed_dict.update({weights[i]:w_batch[i] for i in range(OUT_SEQ_LENGTH)})

                #l, w, o = sess.run([labels, weights, outputs], feed_dict=feed_dict)

                l, _, predictions = sess.run([loss_op, opt_op, predictions_op], feed_dict=feed_dict)

                #loss = sess.run([loss_op], feed_dict=feed_dict)
                losses.append(l)
            loss = np.mean(losses)
            print('epoch ', epoch, ' loss ', loss)
            log.write('epoch '+str(epoch)+' loss '+str(loss)+'\n')

            if epoch % 5 == 0:

                test_losses = []
                test_data_loader.reset_pointer()
                for it in range(test_data_loader.num_batch):
                    #print('epoch ', epoch, ' batch ', it)
                    test_X_batch, test_y_batch, test_w_batch = test_data_loader.next_batch()

                    # feed_dict 構築
                    feed_dict = {}
                    #feed_dict = {test_encoder_inputs[i]: test_X_batch[i] for i in range(IN_SEQ_LENGTH)}
                    #feed_dict.update({test_decoder_inputs[i]:test_y_batch[i] for i in range(OUT_SEQ_LENGTH)})
                    #feed_dict.update({test_labels[i]:test_y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
                    #feed_dict.update({test_weights[i]:test_w_batch[i] for i in range(OUT_SEQ_LENGTH)})
                    feed_dict = {encoder_inputs[i]: test_X_batch[i] for i in range(IN_SEQ_LENGTH)}
                    feed_dict.update({decoder_inputs[i]:test_y_batch[i] for i in range(OUT_SEQ_LENGTH)})
                    feed_dict.update({labels[i]:test_y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
                    feed_dict.update({weights[i]:test_w_batch[i] for i in range(OUT_SEQ_LENGTH)})


                    #l, w, o = sess.run([labels, weights, outputs], feed_dict=feed_dict)

                    test_l, test_predictions = sess.run([test_loss_op, test_predictions_op], feed_dict=feed_dict)
                    #loss = sess.run([loss_op], feed_dict=feed_dict)
                    test_losses.append(test_l)

                test_loss = np.mean(test_losses)
                print('TEST  epoch ', epoch, ' loss ', test_loss)
                log.write('TEST epoch '+str(epoch)+' loss '+str(loss)+'\n')

                # trainのprediction保存
                prediction_path = os.path.join(OUTPUT_DIR, X_batch[:IN_SEQ_LENGTH], y_batch[:OUT_SEQ_LENGTH],'train_prediction2.txt')
                save_predictions(predictions, prediction_path, Squad)

                # testのprediction保存
                prediction_path = os.path.join(OUTPUT_DIR, test_X_batch[:IN_SEQ_LENGTH], test_y_batch[:OUT_SEQ_LENGTH], 'test_prediction2.txt')
                save_predictions(test_predictions, prediction_path, Squad)

    log.close()

if __name__ == '__main__':
    main()
