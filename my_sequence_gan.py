# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import pickle
import datetime
from discriminator import Discriminator
import json

from model import Seq2Seq
from squad import Squad  # text2id, データを生成する
from my_dataloader import Data_loader as Gen_Data_Loader

# path 名
SQUAD_DATA = '/home/nakanishi/data/squad/train-v1.1.json'
OUTPUT_DIR = '/home/nakanishi/my_works/gan/mySeqGan/save'
in_train_data_filename = 'in_train.txt'
in_eval_data_filename = 'in_eval.txt'
out_train_data_filename = 'out_train.txt'
out_eval_data_filename = 'out_eval.txt'


##### 学習のパラメータ #####
PRE_EPOCH_NUM = 120
EPOCH_NUM = 200000
BATCH_SIZE = 64
START_TOKEN = 0
LEARNING_RATE = 0.01

##### 使用するデータのパラメータ #####
DATA_RESET = False # 新しくデータを生成するかどうか
IN_SEQ_LENGTH = 100
OUT_SEQ_LENGTH = 20
VOCAB_SIZE = 5000
MAX_DATA_NUM = 10000

##### 日時ファイル生成 結果の保存 #####
d = datetime.datetime.today()
year = str(d.year)[2:]
time = d.strftime("%m%d_%H%M")
TIME_NAME = year + '_' + str(time)
RESULT_DIR = os.path.join(OUTPUT_DIR, TIME_NAME)
os.mkdir(RESULT_DIR)
print('\nmake directory ', RESULT_DIR)


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

        Loader.save_textfile(in_train_data_file,
                             out_train_data_file, limit=0.8)
        Loader.save_textfile(in_eval_data_file, out_eval_data_file, limit=0.2, Train=False)

        with open(os.path.join(OUTPUT_DIR, DAT_NAME), 'wb') as f:
            pickle.dump(Loader, f)
        print('save ', (os.path.join(OUTPUT_DIR, DAT_NAME)))

    return Loader

"""
def pre_train_epoch(sess, model, data_loader):
    data_loader.reset_pointer()
    loss, predictions = model.pre_train_epoch(
        sess, model, IN_SEQ_LENGTH, OUT_SEQ_LENGTH, data_loader)

    return loss, predictions
"""

"""
def pre_train_epoch(sess, model, data_loader):
    data_loader.reset_pointer()
    encoder_inputs, decoder_inputs, labels, weights, outputs = model.construct_graph()
    loss_op, opt_op, predictions_op, saver_op = model.train_step(encoder_inputs, decoder_inputs, labels, weights, outputs)

    losses = []
    for it in range(data_loader.num_batch):
        X_batch, y_batch, w_batch = data_loader.next_batch()

        # feed_dict 構築
        feed_dict = {}
        feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
        feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)})
        feed_dict.update({labels[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
        feed_dict.update({weights[i]:w_batch[i] for i in range(OUT_SEQ_LENGTH)})

        l, _, predictions = sess.run([loss_op, opt_op, predictions_op], feed_dict=feed_dict)

        losses.append(l)
    loss = np.mean(losses)
"""

def main():

    # generatorを学習するためのデータを構築
    build_file_names = [in_train_data_filename, in_eval_data_filename,
                        out_train_data_filename, out_eval_data_filename]
    build_file_pathes = [os.path.join(
        OUTPUT_DIR, build_file_names[i]) for i in range(4)]
    Squad = make_datafile(build_file_pathes)

    print('Loading Data...')
    gen_data_loader = Gen_Data_Loader(BATCH_SIZE)
    gen_test_data_loader = Gen_Data_Loader(BATCH_SIZE)


    generator = Seq2Seq(VOCAB_SIZE, BATCH_SIZE, embedding_dim=128, hidden_size=100,
                        enc_seq_length=IN_SEQ_LENGTH, dec_seq_length=OUT_SEQ_LENGTH, start_token=START_TOKEN,
                        learning_rate=LEARNING_RATE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # log file
    log = open(os.path.join(RESULT_DIR, 'log.txt'), 'w')

    # 全バッチ生成
    gen_data_loader.create_batches(Squad.in_train_data_file, Squad.out_train_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)
    gen_test_data_loader.create_batches(Squad.in_eval_data_file, Squad.out_eval_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)

    # positiveデータを使ってgeneratorをpre-train
    print('\nStart Pre-Training')
    log.write('----- pre Training -----\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss, predictions = pre_train_epoch(sess, generator, gen_data_loader)
        print('epoch ', epoch, ' loss ', loss)
        log.write('epoch ' + str(epoch) + ' loss ' + str(loss) + '\n')


if __name__ == '__main__':
    main()
