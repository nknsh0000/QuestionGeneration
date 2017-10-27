# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import pickle
import datetime
import json
# 人様のコード
from discriminator import Discriminator
from rollout import ROLLOUT
# 自分のコード
from model import Seq2Seq
from squad import Squad  # text2id, データを生成する
from my_dataloader import Gen_Data_Loader
from my_dataloader import Dis_Data_Loader

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
GENERATED_NUM = 10000
#  Discriminator  Hyper-parameters
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
# Adversarial Training Parameters
TOTAL_BATCH = 200


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
print('\nMake Directory ', RESULT_DIR)


def make_datafile(datafile_names):
    DAT_NAME = 'squad.dat'
    in_train_data_file, in_eval_data_file, out_train_data_file, out_eval_data_file = datafile_names

    if not DATA_RESET:
        if os.path.exists(in_train_data_file) and os.path.exists(in_eval_data_file):
            if os.path.exists(os.path.join(OUTPUT_DIR, DAT_NAME)):
                print('Exists Datafiles Already.')
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

def save_predictions(predictions, Xs, ys, filepath, S):
    predictions = list(np.array(predictions).transpose(1, 0, 2))

    if Xs != None and ys != None:
        Xs = list(np.array(Xs).T)
        ys = list(np.array(ys).T)

        with open(filepath, 'w') as f:
            f.write('='*30+'\n')
            for line, X, y in zip(predictions, Xs, ys):
                f.write('-'*10+'input'+'-'*10+'\n')
                for w in X:
                    index = int(w)
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'label'+'-'*10+'\n')
                for w in y:
                    index = int(w)
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'prediction'+'-'*10+'\n')
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

def save_generated(generateds, filepath):
    # generateds (125, 1, 20, 64, 5000)

    with open(filepath, 'w')as f:
        for predictions in generateds:
            predictions = list(np.array(predictions).transpose(1, 0, 2))
            for line in predictions:
                for w in line:
                    index = int(np.argmax(w))
                    #word = S.return_word(index, error_word='***',output=True)
                    f.write(str(index))
                    f.write(' ')
                f.write('\n')
    print('save ', filepath)


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
    dis_data_loader = Dis_Data_Loader(BATCH_SIZE, OUT_SEQ_LENGTH)


    graph = tf.Graph()
    with graph.as_default():
        generator = Seq2Seq(VOCAB_SIZE, BATCH_SIZE, embedding_dim=128, hidden_size=100,
                            enc_seq_length=IN_SEQ_LENGTH, dec_seq_length=OUT_SEQ_LENGTH, start_token=START_TOKEN,
                            learning_rate=LEARNING_RATE)

        # generate???
        encoder_inputs, decoder_inputs, labels, weights, outputs = generator.construct_graph()
        # 学習
        loss_op, opt_op, predictions_op, saver_op = generator.train_step(encoder_inputs, decoder_inputs, labels, weights, outputs)
        # generate
        generate_op = generator.generate(encoder_inputs, decoder_inputs, labels, weights, outputs)

        discriminator = Discriminator(sequence_length=OUT_SEQ_LENGTH, num_classes=2, vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim,
                            filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)


    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # 全バッチ生成
        gen_data_loader.create_batches(Squad.in_train_data_file, Squad.out_train_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)
        gen_test_data_loader.create_batches(Squad.in_eval_data_file, Squad.out_eval_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)


        # log file
        log = open(os.path.join(RESULT_DIR, 'log.txt'), 'w')


        # positiveデータを使ってgeneratorをpre-train
        print('\nStart Pre-Training')
        print('Generator')
        log.write('----- pre Training Generator -----\n')

        for epoch in range(PRE_EPOCH_NUM):
            losses = []
            gen_data_loader.reset_pointer()

            for it in range(gen_data_loader.num_batch):
                X_batch, y_batch, w_batch = gen_data_loader.next_batch()

                # feed_dict 構築
                feed_dict = {}
                feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
                feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)})
                feed_dict.update({labels[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
                feed_dict.update({weights[i]:w_batch[i] for i in range(OUT_SEQ_LENGTH)})

                l, _, predictions = sess.run([loss_op, opt_op, predictions_op], feed_dict=feed_dict)

                losses.append(l)
            loss = np.mean(losses)

            print('epoch ', epoch, ' loss ', loss)
            log.write('epoch ' + str(epoch) + ' loss ' + str(loss) + '\n')

        # discriminatorをpre-train
        # negativeデータを作成
        print('Discriminator')
        gen_data_loader.reset_pointer()
        generated_samples = []
        for it in range(int(gen_data_loader.num_samples / BATCH_SIZE)):
            X_batch, y_batch, w_batch = gen_data_loader.next_batch()
            feed_dict = {}
            feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
            feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)})
            feed_dict.update({labels[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
            feed_dict.update({weights[i]:w_batch[i] for i in range(OUT_SEQ_LENGTH)})

            generated = sess.run(generate_op, feed_dict=feed_dict) # list (1, 20, 5000)
            generated_samples.append(generated)

        negative_file_path = os.path.join(OUTPUT_DIR, 'generated.txt')
        save_generated(generated_samples, negative_file_path)
        log.write('----- pre Training Discriminator -----\n')

        positive_file_path = Squad.out_train_data_file
        dis_data_loader.load_train_data(positive_file_path, negative_file_path)
        # 3epochを50回やるらしい
        # なんで????
        for ti in range(50):
            for epoch in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, dis_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
                print('epoch ', epoch, ' loss ', dis_loss)
                log.write('epoch ' + str(epoch) + ' loss ' + str(dis_loss) + '\n')


        # positiveデータを使ってgeneratorをpre-train
        print('\nStart Adversarial Training.')
        log.write('----- Adversarial Training -----\n')
        for total_batch in range(TOTAL_BATCH):
            # 1batchに対する操作
            for it in range(1):
                # sampleの作成
                generated_samples = []
                for it in range(int(gen_data_loader.num_samples / BATCH_SIZE)):
                    X_batch, y_batch, w_batch = gen_data_loader.next_batch()
                    feed_dict = {}
                    feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
                    feed_dict.update({decoder_inputs[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)})
                    feed_dict.update({labels[i]:y_batch[i] for i in range(OUT_SEQ_LENGTH)}) # 正解データ
                    feed_dict.update({weights[i]:w_batch[i] for i in range(OUT_SEQ_LENGTH)})

                    generated = sess.run(generate_op, feed_dict=feed_dict) # list (1, 20, 5000)
                    generated_samples.append(generated)




if __name__ == '__main__':
    main()
