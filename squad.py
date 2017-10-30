# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pickle
import re
from collections import Counter


class Squad(object):
	def __init__(self, raw_data, vocab_size, in_seq_length, out_seq_length,max_data_num):
		self.vocab_size = vocab_size
		self.in_seq_length = in_seq_length
		self.out_seq_length = out_seq_length
		self.max_data_num = max_data_num
		self.in_train_data_file = ''
		self.out_train_data_file = ''
		self.in_eval_data_file = ''
		self.out_eval_data_file = ''

		self.in_raw_text = []
		self.out_raw_text = []
		# contextを取り出す.
		# ここに整形した文章のうち, context, questionともにそれぞれのseq_length以下の長さの文章を追加していく
		ds = raw_data['data']
		for d in ds:
			ps = d['paragraphs']
			for p in ps:
				context = p['context']
				qas = p['qas']
				c = self.clean(context)
				if c and len(c) < in_seq_length:
					for qa in qas:
						q = self.clean(qa['question'])
						if q and len(q) < out_seq_length:
							self.in_raw_text.append(c)
							self.out_raw_text.append(q)

		self.in_raw_text = self.in_raw_text[:self.max_data_num]
		self.out_raw_text = self.out_raw_text[:self.max_data_num]

		self.data_num = len(self.in_raw_text)
		print('data num :', self.data_num)

		# vocaburary 構築
		in_words = [] # inputの単語リスト
		for line in self.in_raw_text:
			in_words.extend(line)
			in_words.append(' ')

		out_words = [] # 全体の単語リスト
		for line in self.out_raw_text:
			out_words.extend(line)
			out_words.append(' ')

		in_counter = Counter(in_words)
		self.in_word_freq = {word: cnt for word, cnt in in_counter.most_common(vocab_size-3)} # start, eosの分
		self.in_vocab = ['_START'] + ['<EOS>'] + sorted(list(self.in_word_freq)) + ['***']
		self.in_word2idx = {word:i for i, word in enumerate(self.in_vocab)}
		self.in_idx2word = {i:word for i, word in enumerate(self.in_vocab)}

		out_counter = Counter(out_words)
		self.out_word_freq = {word: cnt for word, cnt in out_counter.most_common(vocab_size-3)} # start, eosの分
		self.out_vocab = ['_START'] + ['<EOS>'] + sorted(list(self.out_word_freq)) + ['***']
		self.out_word2idx = {word:i for i, word in enumerate(self.out_vocab)}
		self.out_idx2word = {i:word for i, word in enumerate(self.out_vocab)}

		print('input word num :', len(self.in_vocab))
		print('output word num :', len(self.out_vocab))

		# データ構築
		self.in_data = np.ones((self.data_num, self.in_seq_length), np.int32) * (self.vocab_size-1)
		for i in range(self.data_num):
			for j in range(len(self.in_raw_text[i])):
				w = self.in_raw_text[i][j]
				if w in self.in_vocab:
					self.in_data[i][j] = self.in_word2idx[w]
				else:
				  pass
			self.in_data[i][len(self.in_raw_text[i])] = 1

		self.out_data = np.ones((self.data_num, self.out_seq_length), np.int32) * (self.vocab_size-1)
		for i in range(self.data_num):
			for j in range(len(self.out_raw_text[i])):
				w = self.out_raw_text[i][j]
				if w in self.out_vocab:
					self.out_data[i][j] = self.out_word2idx[w]
				else:
				  pass
			self.out_data[i][len(self.out_raw_text[i])] = 1


		perm = np.random.permutation(self.data_num)
		self.test_idx = perm[:int(self.data_num/5)]
		self.train_idx = perm[int(self.data_num/5):]


	def clean(self, string):
		# string: 1question
		# 単語のリストを返す
		#string = re.split('.?',string)[0]+' ?'
		string = ' '.join(string.split())
		string = string.split()

		return string

	def get_train_data(self, batch_size):
		idx = np.random.choice(self.train_idx, batch_size, replace=False)
		return self.data[idx]

	def get_test_data(self, batch_size):
		idx = np.random.choice(self.test_idx, batch_size, replace=False)
		return self.data[idx]

	def result2text(self, filepath, outputpath):
		# 結果ファイルを受け取り, idx2wordを使って読めるテキストファイルにする
		with open(filepath, 'rb') as f:
			lines = f.readlines()

		with open(outputpath, 'w') as f:
			for line in lines:
				idxs = line.split()
				for idx in idxs:
					try:
						word = self.idx2word[int(idx)]
					except KeyError:
						word = '***'
					if type(word) == type(u'\xe9'):
						print(word)
						word = word.encode('utf-8')
					f.write(str(word))
					f.write(' ')
				f.write('\n')
		print('save ', outputpath)

	def return_word(self, index, error_word='***', output=True):
		if output:
			try:
				word = self.out_idx2word[int(index)]
			except KeyError:
				word = error_word
		else:
			try:
				word = self.in_idx2word[int(index)]
			except KeyError:
				word = error_word
		if type(word) != type(str('ab')):
			word = word.encode('utf-8')
		return word


	def save_textfile(self, in_outputpath, out_outputpath, limit=1.0, Train=True):
		with open(in_outputpath, 'w') as f:
			for i, line in enumerate(self.in_data):
				if i > limit*self.data_num:
					break

				buffer = ' '.join([str(x) for x in line]) + '\n'
				f.write(buffer)
		print('save ', in_outputpath)

		with open(out_outputpath, 'w') as f:
			for i, line in enumerate(self.out_data):
				if i > limit*self.data_num:
					break
				buffer = ' '.join([str(x) for x in line]) + '\n'
				f.write(buffer)
		print('save ', out_outputpath)
		if Train:
			self.in_train_data_file = in_outputpath
			self.out_train_data_file = out_outputpath
		else:
			self.in_eval_data_file = in_outputpath
			self.out_eval_data_file = out_outputpath



if __name__ == '__main__':
	squad_data_path = os.path.join('/home/nakanishi/data/squad/train-v1.1.json')
	with open(squad_data_path) as f:
		squad_data = json.load(f)
	Loader = Squad(squad_data, vocab_size=5000, seq_length=20, max_data_num=10000)


	output_dir = '/home/nakanishi/my_works/gan/SeqGAN/mydata'
	Loader.save_textfile(os.path.join(output_dir, 'my_real_data.txt'), limit=0.8)
	Loader.save_textfile(os.path.join(output_dir, 'my_eval_data.txt'), limit=0.2)


	with open(os.path.join(output_dir,'squad.dat'), 'wb') as f:
		pickle.dump(Loader, f)
	print('save ', (os.path.join(output_dir,'squad.dat')))
