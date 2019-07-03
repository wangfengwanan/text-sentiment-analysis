import tensorflow as tf
import os
import sys
import numpy as np
import math
from sklearn.metrics import roc_auc_score

tf.logging.set_verbosity(tf.logging.INFO)

def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size = 120,
        num_timesteps = 100,
        num_fc_nodes = 64,
        batch_size = 128,
        clip_lstm_grads = 5.0,
        learning_rate = 0.001,
        num_word_threshold = 10,
    )


train_file = '../senta_data/train.seg.txt'
val_file = '../senta_data/val.seg.txt'
test_file = '../senta_data/test.seg.txt'
vocab_file = '../senta_data/vocab.txt'
category_file = '../senta_data/category.txt'
output_folder = './run_text_rnn'
model_path = './model/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)
    
    def _read_dict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx
    
    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)
    
    @property
    def unk(self):
        return self._unk
    
    def size(self):
        return len(self._word_to_id)
    
    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) \
                    for cur_word in sentence.split()]
        return word_ids\

class CategoryDict:
    def __init__(self, filename):
        self._category_to_id = {}
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx
    
    def size(self):
        return len(self._category_to_id)
        
    def category_to_id(self, category):
        if not category in self._category_to_id:
            print("%s is not in our category list" % category)
        return self._category_to_id[category]

class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        # matrix
        self._inputs = []
        # vector
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)
    
    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            c = 0
            for line in lines:
                c += 1
                line = line.strip('\n').split('\t')
                if len(line) < 2:
                    continue
                label, content = line[0], line[1]
                id_label = self._category_vocab.category_to_id(label)
                id_words = self._vocab.sentence_to_id(content)
                id_words = id_words[0: self._num_timesteps]
                padding_num = self._num_timesteps - len(id_words)
                id_words = id_words + [
                    self._vocab.unk for i in range(padding_num)]
                self._inputs.append(id_words)
                self._outputs.append(id_label)
            self._inputs = np.asarray(self._inputs, dtype = np.int32)
            self._outputs = np.asarray(self._outputs, dtype = np.int32)
            self._random_shuffle()
    
    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]
    
    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            print("batch_size: %d is too large" % batch_size)
        
        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

def conv_wrapper(inputs, dilation_rate, name, output_channel=256, kernel_size=(1,3), activation=tf.nn.relu):
    conv = tf.layers.conv2d(inputs=inputs,
                            filters=output_channel,
                            kernel_size=kernel_size,
                            dilation_rate=dilation_rate,
                            padding='same',
                            activation=activation,
                            name=name)
    return conv

def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    inputs = tf.placeholder(tf.int32, (None, num_timesteps), name='inputs')
    outputs = tf.placeholder(tf.int32, (None, ), name='outputs')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    global_step = tf.Variable(
        tf.zeros([], tf.int64), name = 'global_step', trainable=False)
    
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer = embedding_initializer, reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('embedding', 
                                     shape=[vocab_size, hps.num_embedding_size], 
                                     dtype=tf.float32)
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    
    model_inputs = tf.expand_dims(embed_inputs, 1)
    with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
        conv1_1 = conv_wrapper(model_inputs, 1, 'conv1_1', kernel_size=(1,1))
        conv1_2 = conv_wrapper(model_inputs, 1, 'conv1_2', kernel_size=(1,2))
        conv1_3 = conv_wrapper(model_inputs, 1, 'conv1_3', kernel_size=(1,3))
        conv1_4 = conv_wrapper(model_inputs, 1, 'conv1_4', kernel_size=(1,5))
        conv1_5 = conv_wrapper(model_inputs, 1, 'conv1_5', kernel_size=(1,7))
        conv1 = tf.concat([conv1_1, conv1_2, conv1_3, conv1_4, conv1_5], axis=-1)
        conv2_1 = conv_wrapper(conv1, 1, 'conv2_1')
        conv2_2 = conv_wrapper(conv2_1, 1, 'conv2_2')
        conv2_3 = conv_wrapper(conv2_2, 2, 'conv2_3')
        conv3_1 = conv_wrapper(conv2_3, 1, 'conv3_1')
        conv3_2 = conv_wrapper(conv3_1, 1, 'conv3_2')
        conv3_3 = conv_wrapper(conv3_2, 2, 'conv3_3')
        conv4_1 = conv_wrapper(conv3_3, 1, 'conv4_1')
        conv4_2 = conv_wrapper(conv4_1, 1, 'conv4_2')
        conv4_3 = conv_wrapper(conv4_2, 2, 'conv4_3')
    last = tf.concat([conv2_3, conv3_3, conv4_3], axis=3)
    with tf.variable_scope('flatten', reuse=tf.AUTO_REUSE):
        flatten = tf.layers.flatten(last)
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer = fc_init, reuse=tf.AUTO_REUSE):
        fc1 = tf.layers.dense(flatten, 
                              hps.num_fc_nodes,
                              activation = tf.nn.relu,
                              name = 'fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name = 'fc2')
    
    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = outputs)
        loss = tf.reduce_mean(softmax_loss)
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1, 
                           output_type = tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step = global_step)
    
    return ((inputs, outputs, keep_prob, batch_size),
            (loss, y_pred, accuracy),
            (train_op, global_step))

if __name__ == '__main__':
    hps = get_default_params()
    vocab = Vocab(vocab_file, hps.num_word_threshold)
    vocab_size = vocab.size()
    category_vocab = CategoryDict(category_file)
    num_classes = category_vocab.size()
    train_dataset = TextDataSet(
        train_file, vocab, category_vocab, hps.num_timesteps) 
    val_dataset = TextDataSet(
        val_file, vocab, category_vocab, hps.num_timesteps) 
    test_dataset = TextDataSet(
        test_file, vocab, category_vocab, hps.num_timesteps)

    placeholders, metrics, others = create_model(
        hps, vocab_size, num_classes)

    inputs, outputs, keep_prob, batch_size = placeholders
    loss, y_pred, accuracy = metrics
    train_op, global_step = others
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    train_keep_prob_value = 0.6
    test_keep_prob_value = 1.0
    max_acc = 0.910

    num_train_steps = 5000

    with tf.Session() as sess:
        sess.run(init_op)
        val_data, val_labels = val_dataset.next_batch(1200)
        test_data, test_labels = test_dataset.next_batch(1200)
        for i in range(num_train_steps):
            batch_inputs, batch_labels = train_dataset.next_batch(
                    hps.batch_size)
            outputs_val = sess.run([loss, accuracy, train_op, global_step, outputs, y_pred],
                                       feed_dict = {
                                           inputs: batch_inputs,
                                           outputs: batch_labels,
                                           keep_prob: train_keep_prob_value,
                                           batch_size: hps.batch_size,
                                       })
            loss_val, accuracy_val, _, global_step_val, outputs_val, y_pred_val = outputs_val
            train_auc = roc_auc_score(outputs_val, y_pred_val)
            if global_step_val % 20 == 0:
                tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f, auc: %3.3f"
                                    % (global_step_val, loss_val, accuracy_val, train_auc))
            outputs_ = sess.run([loss, accuracy, global_step, outputs, y_pred],
                                       feed_dict = {
                                           inputs: val_data,
                                           outputs: val_labels,
                                           keep_prob: test_keep_prob_value,
                                           batch_size: len(val_data),
                                       })
            loss_, accuracy_, global_step_, outputs_, y_pred_ = outputs_
            val_auc = roc_auc_score(outputs_, y_pred_)
            outputs_1 = sess.run([loss, accuracy, global_step, outputs, y_pred],
                                       feed_dict = {
                                           inputs: test_data,
                                           outputs: test_labels,
                                           keep_prob: test_keep_prob_value,
                                           batch_size: len(test_data),
                                       })
            loss_1, accuracy_1, global_step_1, outputs_1, y_pred_1 = outputs_1
            test_auc = roc_auc_score(outputs_1, y_pred_1)
            if accuracy_1 > max_acc:
                tf.logging.info("[Val] Step: %5d, loss: %3.3f, accuracy: %3.3f, auc: %3.3f"
                                    % (global_step_, loss_, accuracy_, val_auc))
                tf.logging.info("[Test] Step: %5d, loss: %3.3f, accuracy: %3.3f, auc: %3.3f"
                                    % (global_step_1, loss_1, accuracy_1, test_auc))
                saver.save(sess,model_path)
                max_acc = accuracy_1                
