import os
import pandas as pd
import re

# input files
train_file = '../senta_data/new_train.tsv'
val_file = '../senta_data/new_dev.tsv'
test_file = '../senta_data/new_test.tsv'

# output files
seg_train_file = '../senta_data/train.seg.txt'
seg_val_file = '../senta_data/val.seg.txt'
seg_test_file = '../senta_data/test.seg.txt'

vocab_file = '../senta_data/vocab.txt'
category_file = '../senta_data/category.txt'

def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    df = pd.read_csv(input_file, sep='\t')
    df = df.dropna()
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        df['text_a'] = df['text_a'].apply(lambda x:x.lower())
        df['text_a'] = df['text_a'].apply(lambda x:x.split(' '))
        word_iters = df['text_a'].values
        labels = df['label'].values
        i = 0
        for word_iter in word_iters:
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (labels[i], word_content.strip(' '))
            f.write(out_line)
            i += 1

def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        line = line.strip('\n').split('\t')
        if len(line) < 2:
            continue
        label, content = line[0], line[1]
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    sorted_word_dict = sorted(word_dict.items(), key = lambda d:d[1], reverse=True)
    with open(output_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<UNK>\t10000000\n')
        for item in sorted_word_dict:
            f.write('%s\t%d\n' % (item[0], item[1]))

def generate_category_dict(input_file, category_file):
    category_list = ['0','1']
    with open(category_file, 'w', encoding='utf-8') as f:
        for category in category_list:
            line = category + '\n'
            f.write(line)
