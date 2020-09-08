#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:11:18 2020

@author: ahmed
"""

"""
corpus_helper
1. prepare corpus to .pkl
2. prepare vocab and embedidng for joint train corpus and transfer train corpus
3. tune embedding
"""
import pickle
import numpy as np
from params import *
from collections import Counter
import itertools
import params
import re
import argparse
import sys
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from preprocess import preprocessText


def shuffleData(sent_list):
    np.random.seed(31)
    shuffled_indices = np.random.permutation(np.arange(len(sent_list)))
    shuffled_sent_list = np.array(sent_list)[shuffled_indices]
    return shuffled_sent_list

    
def convertDataToPickle(label, folder,dump, pickle_fn, is_shuffle=False):
    sents = []
    
    sents = read_label_data(label,folder)
    print(label + ": {}".format(len(sents)))
    # shuffle sents
    if (is_shuffle):
        sents = shuffleData(sents)
    print("fn size: {}".format(len(sents)))
    pickle_fn = os.path.join(dump,pickle_fn)
    with open(pickle_fn, "wb") as handle:
        pickle.dump(sents, handle)
    print("done shuffling and transforming txt to pickle...")


# ************** Build Vocab & Init Embedding ****************

def buildVocab(sentences, min_cnt=2):
    # vocab: dict, vocab_inv: list
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocab_inv = extra_words + [x[0] for x in word_counts.most_common() if x[1] >= min_cnt]
    print("Extra words:", extra_words)
    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    print("<EOS> in vocab:", vocab["<EOS>"])
    return [vocab, vocab_inv]


def buildEmbed(vocab_list, embed_fn, word_limit=250000):
    # sanity check: vocab_list should be a list instead of dictionary
    if (type(vocab_list) != type([1,2])):
        print("vocab type: {}, not a list!".format(type(vocab_list)))
        sys.exit(0)
    
    print("load pretrained word embeddings...")
    f = open(embed_fn, "r")
    header = f.readline()
    vocab_size, dim = np.array(header.strip().split(), "int")

    # read pretrained embedding
    all_vocab = []
    all_embed = []
    i = 0
    while (i < word_limit):
        line = f.readline()
        seq = line.strip().split()
        if (seq == []):
            break
        all_vocab.append(seq[0])
        vec = list(np.array(seq[1:], "float"))
        all_embed.append(vec[:])
        i += 1
    f.close()
    print("pretrain vocab:", all_vocab[:10])

    # adapt to dataset
    print("dataset vocabulary:", len(vocab_list))
    init_embed = []
    unknown_word = []
    for w in vocab_list:
        try:
            ind = all_vocab.index(w)
            vec = all_embed[ind]
        except:
            vec = (np.random.rand(dim) - 0.5) * 2
            unknown_word.append(w)
        init_embed.append(vec[:])
    print("unknown word:", len(unknown_word), unknown_word[:10])
    init_embed = np.array(init_embed)
    print("vocab size: {}, embedding size: {}".format(len(vocab_list), np.shape(init_embed)))
    return init_embed

def read_file(file):
    sent_list = []
    with open(file,"rb") as datafile:
        all_data = str(datafile.read())

        sents = all_data.split(".")
        
        for sent in sents:
            sent  = re.sub("b'", " ", sent)
            sent  = re.sub("[0-9]+", "_num_", sent)
            sent  = re.sub("[^_a-zA-Z]+", " ", sent)
            sent = sent.lower()
            sent_split = sent.split()
            sent_list.append(sent_split)
        return sent_list

def read_label_data(label, folder):
    data_path = os.path.join(folder,label)
    entries = os.listdir(data_path)
    string_data = []
    for entry in entries:
        string_data = string_data + read_file(os.path.join(data_path,entry))
    return string_data


def buildVocabEmbed(dump_folder, embed_fn, raw_vec_path, save_folder):
    all_sents = []
    pkl_list = [pkl for pkl in os.listdir(dump_folder) if os.path.isfile(pkl) and pkl.endswith(".pkl")]
    for pkl in pkl_list:
        with open(pkl, "rb") as handle:
            sents = pickle.load(handle)
        all_sents += list(sents)
   
    vocab, vocab_inv = buildVocab(all_sents)
    with open(save_folder+"vocab.pkl", "wb") as handle:
        pickle.dump(vocab, handle)
    with open(save_folder+"vocab_inv.pkl", "wb") as handle:
        pickle.dump(vocab_inv, handle)
    print("Joint vocab size:", len(vocab))
    print("Example vocab words:", vocab_inv[:10])

    
    # joint word embedding for train corpus
    init_embed = buildEmbed(vocab_inv, embed_fn, 40000)
    # save raw_dataset_vec.txt for tuning
    f =  open(raw_vec_path, "w")
    vocab_size = len(vocab)
    f.write(str(vocab_size)+" "+str(VEC_DIM)+"\n")
    for (word, vec) in zip(vocab_inv, init_embed):
        f.write(word+" "+" ".join([str(val) for val in vec])+"\n")
    f.close()
    print("saving raw vecs for tuning...")
    del all_sents, vocab, vocab_inv



    
def tuneEmbed(train_corpus, total_lines, raw_vec_path, tune_vec_path):
    sentences = LineSentence(train_corpus)
    model = Word2Vec(sentences, size=VEC_DIM, window=6, iter=20, workers=10, min_count=1)
    model.intersect_word2vec_format(raw_vec_path, lockf=1.0, binary=False)
    # measure runing time
    start = time.time()
    model.train(sentences, total_examples=total_lines, epochs=20)
    end = time.time()
    print("done retraining using time {} s.".format(end-start))
    #word_vectors = model.mv
    model.wv.save_word2vec_format(tune_vec_path)
    print("done saving tuned word embeddings...")
    

def saveTuneEmbed(save_folder, tune_vec_path):
    with open(save_folder+"vocab_inv.pkl", "rb") as handle:
        vocab_list = pickle.load(handle)
    init_embed = buildEmbed(vocab_list, tune_vec_path, word_limit=250000)
    with open(save_folder+"init_embed.pkl", "wb") as handle:
        pickle.dump(init_embed, handle)
    print("saving init_embed shape: {}".format(np.shape(init_embed)))
    del vocab_list, init_embed
  

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_folder", type=str, default="../../C50/C50train") #gyafc_family, yelp
    parser.add_argument("--save_folder", type=str, default="../C50_save") #gyafc_family, yelp
    parser.add_argument("--tokenize", default=False, action="store_true")
    parser.add_argument("--vec_dim", type=int, default=300)
    parser.add_argument("--embed_fn", type=str, default="../data/glove.6B.300d.txt")
    
    parser.add_argument("--train", type=bool, default=True)
    args = parser.parse_args()
    save_folder = args.save_folder
    tok_flag = args.tokenize
    VEC_DIM = args.vec_dim
    embed_fn = args.embed_fn
    data_folder = args.data_folder
#    data_folder = "../"+str(save_folder)+"/data/"
    
    dump_folder = "../"+str(save_folder)+"/dump/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)
    
 
    
    current_labels = os.path.join(dump_folder,"all_labels.txt")
    labels = os.listdir(data_folder)
    with open(current_labels,"w") as targes:
        for label in labels:
            is_shuffle = args.train
            print(label)
            # convert .txt to .pkl
            convertDataToPickle(label,data_folder,dump_folder, label+".pkl", is_shuffle=is_shuffle)
            targes.write(label)
    print("build vocabulary and embedding...")
    raw_vec_path = dump_folder + "raw_vec.txt"
    
    buildVocabEmbed(dump_folder, embed_fn, raw_vec_path, dump_folder)
       
    
