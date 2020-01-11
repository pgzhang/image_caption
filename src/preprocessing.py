# -*- coding:utf-8 -*-
# @Author:pgzhang

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from numpy import array
import numpy as np
import os
from PIL import Image
import utils
from parameters import *

def img_preprocess(img_path:str, img_size:tuple=INPUT_IMG_SIZE) ->np.array:
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w_h_tuple = (img_size[1], img_size[0])

    if img.size != w_h_tuple:
        img = img.resize(w_h_tuple,Image.NEAREST)

    img = np.asarray(img, dtype=K.floatx())
    #img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2]))

    img = img[...,::-1]
    mean = [103.939, 116.779, 123.68]
    img[...,0] -= mean[0]
    img[...,1] -= mean[1]
    img[...,2] -= mean[2]

    return img

def create_tokenizer(imgFile:str,captFile:str) -> Tokenizer:
    img_id = utils.load_dataset(imgFile)
    caption_dict =  utils.load_captions(captFile,img_id)
    texts_list = utils.dict_to_list(caption_dict)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_list)
    return tokenizer

def create_input_one_img(tokenizer:Tokenizer, img:np.array,max_len:int,
                         capt_list:list, vocab_size:int) -> tuple:
    img_input, seq_input, output = list(), list(), list()
    for capt in capt_list:
        seq = tokenizer.texts_to_sequences([capt])[0]
        for i in range(len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq],maxlen=max_len)[0]
            out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
            img_input.append(img)
            seq_input.append(in_seq)
            output.append(out_seq)

    return array(img_input),array(seq_input), array(output)

def data_generator(tokenizer:Tokenizer, max_len:int,
                   captions:dict, vocab_size:int) -> list:
    #while 1:
    for img_id, capt_list in captions.items():
        img_path = os.path.join(DATA_DIR, IMG_DIR, img_id+'.jpg')
        img = img_preprocess(img_path)
        img_input, seq_input, output = create_input_one_img(
            tokenizer, img, max_len,capt_list, vocab_size)

        yield [[img_input,seq_input],output]






