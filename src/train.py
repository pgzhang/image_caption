# -*- coding:utf-8 -*-
# @Author:pgzhang

import os
import utils
from parameters import *
from model import model
from pickle import load
from preprocessing import create_tokenizer, data_generator
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def train():
    imgFile = os.path.join(DATA_DIR,'Flickr_8k.trainImages.txt')

    train = utils.load_dataset(imgFile)
    captFile = os.path.join(DATA_DIR, 'descriptions.txt')
    train_capt = utils.load_captions(captFile,train)
    token_path = os.path.join(DATA_DIR,'tokenizer.pkl')
    if os.path.exists(token_path):
        tokenizer = load(open(token_path,'rb'))
    else:
        tokenizer = create_tokenizer(imgFile, captFile)
        with open(os.path.join(DATA_DIR,'tokenizer.pkl'),'wb') as file:
            pickle.dump(tokenizer,file)
    max_length = utils.get_max_length(train_capt)
    vocab_size= utils.get_vocab_size(captFile,train)

    capt_model = model(vocab_size,max_length)
    #capt_model.summary()
    epochs = 32
    batch_size = 32
    steps = len(train_capt) // batch_size

    for i in range(epochs):
        generator = data_generator(tokenizer, max_length,
                                   train_capt, vocab_size)
        capt_model.fit_generator(generator, epochs=1, steps_per_epoch=steps,
                            verbose=1)
        capt_model.save('../model/'+ str(i)+ '.h5')


if __name__ == '__main__':
    train()
    print('training finished')




