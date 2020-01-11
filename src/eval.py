
# -*- coding:utf-8 -*-
# @Author: pgzhang

from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from parameters import *
import os
from pickle import load
import pickle
from preprocessing import img_preprocess, create_tokenizer
import utils
def id2word(integer:int, tokenizer:Tokenizer) ->str:
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model,tokenizer, img,max_length):
    in_text = 'BeginSeq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq],maxlen=max_length)
        #print (len(seq))
        output = model.predict([img,seq],verbose=0)
        integer = np.argmax(output)
        word = id2word(integer,tokenizer)

        if word is None:
            break
        in_text += ' '+word
        if word == 'EndSeq':
            break
    return in_text[9:-6]

def evaluate_model(model, captions,img_path,tokenizer,max_length):
    img = img_preprocess(img_path)
    img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    truth, predicted = list(), list()

    yhat = generate_caption(model,tokenizer,img, max_length)
    img_name = img_path.split('\\')[-1]
    gt = [caption.split() for caption in captions[img_name.split('.')[0]]]

    truth.append(gt)
    predicted.append(yhat.split())

    bleu1 = corpus_bleu(truth,predicted,weights=(1.0, 0,0,0))
    bleu2 = corpus_bleu(truth,predicted,weights=(0.5, 0.5,0,0))
    bleu3 = corpus_bleu(truth,predicted,weights=(0.3, 0.3,0.3,0))
    bleu4 = corpus_bleu(truth,predicted,weights=(0.25, 0.25,0.25,0.25))
    return bleu1,bleu2,bleu3,bleu4

if __name__ == '__main__':
    img_path = os.path.join(DATA_DIR,IMG_DIR,'2513260012_03d33305cf.jpg')
    modelpath = os.path.join(MODEL_DIR,'0.h5')
    model = load_model(modelpath)
    img = img_preprocess(img_path)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    imgFile = os.path.join(DATA_DIR, 'Flickr_8k.trainImages.txt')

    train = utils.load_dataset(imgFile)
    captFile = os.path.join(DATA_DIR, 'descriptions.txt')
    train_capt = utils.load_captions(captFile,train)
    token_path = os.path.join(DATA_DIR,'tokenizer.pkl')
    if os.path.exists(token_path):
        tokenizer = load(open(token_path,'rb'))
    else:
        tokenizer = create_tokenizer(imgFile, captFile)
        with open(token_path,'wb') as file:
            pickle.dump(tokenizer,file)
    max_length = utils.get_max_length(train_capt)
    result = evaluate_model(model,train_capt,img_path,tokenizer,max_length)
    print(result)
    texts = generate_caption(model,tokenizer,img,max_length)
    image = Image.open(img_path)
    plt.imshow(image)
    plt.xlabel(texts,fontsize=15,color='red')
    plt.show()





