# -*- coding:utf-8 -*-
# @Author:pgzhang

def load_file(filename:str) :
    with open(filename,'r') as f:
        file = f.read()
    return file


def dict_to_list(descriptions:dict) -> list:
    desc = list()
    for key in descriptions.keys():
        [desc.append(d) for d in descriptions[key] ]

    return desc


def get_max_length(captions:dict) -> int:
    lines = dict_to_list(captions)
    return max(len(line.split())for line in lines)


def get_vocab_size(captFile:str, dataset:list) -> int:
    words_list = ['BegainSeq','EndSeq']
    file = load_file(captFile)
    for line in file.split('\n'):
        if len(line) < 1:
            continue
        words_list += line.split()[1:]
    return len(set(words_list))


def load_dataset(imgFile:str) -> set:
    file = load_file(imgFile)
    dataset = list()
    for line in file.split('\n'):
        if len(line) < 1:
            continue
        img_id = line.split('.')[0]
        dataset.append(img_id)

    return set(dataset)


def load_captions(captFile:str, dataset:list) -> dict:
    file = load_file(captFile)
    descriptions = dict()
    for line in file.split('\n'):
        tokens = line.split()
        img_id, img_capt = tokens[0], tokens[1:]
        if img_id in dataset:
            if img_id not in descriptions:
                descriptions[img_id] = list()
            img_capt = 'BegainSeq ' + ' '.join(img_capt) + ' EndSeq'
            descriptions[img_id].append(img_capt)
    return descriptions













































