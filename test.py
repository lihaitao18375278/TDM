import os
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
# from construct_tree import TreeInitialize
import pickle 

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def get_data():
    with open('/home/lihaitao/test/data/data.pkl', 'rb') as f:
        data_train = pickle.load(f)
        data_validate = pickle.load(f)
        data_test = pickle.load(f)
        cache = pickle.load(f)
        return data_train, data_validate,data_test,cache


# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/passages.memmap    #passage embedding(8841823,768)
# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query.memmap   #(6980,768)
# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query-id.memmap #6980
# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/lead-query.memmap  #(6837,768)
# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/test-query.memmap  #(43,768)
# /home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/train.qembed.memmap #((502939, 768))
# /home/zhanjingtao/public_code/DRhard/data/passage/preprocess/train-qid2offset.pickle #(502939)  qid:行号
# /home/zhanjingtao/datasets/msmarco-passage/qidpidtriples.train.full.tsv   (269919003,3)
# /home/zhanjingtao/datasets/msmarco-passage/2019qrels-pass.txt
# '/home/zhanjingtao/datasets/msmarco-passage/qrels.train.tsv'
# /home/zhanjingtao/public_code/DRhard/data/passage/preprocess/pid2offset.pickle
# /home/zhanjingtao/public_code/DRhard/data/passage/preprocess/dev-query.memmap


if __name__ == '__main__':#(8841823, 768)
    d = np.memmap("/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/dev-query.memmap",dtype=np.float32,mode="r")
    print(d.shape)
    print(d[0:10])
    # path="/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/train-qid2offset.pickle"
    # obj=load_object(path)
    # print(len(obj))
    # print(obj[114749])
    # path='/home/zhanjingtao/datasets/msmarco-passage/qrels.train.tsv'
    # file=pd.read_csv(path, header=None, names=["qid","0","pid","rel"], sep='\t')
    # print(file)
    # print(file.iloc[1,:].tolist())
    # qrels_pass_dir = '/home/zhanjingtao/datasets/msmarco-passage/2019qrels-pass.txt'  
    # q_id = []
    # file_q2p = open(qrels_pass_dir,'r')
    # for lines in file_q2p:
    #     content = lines.strip(' ')
    #     print(content)
    #     q_id.append(content)
    data_train, data_val, data_test, cache = get_data()
    print(data_train)
    
    #user_ID  timestamp  item_ID  behaviors