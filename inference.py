from cProfile import label
from operator import index
import os
from signal import pthread_kill
from sqlite3 import Timestamp
from tkinter import S
from deep_network import NeuralNet
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
from construct_tree import TreeInitialize
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import tensorflow as tf
from sample_init import tree_generate_samples, Dataset, map_generate, sample_merge_multiprocess
from prediction import metrics_count



query_dev_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query.memmap'
query_dev_offset = '/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/dev-qid2offset.pickle'
qrels_dev_dir = '/home/zhanjingtao/datasets/msmarco-passage/qrels.dev.small.tsv'


save_qrels_path = "/home/lihaitao/test/rank_qrels/dev_qrels_all.csv"
save_rank_path = "/home/lihaitao/test/rank_qrels/dev_rank_all.csv"
tree_path = "/home/lihaitao/test/tree/tree_6980.pickle"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD = MODEL_DIR + '/models_checkpoints/network_model.ckpt-1'

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def load_train(path_data):
    tdata = pd.DataFrame(columns=['qid', 'pid', 'node', 'is_leaf','label'])
    for i in os.listdir(path_data) :
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            data_train = pd.read_csv(file_data, header=None, names=["qid","pid","node","is_leaf", "label"])
            tdata = pd.concat([tdata,data_train],axis=0,ignore_index=True)
    return tdata

def get_embeddings(id_list, offest, embeddings, use_gpu='Ture'):
    device = '/gpu:1' if use_gpu else '/cpu:0'
    item_embeddings = []
    # id_line = []
    with tf.device(device):
        length = len(id_list)
        for i in range(length):        
            id_line = offest[id_list[i]]
            item_embeddings.append(list(embeddings[id_line]))
        # res = item_embeddings.numpy()
    return item_embeddings
    


def inference():
    dev_query_embedding = np.memmap(query_dev_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    dev_query2offset_dict = load_object(query_dev_offset)
    tree = load_object(tree_path)
    data_dev = pd.read_csv(save_qrels_path, header=None, names=["qid","0","pid","label"], sep='\t')
    dev_qid_list = data_dev.qid.drop_duplicates().to_list() 
    dev_qid_embeddings = get_embeddings(dev_qid_list, dev_query2offset_dict, dev_query_embedding)
    ddata = pd.DataFrame(dev_qid_list,columns=['qid']) 
    dev_data = Dataset(ddata,100)   
    
    # 加载模型
    model = NeuralNet()
    check_point = tf.train.Checkpoint(myModel=model)
    check_point.restore(MODEL_LOAD)
    
    save = metrics_count(dev_data, dev_qid_embeddings, tree.root, 10, model)
    save.to_csv(save_rank_path,index=False,header=False)
    print("END")

if __name__ == '__main__':
    inference()
