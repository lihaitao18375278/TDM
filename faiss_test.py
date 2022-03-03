import argparse
import sys
sys.path.append("./")
import faiss
import logging
import os
import numpy as np
import pickle as pkl
import pandas as pd
import tensorflow as tf
from timeit import default_timer as timer
query_dev_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query.memmap'
query_dev_offset = '/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/dev-qid2offset.pickle'
qrels_dev_dir = '/home/zhanjingtao/datasets/msmarco-passage/qrels.dev.small.tsv'

pass_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/passages.memmap' 
pass_offset = "/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/pid2offset.pickle"
dev_query_id_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query-id.memmap'
def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

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
    
def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        ids = ids.astype(np.int64)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index

def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
  

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors

def test():
    dev_query_embedding = np.memmap(query_dev_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768) 
    pass_embedding = np.memmap(pass_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    print(pass_embedding.shape)
    dev_query2offset_dict = load_object(query_dev_offset)
    passage2offset_dict = load_object(pass_offset)
    # print(list(passage2offset_dict.keys()))
    data_dev = pd.read_csv(qrels_dev_dir, header=None, names=["qid","0","pid","label"], sep='\t').drop(columns=['0','label'])
    data_dev.to_csv("/home/lihaitao/test/faiss_test/dev_qrels.csv",index=False,header=False)
    # pid_list = data_dev.pid.drop_duplicates().to_list() 
    pid_list = list(passage2offset_dict.keys())
    print(type(pid_list))
    dev_qid_list = data_dev.qid.drop_duplicates().to_list() 
    pid_embeddings = pass_embedding 
    # pid_embeddings = get_embeddings(pid_list, passage2offset_dict, pass_embedding) #1000,768
    dev_qid_embeddings = get_embeddings(dev_qid_list, dev_query2offset_dict, dev_query_embedding)
    pid_list = np.array(pid_list)
    dev_qid_list = np.array(dev_qid_list)
    pid_embeddings = np.array(pid_embeddings)
    dev_qid_embeddings = np.array(dev_qid_embeddings)
    
    index = construct_flatindex_from_embeddings(pid_embeddings, pid_list)
 
    
    

    faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, dev_qid_embeddings, 10, batch=None)
    print(nearest_neighbors)

    with open('/home/lihaitao/test/faiss_test/dev_rank.csv', 'w') as outputfile:
        for qid, neighbors in zip(dev_qid_list, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid},{pid},{idx+1}\n")
    print("END")
                
                

if __name__ == '__main__':
    test()
