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

query_train_embedding_dir="/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/train.qembed.memmap"
query_train_offset="/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/train-qid2offset.pickle"
qrels_train_dir = '/home/zhanjingtao/datasets/msmarco-passage/qrels.train.tsv'

query_dev_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/dev-query.memmap'
query_dev_offset = '/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/dev-qid2offset.pickle'
qrels_dev_dir = '/home/zhanjingtao/datasets/msmarco-passage/qrels.dev.small.tsv'

query_test_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/test-query.memmap'
query_test_offset = '/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/test-qid2offset.pickle'
qrels_test_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/test-qrel.tsv'

pass_embedding_dir = '/home/zhanjingtao/public_code/DRhard/data/passage/evaluate/star/passages.memmap' 
pass_offset = "/home/zhanjingtao/public_code/DRhard/data/passage/preprocess/pid2offset.pickle"

save_qrels_path = "/home/lihaitao/test/rank_qrels/dev_qrels_all.csv"
save_rank_path = "/home/lihaitao/test/rank_qrels/dev_rank_all.csv"
train_dir = "/home/lihaitao/test/data/train_all"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = MODEL_DIR + '/models_checkpoints/network_model.ckpt'

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

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
    


def tdm():
    # data_train, data_dev, pid_list, train_qid_list, dev_qid_list, pid_embeddings, train_qid_embeddings, dev_qid_embeddings = data_process()
      #获取全部embedding
    train_query_embedding = np.memmap(query_train_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768) #(502939, 768)
    dev_query_embedding = np.memmap(query_dev_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768) #(502939, 768
    pass_embedding = np.memmap(pass_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    
    #获取全部offset
    train_query2offset_dict = load_object(query_train_offset)
    dev_query2offset_dict = load_object(query_dev_offset)
    test_query2offset_dict = load_object(query_test_offset)
    passage2offset_dict = load_object(pass_offset)
    
    #获取data_train data_dev data_test
    data_train = pd.read_csv(qrels_train_dir, header=None, names=["qid","0","pid","label"], sep='\t').drop(columns=['0','label'])
    data_dev = pd.read_csv(qrels_dev_dir, header=None, names=["qid","0","pid","label"], sep='\t').drop(columns=['0','label'])
    data_test = pd.read_csv(qrels_test_dir, header=None, names=["qid","0","pid","label"], sep='\t').drop(columns=['0','label'])
    
    # data_1 = data_train[:1000]   #先去10000个数据试试水
    # data_train = data_train[0:800]
    # data_dev = data_1[801:1000]
    data_train = data_train
    data_dev = data_dev
    
    
    data_dev.to_csv(save_qrels_path,index=False,header=False)
    #获取id_list
    pid_list = data_train.pid.drop_duplicates().to_list()  #用来建树的pid
    train_qid_list = data_train.qid.drop_duplicates().to_list()
    dev_qid_list = data_dev.qid.drop_duplicates().to_list()  
    
    #获取list对应的embedding
    pid_embeddings = get_embeddings(pid_list, passage2offset_dict, pass_embedding) #1000,768
    train_qid_embeddings = get_embeddings(train_qid_list, train_query2offset_dict, train_query_embedding)  #(923,768)
    # dev_qid_embeddings = get_embeddings(dev_qid_list, dev_query2offset_dict, dev_query_embedding) 
    dev_qid_embeddings = get_embeddings(dev_qid_list, dev_query2offset_dict, dev_query_embedding)
   
   
    #树初始化
    s = time.perf_counter()
    tree = TreeInitialize(pid_embeddings, pid_list)
    _ = tree.clustering_binary_tree()
    o = time.perf_counter()
    print('finish tree_initialize %d' % (o-s))
    
    pass_ids, pass_size = pid_list, len(pid_list) 
    model = NeuralNet()
    
    #超参数
    num_epoch = 1
    check_point = tf.train.Checkpoint(myModel=model)
    while num_epoch > 0:
        
        #生成树样本 把二叉树提出来用数组存储 提高访问效率
        node_list = tree._node_list(tree.root) #list是树每一行的节点
        print('node_list_len %d' % len(node_list))
   
        
        #采样修改 时间太久
        s = time.perf_counter()
        tree_samples = tree_generate_samples(pass_ids, tree.leaf_dict, node_list)
        o = time.perf_counter() #pid node is_leaf label
        print('finish tree_samples %d' % (o-s))
        print('tree_sample length %d' % len(tree_samples))   #1000个数据有116520个样本

        #优化数据结构 形成map
        s = time.perf_counter()
        tree_map = map_generate(tree_samples)
        o = time.perf_counter()
        print('finish build map %f' % (o-s))
        # print(tree_map[0])
        
        
        s = time.perf_counter()
        sample_merge_multiprocess(data_train , tree_map,'train',10,train_dir)   #生成训练样本
        # tdata = merge_samples(data_train, tree_samples) #qid pid node is_leaf label
        o = time.perf_counter()
        print('finish merge_samples %d' % (o-s))
        

        tdata = load_train(train_dir)
        ddata = pd.DataFrame(dev_qid_list,columns=['qid'])
        
        train_data = Dataset(tdata, 100, shuffle=True)
        
        dev_data = Dataset(ddata,100)
        
        use_gpu=True
        r=0.001 
        lr=0.001 
        b1=0.9
        b2=0.999
        eps=1e-08
        all_epoch=3
        check_epoch=2
        save_epoch=3
        device = '/gpu:0' if use_gpu else '/cpu:0'
        
        with tf.device(device):
            for epoch in range(all_epoch):
                s = time.perf_counter()
                iter_epoch = 1
                print("Start epoch %d" % epoch) 
                all_loss = 0
                k = 0
                for qid_tr, pid_tr, nodes_tr, is_leafs_tr, labels_tr in train_data:   #800个数据大概 934iteration
                    pid_tr_embeddings = get_embeddings(pid_tr, passage2offset_dict, pass_embedding) 
                    qid_tr_embeddings = get_embeddings(qid_tr, train_query2offset_dict, train_query_embedding)
                    with tf.GradientTape() as tape:
                        scores = model(qid_tr_embeddings, pid_tr_embeddings, 1)   #100 2
                        labels_tr = labels_tr.reshape(-1,1) 
                        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=labels_tr, logits=scores)
                        loss = tf.reduce_sum(loss)   #求和函数
                        all_loss += loss
                        print("Epoch {}, Iteration {}, loss {}".format(epoch, iter_epoch, loss))
                        trainable_params = model.trainable_variables  #model.variables
                        gradients = tape.gradient(loss,trainable_params)
                        # optimizer=tf.optimizers.Adam(learning_rate=lr,  epsilon=eps)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
                        optimizer.apply_gradients(zip(gradients, trainable_params))
                        iter_epoch += 1
                        if(iter_epoch > 500):
                            break
                
                check_point.save(MODEL_NAME)       
                o = time.perf_counter()
                print("End epoch %d" % (epoch)) 
                print("epoch time %f" % (o-s)) 
                print("Epoch {}, all_loss {}".format(epoch, all_loss/iter_epoch))
                save = metrics_count(dev_data, dev_qid_embeddings, tree.root, 10, model)
                save.to_csv('/home/lihaitao/test/rank_qrels/dev_rank_all_%s.csv'% (epoch),index=False,header=False)
        print("It's completed to train the network.")
        
        
        check_point.save(MODEL_NAME)
        # model.save('tdm.h5',save_format="tf")
    
        save = metrics_count(dev_data, dev_qid_embeddings, tree.root, 10, model)
        save.to_csv(save_rank_path,index=False,header=False)
        print("END")
        # metrics_count(dev_data, tree.root, 10, model)
        # mrr = compute_metrics(data_dev, save)
        # print("end mrr")
        # print(mrr)
        
        #学习一下embedding表示
        
        
        num_epoch -= 1


    

if __name__ == '__main__':
    tdm()
