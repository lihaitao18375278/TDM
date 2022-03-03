from cProfile import label
from logging.handlers import NTEventLogHandler
from operator import index
import os
from signal import pthread_kill
from sqlite3 import Timestamp
from tkinter import S
from deep_network_2 import NeuralNet
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
from prediction_pair import metrics_count

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

save_qrels_path = "/home/lihaitao/test/rank_qrels/dev_qrels_6980.csv"
save_rank_path = "/home/lihaitao/test/rank_qrels/dev_rank_6980.csv"
train_dir = "/home/lihaitao/test/data/train_6980"
tree_path = "/home/lihaitao/test/tree/tree_6980.pickle"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = MODEL_DIR + '/models_checkpoints/network_model.ckpt'
MODEL_LOAD = MODEL_DIR + '/models_checkpoints/network_model.ckpt-1'
def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def load_train(path_data):
    tdata = pd.DataFrame(columns=['qid', 'pid', 'positive', 'negetive', 'p_isleaf', 'n_isleaf', 'label'])
    for i in os.listdir(path_data) :
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            data_train = pd.read_csv(file_data, header=None, names=['qid', 'pid', 'positive', 'negetive', 'p_isleaf', 'n_isleaf', 'label'])
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
    train_query_embedding = np.memmap(query_train_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768) #(502939, 768)
    dev_query_embedding = np.memmap(query_dev_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768) #(502939, 768
    pass_embedding = np.memmap(pass_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    #获取全部offset
    train_query2offset_dict = load_object(query_train_offset)
    dev_query2offset_dict = load_object(query_dev_offset)
    passage2offset_dict = load_object(pass_offset)
    
    #获取data_train data_dev data_test
    data_dev = pd.read_csv(save_qrels_path, header=None, names=["qid","0","pid","label"], sep=',')
    dev_qid_list = data_dev.qid.drop_duplicates().to_list()  
    
    #获取list对应的embedding
    dev_qid_embeddings = get_embeddings(dev_qid_list, dev_query2offset_dict, dev_query_embedding)
   
   
    #树初始化
    tree = load_object(tree_path)
    


    node_embeddings =[]
    for i in range(len(tree.node_dict)):
        node_embeddings.append(tree.node_dict[i].embedding)
   
    print("load tree finish")
    model = NeuralNet(node_embeddings)
    check_point = tf.train.Checkpoint(myModel=model)
    #超参数
    num_epoch = 1
    while num_epoch > 0:
        tdata = load_train(train_dir)
        labels_dict = dict(zip(['[0, 1]', '[1, 0]'], [[0, 1],[1, 0]]))
        tdata['label'] = tdata.label.apply(lambda x: labels_dict[x])
        ddata = pd.DataFrame(dev_qid_list,columns=['qid'])
        train_data = Dataset(tdata, 50)
        dev_data = Dataset(ddata,100)
        
        use_gpu=True
        r=0.001 
        lr=0.00001 
        b1=0.9
        b2=0.999
        eps=1e-08
        all_epoch=10
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
                for qid_tr, pid_tr, positive_tr, negetive_tr, p_isleafs_tr, n_isleafs_tr,labels_tr in train_data:   #800个数据大概 934iteration
                    # pid_tr_embeddings = get_embeddings(pid_tr, passage2offset_dict, pass_embedding) 

                    qid_tr_embeddings = get_embeddings(qid_tr, dev_query2offset_dict, dev_query_embedding)
                    with tf.GradientTape() as tape:
                        labels_tr2 =np.full((50,2),[0,1])
                        #正样本的分数
                        scores1 = model(qid_tr_embeddings, positive_tr, p_isleafs_tr, tree.node_dict, tree.leaf_dict, 1) 
                        #负样本的分数
                        scores2 = model(qid_tr_embeddings, negetive_tr, n_isleafs_tr, tree.node_dict, tree.leaf_dict, 1)     
                        scores = tf.concat([scores1,scores2],axis=0)
                        #标签都是[1,0]
    
                        labels_tr2[:,[0, -1]] = labels_tr[:,[-1, 0]]
                        
                        labels_tr = tf.convert_to_tensor(labels_tr, dtype=tf.float32)
                        labels_tr2 = tf.convert_to_tensor(labels_tr2, dtype=tf.float32)
                        labels_tr = tf.concat([labels_tr,labels_tr2],axis=0)
                        
                        # loss = -labels_tr * tf.math.log(pred) -(1-labels_tr) * tf.math.log(1-pred)
                        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels_tr)
                        
                       
                        
                        loss = tf.reduce_mean(loss)   
                        all_loss += loss
                        print("Epoch {}, Iteration {}, loss {}".format(epoch, iter_epoch, loss))
                        trainable_params = model.trainable_variables  #model.variables
                        gradients = tape.gradient(loss,trainable_params)
                        # optimizer=tf.optimizers.Adam(learning_rate=lr,  epsilon=eps)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
                        optimizer.apply_gradients(zip(gradients, trainable_params))
                        iter_epoch += 1
                        if(iter_epoch > 50):
                            break
          
                check_point.save(MODEL_NAME)       
                o = time.perf_counter()
                print("End epoch %d" % (epoch)) 
                print("epoch time %f" % (o-s)) 
                print("Epoch {}, all_loss {}".format(epoch, all_loss/iter_epoch))
                save = metrics_count(dev_data, dev_qid_embeddings, tree.root, tree.leaf_dict, 10, model)
                save.to_csv('/home/lihaitao/test/rank_qrels/dev_rank_6980_%s.csv'% (epoch),index=False,header=False)
        print("It's completed to train the network.")

        
        check_point.save(MODEL_NAME)
        # check_point.restore(MODEL_LOAD)
        save = metrics_count(dev_data, dev_qid_embeddings, tree.root, tree.leaf_dict, 10, model)
        save.to_csv(save_rank_path,index=False,header=False)
        print("END")
        
        
        num_epoch -= 1


    

if __name__ == '__main__':
    tdm()
