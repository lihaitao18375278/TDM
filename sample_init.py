from operator import index
import os
from sqlite3 import Timestamp
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
from construct_tree import TreeInitialize
import pickle



def df_split(df, num):
    row = df.shape[0]
    part_size = row // num
    df_list = []
    for i in range(num):
        start, end = part_size * i, part_size * (i + 1)
        df_tmp = df.iloc[start: end, :]
        df_list.append(df_tmp)
    if row % num != 0:
        df_list.append(df.iloc[end:row, :])
    return df_list

def del_file(path_data):
    if os.path.isfile(path_data) == True:
        os.remove(path_data)


            
            

# def _single_node_sample_1(item_id, node, node_list):  #对一个节点取样
#     samples = []
#     positive_info = []
#     i = 0
#     while node:   #不断直接到父节点
#         if node.item_id is None:   #非叶节点
#             single_sample = [item_id, node.val, 0, [1,0]]   #['item_ID', 'node', 'is_leaf', 'label']
#             id = node.val
#         else:                     #叶节点
#             single_sample = [item_id, node.item_id, 1, [1,0]]
#             id = node.item_id
#         samples.append(single_sample)
#         positive_info.append(id) 
        
#         node = node.parent
#         i += 1
#     #j从root下面一层开始的层id
#     j = i-2
#     #当前tree_list_map数据结构为[[(id,is_leaf)],[]]
#     tree_depth = len(node_list)   #层数
 
#     for i in range(1,tree_depth):
#         #i为数的当前层数从1开始   node_list是 [1,0] [2,0] 
#         tmp_map = node_list[i]   #这一层所有的节点
    
#         # if(i <= 2):
#         #     index_list = random.sample(range(len(tmp_map)), 2)
#         # else:
#         index_list = random.sample(range(len(tmp_map)), 2)   #随机选2*i个数据 随机全部选出来
#         if j == 0:
#             remove_item = (positive_info[j], 1)  #到了最后一层 标记1
#         else:
#             remove_item = (positive_info[j], 0)   #1,0
            


#         sample_iter = []
#         for level_index in index_list:   #对每层随机取的2*i个节点认为为负样本
#             single_sample = [item_id, tmp_map[level_index][0], tmp_map[level_index][1], [0,1]]  #label为0
#             sample_iter.append(single_sample)

#         if [item_id, remove_item[0], remove_item[1], [0,1]] in sample_iter:
#             sample_iter.remove([item_id, remove_item[0], remove_item[1], [0,1]])   #去除掉阳性样本
#         samples.extend(sample_iter)
#         j -= 1
#         if(j < 0):
#             break
#     return samples



def _single_node_sample_1(pid, node, node_list):  #对一个节点取样
    samples = []
    positive_info = []
    i = 0
    while node:   #不断直接到父节点
        if node.item_id is None:   #非叶节点
            # single_sample = [item_id, node.val, 0, [1,0]]   #['item_ID', 'node', 'is_leaf', 'label']
            id = node.val
        else:                     #叶节点
            # single_sample = [item_id, node.item_id, 1, [1,0]]
            id = node.item_id
        # samples.append(single_sample)
        positive_info.append(id) 
        
        node = node.parent
        i += 1
    #j从root下面一层开始的层id
    j = i-2
    #当前tree_list_map数据结构为[[(id,is_leaf)],[]]
    tree_depth = len(node_list)   #层数
 
    for i in range(1,tree_depth):
        #i为数的当前层数从1开始   node_list是 [1,0] [2,0] 
        tmp_map = node_list[i]   #这一层所有的节点
    
        # if(i <= 2):
        #     index_list = random.sample(range(len(tmp_map)), 2)
        # else:
        index_list = random.sample(range(len(tmp_map)), 2)   #随机选2*i个数据 随机全部选出来
        if j == 0:
            remove_item = (positive_info[j], 1)  #到了最后一层 标记1
        else:
            remove_item = (positive_info[j], 0)   #1,0
            

        sample_iter = []
        for level_index in index_list:   #对每层随机取的2*i个节点认为为负样本
            # pair = [positive_info[j],tmp_map[level_index][0]]
            # is_leafs =[remove_item[1],tmp_map[level_index][1]]

            single_sample = [pid, positive_info[j],tmp_map[level_index][0], remove_item[1],tmp_map[level_index][1], [1,0]] 
            sample_iter.append(single_sample)

        if [pid, remove_item[0],remove_item[0], remove_item[1],remove_item[1], [1,0]] in sample_iter:
            sample_iter.remove([pid, remove_item[0],remove_item[0], remove_item[1],remove_item[1], [1,0]])   #去除掉阳性样本
        samples.extend(sample_iter)
        j -= 1
        if(j < 0):
            break
    return samples














def map_generate(df):  #输入tree_sasmples
    #生成map 为了提高访问速度
    r_value = {}
    df = df.values  #变成list形式
    for i in df:
        value = r_value.get(i[0])  #pid
        if value == None:
            r_value[i[0]] = [[i[1],i[2],i[3],i[4],i[5]]]  #对应的node is_leaf label
        else:
            r_value[i[0]].append([i[1], i[2], i[3],i[4],i[5]])
    return r_value



def tree_generate_samples(items, leaf_dict, node_list):  #pass_ids, tree.leaf_dict, node_list
    """Sample based on the constructed tree with multiprocess."""
    samples_total = []
    for item in items:
        if item != -2:
            node = leaf_dict[item]   #取出物品对应的叶子结点
            samples = _single_node_sample_1(item, node, node_list)  #对一个item取出对应的node取样
            samples_total.extend(samples)
        # total_samples = pd.concat(samples, ignore_index=True)
    samples = pd.DataFrame(samples_total, columns=['pid', 'positive', 'negetive', 'p_isleaf', 'n_isleaf', 'label'])
    return samples


def sample_merge_multiprocess(data, tree_map, mode, split_num ,dir):  #data_train , tree_map,'train',7,train_dir
    del_file(dir)
    isExists=os.path.exists(dir)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(dir)
    
    
    
    df_list = df_split(data, split_num)
 
    
    length = len(df_list)
    print("total dataset length %d df_list_length is %d" % (len(data),length))
    from multiprocessing import Pool, Process
    # datas = Manager().list()
    p_list = []
    for i in range(length):
        p = Process(target=merge_samples, args=(df_list[i], tree_map, mode, i, dir))
        p.start()
        p_list.append(p)
    for res in p_list:
        res.join()

def merge_samples(data, tree_map,mode,process_id, dir): #data时 qid pid
 
    t_1 = time.perf_counter()
    print('-----------> 进程: %d - chunk: %s <------------' % (process_id, data.shape[0]))
    #生成样本数据 为了效率 树生成的物品index改成map结构
    train_size = data.shape[0]
    r_value = []
    #[qid,pid] ['node', 'is_leaf', 'label']
    j = 0
    s = time.perf_counter()
    for i in range(train_size):
        data_row = data.iloc[i]  #iloc[行索引位置,列索引位置] 取一行数据
        data_row_values = data_row.values  #qid pid numpy
        pid = data_row.pid
        # data_row_values_tile = list_tile(data_row_values,1)
        data_row_values_tile = data_row_values
        l_len = len(tree_map[pid])
      

        tmp = np.append(l_len*[data_row_values_tile],tree_map[pid],axis=1)
        r_value.extend(tmp)
        if(i % 10000 == 0 and i != 0):   #1000个数据一保存
            # np.savetxt('/home/dev/data/andrew.zhu/tdm/data_flow/%s/%s_%s.csv' % (mode,process_id,j), r_value, delimiter=",",fmt='%d')
            pd.DataFrame(r_value)\
                .to_csv(dir+'/%i_%s.csv' % (process_id,j),
                                         header=False,index=False)
            print('mode:%s,process:%s,epoch:%d,time:%f,length:%d' % (mode,process_id,j, time.clock() - s,len(r_value)))
            s = time.perf_counter()
            r_value = []
            j = j + 1
    if len(r_value)!= 0:
        pd.DataFrame(r_value) \
            .to_csv(dir+'/%i_%s.csv' % (process_id, j),
                    header=False, index=False)
    t_2 = time.perf_counter()
    print('进程 %d : time_use=%.2f s' % (process_id, t_2 - t_1))
    """combine the preprocessed samples and tree samples."""




class Dataset(object):
    """construct the dataset iterator."""
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        
        # self.data = self.data.drop(columns=['user_ID', 'timestamp'])
        N, B = self.data.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        if self.data.shape[1] > 2:  #训练数据
            return ((np.array(self.data.loc[idxs[i:i+B], 'qid'].tolist()),
                     np.array(self.data.loc[idxs[i:i+B], 'pid'].tolist()),
                     self.data.loc[idxs[i:i+B], 'positive'].values[:, None],
                     self.data.loc[idxs[i:i+B], 'negetive'].values[:, None],
                     self.data.loc[idxs[i:i+B], 'p_isleaf'].values[:, None],
                     self.data.loc[idxs[i:i+B], 'n_isleaf'].values[:, None],
                     np.array(self.data.loc[idxs[i:i+B], 'label'].tolist())) for i in range(0, N, B))
        else:   #代表测试数据
            return (np.array(self.data.loc[idxs[i:i+B], 'qid'].tolist()) for i in range(0, N, B))





if __name__ == '__main__':
    data_process()
    test_pickle()