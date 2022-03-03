import numpy as np
import time
import os
import pandas as pd

class TreeNode(object):
    """define the tree node structure."""
    def __init__(self, x, item_id=None,item_embedding=None):
        self.val = x
        self.item_id = item_id   #pid 这个记录了pid
        self.embedding = item_embedding  #array形式
        self.parent = None
        self.left = None
        self.right = None


class TreeInitialize(object):
    """"Build the random binary tree."""
    def __init__(self, pid_embeddings, pid_list):  #pid_embeddings, pid_list
        #pass-embedding
        self.embeddings = pid_embeddings
        # 唯一passage-list
        self.pid_list = pid_list 
        #root节点
        self.root = None
        #叶子节点item_id -> TreeNode
        self.leaf_dict = {}
        self.node_dict = {}
        #非叶子节点的个数
        self.node_size = 0
        
    def _balance_clutering(self, c1, c2, item1, item2):
        amount = item1.shape[0] - item2.shape[0]
        if amount > 1:
            num = int(amount / 2)
            distance = np.sum(np.square(item1 - c1), axis=1)
            item_move = item1[distance.argsort()[-num:]]
            item2_adjust = np.concatenate((item2, item_move), axis=0)
            item1_adjust = np.delete(item1, distance.argsort()[-num:], axis=0)
        elif amount < -1:
            num = int(abs(amount) / 2)
            distance = np.sum(np.square(item2 - c2), axis=1)
            item_move = item2[distance.argsort()[-num:]]
            item1_adjust = np.concatenate((item1, item_move), axis=0)
            item2_adjust = np.delete(item2, distance.argsort()[-num:], axis=0)
        else:
            item1_adjust, item2_adjust = item1, item2
        return item1_adjust, item2_adjust

    def _k_means_clustering(self, items):  #可不可以改成用函数
        m1, m2 = items[0], items[1]
    
        while True:
            indicate = np.sum(np.square(items - m1), axis=1) - np.sum(np.square(items - m2), axis=1) #在列展开角度上求和也就是每行求和
    
            items_m1, items_m2 = items[indicate < 0], items[indicate >= 0]
            if items_m1.shape[0] == 0 or items_m2.shape[0] == 0:
                break
            if items_m1.shape[0] == 1 and items_m2.shape[0] == 1:
                break
            m1_new = np.sum(items_m1, axis=0) / items_m1.shape[0]   #第一部分的质心
            m2_new = np.sum(items_m2, axis=0) / items_m2.shape[0]   #第二部分的质心
            if np.sum(np.absolute(m1_new - m1)) < 1e-3 and np.sum(np.absolute(m2_new - m2)) < 1e-3: #再查
                break
            m1, m2 = m1_new, m2_new
        items_m1, items_m2 = self._balance_clutering(m1, m2, items_m1, items_m2)
        return items_m1, items_m2

    def _build_binary_tree(self, root, items):
        if items.shape[0] == 1:
            leaf_node = TreeNode(0, item_id=self.pid_list[self.embeddings.index(list(items[0]))], item_embedding=np.array(items[0]))
            leaf_node.parent = root.parent
            return leaf_node
        left_items, right_items = self._k_means_clustering(items)
        # print(len(left_items))
        # print(len(right_items))
        left_child, right_child = TreeNode(0), TreeNode(0)
        left_child.parent, right_child.parent = root, root
        left = self._build_binary_tree(left_child, left_items)
        right = self._build_binary_tree(right_child, right_items)
        root.left, root.right = left, right
        return root

    def clustering_binary_tree(self):  #items=item_embeddings
        root = TreeNode(0)
        embeddings = np.array(self.embeddings)
        self.root = self._build_binary_tree(root, embeddings)
        _ = self._define_node_index(self.root)
        _ = self._define_node_emebedding(self.root)
        _ = self._get_node_dict(self.root)
        return self.root
    
    def _define_node_index(self, root):
        node_queue = [root]
        i = 0
        try:
            while node_queue:
                current_node = node_queue.pop(0)  #第一个出来
                if current_node.left:
                    node_queue.append(current_node.left)
                if current_node.right:
                    node_queue.append(current_node.right)
                if current_node.item_id is not None:  #只有叶节点有item_id
                    self.leaf_dict[current_node.item_id] = current_node
                else:
                    current_node.val = i
                   
                    i += 1
            self.node_size = i
            return 0
        except RuntimeError as err:
            print("Runtime Error Info: {0}".format(err))
            return -1
        
    def _define_node_emebedding(self, root):
        current_node = root
        i = 0
        try:
            if current_node.embedding is None:  #只有叶节点有item_id
                # print('node')
                # print(current_node.val)
                # print(current_node.left.embedding)
                # print(current_node.right.embedding)
                current_node.embedding = (self._define_node_emebedding(current_node.left)+self._define_node_emebedding(current_node.right))/2
                return np.array(current_node.embedding)
            else:
                # print('item')
                # print(current_node.val)
                # # print(current_node.item_id)
                return np.array(current_node.embedding)
        except RuntimeError as err:
            print("Runtime Error Info: {0}".format(err))
            return -1
    
    def _get_node_dict(self, root):
        node_queue = [root]
        try:
            while node_queue:
                current_node = node_queue.pop(0)  #第一个出来
                if current_node.left:
                    node_queue.append(current_node.left)
                if current_node.right:
                    node_queue.append(current_node.right)
                if current_node.item_id is None:  #非叶节点
                    self.node_dict[current_node.val] = current_node
               
            return 0
        except RuntimeError as err:
            print("Runtime Error Info: {0}".format(err))
            return -1
        
        
    def _node_list(self, root):
        #将二叉树数据提出放入list
        def node_val(node):
            if(node.left or node.right):
                return (node.val,0)
            else:
                return (node.item_id,1)
            
        node_queue = [root]
        arr_arr_node = []
        arr_arr_node.append([node_val(node_queue[0])])
        while node_queue:
            tmp = []
            tmp_val = []
            for i in node_queue:
                if i.left:
                    tmp.append(i.left)
                    tmp_val.append(node_val(i.left))
                if i.right:
                    tmp.append(i.right)
                    tmp_val.append(node_val(i.right))
            if len(tmp_val) > 0:
                arr_arr_node.append(tmp_val)
            node_queue = tmp
        return arr_arr_node

    





if __name__ == '__main__':
    data_raw = pd.read_csv(LOAD_DIR, header=None,
                           names=['user_ID', 'item_ID', 'category_ID', 'behavior_type', 'timestamp'])
    # data_raw = data_raw[:10000]
    user_list = data_raw.user_ID.drop_duplicates().to_list()  ##去除重复数据
    user_dict = dict(zip(user_list, range(len(user_list))))
    data_raw['user_ID'] = data_raw.user_ID.apply(lambda x: user_dict[x])
    # print(data_raw['user_ID'])
    item_list = data_raw.item_ID.drop_duplicates().to_list()
    item_dict = dict(zip(item_list, range(len(item_list))))
    data_raw['item_ID'] = data_raw.item_ID.apply(lambda x: item_dict[x])
    category_list = data_raw.category_ID.drop_duplicates().to_list()
    category_dict = dict(zip(category_list, range(len(category_list))))
    data_raw['category_ID'] = data_raw.category_ID.apply(lambda x: category_dict[x])
    behavior_dict = dict(zip(['pv', 'buy', 'cart', 'fav'], range(4)))
    data_raw['behavior_type'] = data_raw.behavior_type.apply(lambda x: behavior_dict[x])

    time_window = _time_window_stamp()
    data_raw['timestamp'] = data_raw.timestamp.apply(_time_converter, args=(time_window,))
    # print(data_raw)
    random_tree = TreeInitialize(data_raw)
    _ = random_tree.random_binary_tree()


