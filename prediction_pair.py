import numpy as np
import pandas as pd
def make_data(state,node):
    length = len(state)
    r_val = []
    r_val.append(state)
    # r_val.append(length)
    if node.item_id is not None:
        r_val.append(node.item_id)
        r_val.append(1)
        # r_val.append(np.array([[node.item_id]]))
        # r_val.append(np.array([1]))
    else:
        r_val.append(node.val)
        r_val.append(0)
        # r_val.append(np.array([[node.val]]))
        # r_val.append(np.array([0]))
    # return np.array([r_val])
    return  r_val




def candidates_generator(qids, embedddings, root, leaf_dict, k, model):  #qid
    """layer-wise retrieval algorithm in prediction."""
    Q, A = [root], []
    
    while Q:
        for node in Q:
            if node.item_id is not None:
                A.append(node)
                Q.remove(node)
        
        probs = []
        for node in Q:
            prob = model.predict(qids, embedddings, node, leaf_dict)
           
            probs.append(prob[0])

        prob_list = list(zip(Q, probs))
      
        prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
      
        I = []
        if len(prob_list) > k:
            for i in range(k):
                I.append(prob_list[i][0])
        else:
            for p in prob_list:
                I.append(p[0])
        Q = []
        while I:
            node = I.pop()
            if node.left:
                Q.append(node.left)
            if node.right:
                Q.append(node.right)
    probs = []
    for leaf in A:
        prob = model.predict(qids,embedddings,leaf, leaf_dict)
        probs.append(prob[0])
    prob_list = list(zip(A, probs))
    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    A = []
    for i in range(k):
        A.append(prob_list[i][0].item_id)  #pid
    return A


def metrics_count(data, embeddings, root, leaf_dict, k, model):   #(vtest, tree.root, 10, model
    # for items in data:
    rank_list = []
    for qids in data:
        size = qids.shape[0]  #行
        for i in range(size):
            cands = candidates_generator(qids[i], embeddings[i], root, leaf_dict, k, model)  #返回的节点
            for j in range(k):
                rank = [qids[i],cands[j],j+1]
                rank_list.append(rank)
        break
    np_data = np.array(rank_list)
    save = pd.DataFrame(np_data, columns = ['qid', 'pid', 'rank'])
    return save           
            
            

 

