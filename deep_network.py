from email.errors import InvalidMultipartContentTransferEncodingDefect
from multiprocessing.dummy import active_children
import os
from signal import pthread_kill
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
import numpy as np
import tensorflow as tf
from tensorflow import keras


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution() 


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = MODEL_DIR + '/models/network_model.ckpt'
SUMMARY_DIR = MODEL_DIR + '/logs'


class NeuralNet(tf.keras.Model):
    """Deep network structure:
    input_embedding+node_embedding >>
    attention_block >>
    union_embedding >>
    MLP(128>64>24>2) >>
    label_probabilities.
    """
    def __init__(self,node_embeddings):
        super().__init__()
        
        
        # self.node_embeddings = tf.Variable(tf.random.normal([4999,768],stddev=2),trainable=True)
        self.node_embeddings = tf.Variable(node_embeddings,trainable=True)
        self.layer00 = tf.keras.layers.Dense(36,activation=self._PRelu)
        self.layer01 = tf.keras.layers.Dense(1)
        
        
        self.layer10 = tf.keras.layers.Dense(1024,activation=self._PRelu)
        self.layer10_bn = tf.keras.layers.BatchNormalization()
        self.layer20 = tf.keras.layers.Dense(512,activation=self._PRelu)
        self.layer20_bn = tf.keras.layers.BatchNormalization()
        self.layer30 = tf.keras.layers.Dense(256,activation=self._PRelu)
        self.layer30_bn = tf.keras.layers.BatchNormalization()
        
        
        
        self.layer1 = tf.keras.layers.Dense(128,activation=self._PRelu)
       
        self.layer1_bn = tf.keras.layers.BatchNormalization()
        self.layer2 = tf.keras.layers.Dense(64,activation=self._PRelu)
        self.layer2_bn = tf.keras.layers.BatchNormalization()
        self.layer3 = tf.keras.layers.Dense(24,activation=self._PRelu)
        self.layer3_bn = tf.keras.layers.BatchNormalization()
        self.layer4 = tf.keras.layers.Dense(2,activation='softmax')
 

    def call(self, qid_embeddings, positives = None, negetives = None, p_isleafs=None, n_isleafs=None, node_dict=None, leaf_dict=None, is_training=1): 
        batch=np.array(qid_embeddings).shape[0]
        # batch, _ = tf.shape(items)
        features = None
        for i in range(batch):
            qid_embedding = qid_embeddings[i]
            if(is_training == 1):
                positive = int(positives[i])
                negetive = int(negetives[i])
                p_isleaf = int(p_isleafs[i])
                n_isleaf = int(n_isleafs[i])
           
                if(p_isleaf == 1):
                    positive_embedding = leaf_dict[positive].embedding
                else:
                    positive_embedding = tf.nn.embedding_lookup(self.node_embeddings, positive)

                if(n_isleaf == 1):
                    negetive_embedding = leaf_dict[negetive].embedding
                else:
                    negetive_embedding = tf.nn.embedding_lookup(self.node_embeddings, negetive)

            
            
                positive_embedding,negetive_embedding = tf.reshape(positive_embedding, [1, -1]), tf.reshape(negetive_embedding, [1, -1])

            else:  
                positive = positives
                p_isleaf = p_isleafs
                if(p_isleaf == 1):
                    positive_embedding = leaf_dict[positive].embedding
                else:
                    positive_embedding = tf.nn.embedding_lookup(self.node_embeddings, positive)
                positive_embedding = tf.reshape(positive_embedding, [1, -1])
                negetive_embedding = tf.ones([1, 768], dtype=tf.float32)
                # negetive_embedding = tf.reshape(qid_embedding, [1, -1])


            hybrid_positive = qid_embedding * positive_embedding
            hybrid_negetive = qid_embedding * negetive_embedding
           
            qid_embedding = tf.reshape(qid_embedding, [1, -1])
            # hybrid_positive = tf.reduce_sum(hybrid_positive)
            # hybrid_negetive = tf.reduce_sum(hybrid_negetive)
            # print(hybrid_positive)
            # print(hybrid_negetive)

            feature = tf.concat([qid_embedding,positive_embedding, negetive_embedding], axis=1)
            layer00 = self.layer00(feature)     
            weight = self.layer01(layer00) 
            item_weight = weight
            item_feature1 = item_weight * positive_embedding
            item_feature2 = item_weight * negetive_embedding
            item_feature = tf.concat([tf.reshape(item_feature1, [1, -1]), tf.reshape(item_feature2, [1, -1]),  qid_embedding], axis=1)
          
            # item_feature = tf.concat([qid_embedding,hybrid_positive, negetive_embedding], axis=1) 
             # print(sum)
            if features is None:
                features = item_feature
            else:
                features = tf.concat([features, item_feature], axis=0)
   

        batch_features=features
        # layer10 = self.layer10(batch_features)
        # layer10_bn = self.layer10_bn(layer10)
        # layer20 = self.layer20(layer10_bn)
        # layer20_bn = self.layer20_bn(layer20)
        # layer30 = self.layer30(layer20_bn)
        # layer30_bn = self.layer30_bn(layer30)
    
        
        
        layer1 = self.layer1(batch_features)
       
        layer1_bn = self.layer1_bn(layer1)
        layer2 = self.layer2(layer1_bn)
        layer2_bn = self.layer2_bn(layer2)
        layer3 =self.layer3(layer2_bn)
        layer3_bn = self.layer3_bn(layer3)
        logits = self.layer4(layer3_bn)
        return logits


    def _PRelu(self, x):
        # m, n = tf.shape(x)
        m=x.shape[0]
        n=x.shape[1]
        value_init = 0.25 * tf.ones((1, n))
        a = tf.Variable(initial_value=value_init)
        y = tf.maximum(x, 0) + a * tf.minimum(x, 0)
        return y
    
    
    
    
    # def predict(self, qid, embedding, node, use_gpu=True):
    #     """
    #     TODO: validate and optimize
    #     """
    #     device = '/gpu:1' if use_gpu else '/cpu:0'
    #     with tf.device(device):
    #         qid_embedding = embedding
    #         qid_embedding = tf.reshape(qid_embedding, [1, -1])
    #         pid=node.item_id
    #         pid_embedding = node.embedding
    #         pid_embedding = tf.reshape(pid_embedding, [1, -1])
    #         # items=np.array(items).reshape((1,-1))
    #         # nodes=np.array(nodes).reshape((1,-1))
    #         # is_leafs=np.array(is_leafs).reshape((1,-1))
    #         scores = self.call(qid_embedding, pid_embedding, is_training=0)
    #         scores = scores.numpy()
    #     return scores[:, 0]


    def predict(self, qid, embedding, node, leaf_dict, use_gpu=True):
        """
        TODO: validate and optimize
        """
        device = '/gpu:1' if use_gpu else '/cpu:0'
        with tf.device(device):
            qid_embedding = embedding
            qid_embedding = tf.reshape(qid_embedding, [1, -1])
            pid_embedding = node.embedding
            pid_embedding = tf.reshape(pid_embedding, [1, -1])
            if(node.item_id is not None):
  
                pid = node.item_id
                is_leaf = 1
            else:
    
                pid = node.val
                is_leaf = 0

            # items=np.array(items).reshape((1,-1))
            # nodes=np.array(nodes).reshape((1,-1))
            # is_leafs=np.array(is_leafs).reshape((1,-1))
            scores = self.call(qid_embedding, positives = pid, p_isleafs = is_leaf, leaf_dict=leaf_dict, is_training=0)
            scores = scores.numpy()
        return scores[:, 0]


if __name__ == '__main__':
    model = NeuralNet(744, 743, 24)
