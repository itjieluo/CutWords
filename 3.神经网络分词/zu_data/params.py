import tensorflow as tf
# 所有参数
RNN_Cell = tf.nn.rnn_cell.LSTMCell
hidden_size = 128
batch_size = 64
cell_nums = 2
epoch_num = 2
optimizer = 'Adam'
lr = 0.001
clip = 5.0
dropout = 1
num_tags = 5
update_embedding = True
embedding_dim = 300
shuffle = False
isTrain = True
CRF = True
tag2label = {"O": 0,
             "B-BRAND": 1, "I-BRAND": 2, "E-BRAND": 3, "S-BRAND": 4,
             "B-STYLE": 5, "I-STYLE": 6, "E-STYLE": 7, "S-STYLE": 8
             }