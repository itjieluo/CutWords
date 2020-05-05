import tensorflow as tf
import pickle
import numpy as np

with open('data_my.pk', 'rb') as f1:
    word2id, id2word, tag2id = pickle.load(f1)

def run_model(all_seq, seq_length):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # 加载模型
        check_point_path = 'checkpoint/'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
        last_but_one_ckpt = ckpt.all_model_checkpoint_paths[-2]
        saver = tf.train.import_meta_graph(last_but_one_ckpt+'.meta')
        saver.restore(sess, last_but_one_ckpt)
        input_placeholder = tf.get_default_graph().get_tensor_by_name("word_ids:0")  # [batch_size, 200, 200, 3]
        keep_prob_placeholder = tf.get_default_graph().get_tensor_by_name("dropout:0")  # [batch_size, 200, 200, 3]
        sequence_lengths = tf.get_default_graph().get_tensor_by_name("sequence_lengths:0")  # [batch_size, 200, 200, 3]
        output_feature = tf.get_default_graph().get_tensor_by_name("evaluation/Cast_1:0")
        pred = sess.run(output_feature, feed_dict={input_placeholder: all_seq, sequence_lengths:seq_length, keep_prob_placeholder:1.0})
        return pred


def make_inputs(words):
    pad = word2id['<PAD>']
    all_seq = [word2id[word] for word in words]
    sent_new = np.concatenate((all_seq, np.tile(pad, 32 - len(all_seq))), axis=0)
    sent_new = np.reshape(sent_new, (1, 32))
    print(sent_new)
    return sent_new

def prediction_res(words,pred):
    # print(pred)
    # print(pred[0])
    id2tag = {tag2id[key]:key for key in tag2id}
    tag_res = [id2tag[pred[0][i]] for i in range(len(words))]
    # tag_res = [id2tag[pred[i]] for i in range(len(words))]
    print(words)
    print(tag_res)
if __name__ == '__main__':
    # words没有做新词，默认所有词都有，不能写空格
    # words = '我家住在北京天安门'
    words = '我爱你中国'
    # words = '王军虎去广州了'
    # words = '在北京大学生活区喝进口红酒'
    # words = '学生会宣传部'
    # words = '沿海南方向逃跑'
    # words = '这样的人才能经受住考验'
    # words = '网曝徐峥夜会美女'
    sent_new = make_inputs(words)
    pred = run_model(sent_new, [len(words)])
    prediction_res(words, pred)
