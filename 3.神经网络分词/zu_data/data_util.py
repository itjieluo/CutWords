import re
import pickle
import random
import numpy as np


# 将2维的tag转化为one-hot形式，return结果为3维
def to_one_hot(labels, tag_nums):
    """

    :param labels:
    :param tag_nums:
    :return:
    """
    length = len(labels)
    len_lab = len(labels[0])
    res = np.zeros((length, len_lab, tag_nums), dtype=np.float32)
    for i in range(length):
        for j in range(len_lab):
            res[i][j][labels[i][j]] = 1.
    return np.array(res)



# 产生随机的embedding矩阵
def random_embedding(word2id, embedding_dim):
    """

    :param id2word:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        word_list = f.read().split()
    all_list = [[word.split('/')[0], word.split('/')[1]] for word in word_list]
    return all_list


# 制作word2id和id2word
def make_dict(all_list, dick_name):
    all_char = []
    all_tag = []
    for i in all_list:
        if i[0] not in all_char:
            all_char.append(i[0])
        if i[1] not in all_tag:
            all_tag.append(i[1])
    word2id = {}
    id2word = {}
    tag2id = {}
    all_char.append('<UNK>')
    all_char.append('<PAD>')
    all_tag.append('x')
    for index, char in enumerate(all_char):
        word2id[char] = index
        id2word[index] = char
    for index, char in enumerate(all_tag):
        tag2id[char] = index
    with open(dick_name, 'wb') as f1:
        pickle.dump((word2id, id2word, tag2id), f1)
    return word2id, id2word, tag2id


def data_util(data_path,word2id, tag2id):
    with open(data_path, "r", encoding="utf8") as f:
        data = f.read()
    rr = re.compile(r'[,，。、“”‘’－》《（）●：！;…？]/s')
    sentences = rr.split(data)
    sentences = list(filter(lambda x: x.strip(), sentences))

    sentences = list(map(lambda x: x.strip(), sentences))

    all_list = []
    for i in sentences:
        word_list = i.split()
        one_list = [[word2id[word.split('/')[0]], tag2id[word.split('/')[1]]] for word in word_list]
        one_list.append(len(word_list))
        all_list.append(one_list)
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     word_list = f.read().split()
    # all_list = [[word2id[word.split('/')[0]], tag2id[word.split('/')[1]]] for word in word_list]
    # all_list = [all_list[i * 32:(i+1) * 32] for i in range(len(all_list)//32)]
    return all_list


# 将数据pad，生成batch数据返回，这里没有取余数。
def get_batch(data, batch_size, word2id, tag2id, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        random.shuffle(data)
    pad = word2id['<PAD>']
    tag_pad = tag2id["x"]
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        seqs, labels, sentence_legth = [], [], []
        for i in data_size:
            one_line = np.array(i[:-1])
            seqs.append(one_line[:,0])
            labels.append(one_line[:,1])
            sentence_legth.append(i[-1])
        max_l = max(sentence_legth)

        res_seq = []
        for sent in seqs:
            sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
            res_seq.append(sent_new)

        res_labels = []
        for label in labels:
            label_new = np.concatenate((label, np.tile(tag_pad, max_l - len(label))), axis=0)  # 以pad的形式补充成等长的帧数
            res_labels.append(label_new)

        res_labels = to_one_hot(res_labels, 5)
        yield np.array(res_seq), res_labels, sentence_legth

# 将数据pad，生成batch数据返回，这里没有取余数。
def get_batch1(data, batch_size, word2id, tag2id, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        random.shuffle(data)
    pad = word2id['<PAD>']
    tag_pad = tag2id["x"]
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        seqs, labels, sentence_legth = [], [], []
        for i in data_size:
            one_line = np.array(i[:-1])
            seqs.append(one_line[:, 0])
            labels.append(one_line[:, 1])
            sentence_legth.append(i[-1])
        max_l = max(sentence_legth)
        res_seq = []
        for sent in seqs:
            if len(sent)>=32:
                sent_new = sent[:32]
            else:
                sent_new = np.concatenate((sent, np.tile(pad, 32 - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
            res_seq.append(sent_new)
        res_labels = []
        for label in labels:
            if len(label)>=32:
                label_new = label[:32]
            else:
                label_new = np.concatenate((label, np.tile(tag_pad, 32 - len(label))), axis=0)  # 以pad的形式补充成等长的帧数
            res_labels.append(label_new)
        res_labels = to_one_hot(res_labels, 5)
        yield np.array(res_seq), res_labels, sentence_legth
# # 生成器获取数据
# def gen_batch(dataset, batchsize,word2id, tag2id, shuffle=False):
#     # 乱序没有加
#     if shuffle:
#         np.random.shuffle(dataset)
#     pad = word2id['<PAD>']
#     tag_pad = tag2id["O"]
#     for i in range(dataset.shape[0]//batchsize):
#         pos = i * batchsize
#         x = dataset[pos:pos + batchsize, :, 0]
#         y = dataset[pos:pos + batchsize, :, 1]
#
#         y = to_one_hot(y, 4)
#         yield x, y
if __name__ == '__main__':
    data_path = 'data.txt'
    all_list = read_data(data_path)
    make_dict(all_list, 'data_my.pk')
    # with open('data_my.pk', 'rb') as f1:
    #     word2id, id2word, tag2id = pickle.load(f1)
    #     # pickle.dump((word2id, id2word, tag2id), f1)
    # data = data_util(data_path, word2id, tag2id)
    # for x,y,z in get_batch1(data, 64, word2id, tag2id):
    #     print(x.shape)
    #     print(y.shape)