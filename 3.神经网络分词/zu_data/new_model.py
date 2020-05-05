import tensorflow as tf
import pickle
import params
import sys
sys.path.append("..")
import data_util

data_path = 'data.txt'
# all_list = read_data(data_path)
# make_dict(all_list, 'data_my.pk')
with open('data_my.pk', 'rb') as f1:
    word2id, id2word, tag2id = pickle.load(f1)
    # pickle.dump((word2id, id2word, tag2id), f1)
data = data_util.data_util(data_path, word2id, tag2id)
train_set = data[:-1000]
test_set = data[-1000:]

# 这里这里注意，换了embeddings方式之后，之前存的chekpoint就不能用了！
# 随机初始化的embedding方式
embeddings = data_util.random_embedding(word2id, 200)

# 使用我们自己训练的word2vec的embedding方式
# fname = '../data_util/data_preparation/data_word2vec.model'
# embeddings = data_util.word2vec_embedding(id2word, params.embedding_dim, fname)

# # 使用腾讯开源大语料的word2vec的embedding方式
# fname = 'res_dick_new.pk'
# embeddings = data_util.word2vec_embedding_big_corpus(id2word, params.embedding_dim, fname)



graph = tf.Graph()
with graph.as_default():
    # word_ids[batch_size, words]
    word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
    # word_ids[batch_size, words, labels]
    labels = tf.placeholder(tf.int32, shape=[None, None, params.num_tags], name="labels")
    # 真实序列长度
    sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")

    with tf.variable_scope("words"):
        _word_embeddings = tf.Variable(embeddings,
                                       dtype=tf.float32,
                                       trainable=params.update_embedding,
                                       name="_word_embeddings")
        # word_embeddings的shape是[None, None,params.embedding_dim]
        word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                 ids=word_ids,
                                                 name="word_embeddings")
        word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)

    with tf.variable_scope("fb-lstm"):
        cell_fw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
        cell_bw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
        rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
        rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
        (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, word_embeddings,
                                                          sequence_length=sequence_lengths, dtype=tf.float32)
        # output的shape是[None, None, params.hidden_size*2]
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        output = tf.nn.dropout(output, dropout_pl)

    with tf.variable_scope("classification"):
        # logits的shape是[None, None, params.num_tags]
        logits = tf.layers.dense(output, params.num_tags)

    with tf.variable_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        # mask的功能是产生True、False矩阵，根据最长的序列产生。类似[Ture,Ture,Ture,Ture,Ture,Ture,Ture,Ture,Ture,False]
        mask = tf.sequence_mask(sequence_lengths)
        # boolean_mask的作用将loss里面超过真实长度的loss去掉
        # 如果你这样做了，写评价函数时，也需要将pad的部分去掉。why？
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)

    with tf.variable_scope("train_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        global_add = global_step.assign_add(1)
        if params.optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=params.lr)
        elif params.optimizer == 'Adadelta':
            optim = tf.train.AdadeltaOptimizer(learning_rate=params.lr)
        elif params.optimizer == 'Adagrad':
            params = tf.train.AdagradOptimizer(learning_rate=params.lr)
        elif params.optimizer == 'RMSProp':
            optim = tf.train.RMSPropOptimizer(learning_rate=params.lr)
        elif params.optimizer == 'Momentum':
            optim = tf.train.MomentumOptimizer(learning_rate=params.lr, momentum=0.9)
        elif params.optimizer == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate=params.lr)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=params.lr)

        grads_and_vars = optim.compute_gradients(loss)
        # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        grads_and_vars_clip = [[tf.clip_by_value(g, -params.clip, params.clip), v] for g, v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

    with tf.variable_scope("evaluation"):
        true_ = tf.cast(tf.argmax(labels, axis=-1), tf.float32)
        labels_softmax_ = tf.argmax(logits, axis=-1)
        pred_ = tf.cast(labels_softmax_, tf.float32)
        zeros_like_actuals = tf.zeros_like(true_)
        four_like_actuals = tf.ones_like(true_) * 4
        mask1 = tf.equal(tf.cast(tf.equal(four_like_actuals, true_), tf.float32), zeros_like_actuals)
        true = tf.boolean_mask(true_, mask1)
        pred = tf.boolean_mask(pred_, mask1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))
        # true_ = tf.cast(tf.argmax(labels, axis=-1), tf.float32)
        # labels_softmax_ = tf.argmax(logits, axis=-1)
        # pred_ = tf.cast(labels_softmax_, tf.float32)
        # zeros_like_actuals = tf.ones_like(true_)*4
        # mask = tf.equal(tf.cast(tf.equal(zeros_like_actuals, true_), tf.float32), zeros_like_actuals)
        # true = tf.boolean_mask(true_, mask)
        # pred = tf.boolean_mask(pred_, mask)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

with tf.Session(graph=graph) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        try:
            ckpt_path = tf.train.latest_checkpoint('./checkpoint/')
            saver.restore(sess, ckpt_path)
        except ValueError:
            init = tf.global_variables_initializer()
            sess.run(init)
        for epoch in range(params.epoch_num):
            for res_seq, res_labels, sentence_legth in data_util.get_batch(train_set, params.batch_size, word2id, tag2id, shuffle=params.shuffle):
                _, l, acc, global_nums, logits_, labels_ = sess.run([train_op, loss, accuracy, global_add,pred_, true_], {
                    word_ids: res_seq,
                    labels: res_labels,
                    sequence_lengths: sentence_legth,
                    dropout_pl: params.dropout
                })
                if global_nums % 20 == 0:
                    saver.save(sess, './checkpoint/model.ckpt', global_step=global_nums)
                    print(
                    'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} '.format(
                        epoch + 1, global_nums + 1, l, acc))
        nums = 0
        for res_seq, res_labels, sentence_legth in data_util.get_batch(test_set, params.batch_size,
                                                                       word2id, tag2id,
                                                                       shuffle=params.shuffle):
            l, acc = sess.run(
                [loss, accuracy], {
                    word_ids: res_seq,
                    labels: res_labels,
                    sequence_lengths: sentence_legth,
                    dropout_pl: params.dropout
                })
            nums += 1
            if nums % 1 == 0:
                print(
                    'global_step {}, loss: {:.4}, accuracy: {:.4} '.format(
                        nums + 1, l, acc))
                # if global_nums % 20 == 0:
                #     saver.save(sess, './checkpoint2/model.ckpt', global_step=global_nums)
                #     # 获取真实序列、标签长度。
                #     pred_list, label_list = evaluation_utils.make_mask(logits_, labels_, sentence_legth)
                #     # 分别计算品牌和服饰类型的召回率
                #     recall1, recall2 = evaluation_utils.recall_rate(pred_list, label_list)
                #     print(
                #     'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}   brand_recall: {:.4}    style_recall: {:.4} '.format( epoch + 1, global_nums + 1,
                #                                                                 l, acc, recall1, recall2))
                # if global_nums % 200 == 0:
                #     print('--------test-------')
                #     for res_seq, res_labels, sentence_legth in data_util.get_batch(test_set, params.batch_size,
                #                                                                    word2id, params.tag2label, shuffle=params.shuffle):
                #         l, acc, logits_, labels_= sess.run([loss, accuracy, pred_, true_], {
                #             word_ids: res_seq,
                #             labels: res_labels,
                #             sequence_lengths: sentence_legth,
                #             dropout_pl: params.dropout
                #         })
                #         pred_list, label_list = evaluation_utils.make_mask(logits_, labels_, sentence_legth)
                #         recall1, recall2 = evaluation_utils.recall_rate(pred_list, label_list)
                #         print('test_accuracy: {}   brand_recall: {:.4}    style_recall: {:.4}  '.format(acc, recall1, recall2))
                #     print('--------test-------')


