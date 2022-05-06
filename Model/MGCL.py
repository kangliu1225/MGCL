import time
import tensorflow as tf
import os
import sys
from load_data import Data
import numpy as np
import math
import multiprocessing
import heapq
import random as rd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'MGCL'

# ['amazon-beauty', 'Art', 'ali']
dataset = 'amazon-beauty'

'''
#######################################################################
Hyper-parameter settings.
'''

if dataset == 'amazon-beauty':
    n_layers = 4
    temp = 0.04
    decay = 0.001
    lambda_m = 0.4
    interval = 10

elif dataset == 'Art':
    n_layers = 5
    temp = 0.04
    decay = 0.001
    lambda_m = 0.4
    interval = 10
else:
    n_layers = 5
    temp = 0.04
    decay = 0.001
    lambda_m = 0.4
    interval = 5


data_path = '../Data/'
lr = 0.001
batch_size = 2048
embed_size = 64
epoch = 500
data_generator = Data(path=data_path + dataset, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = batch_size
Ks = np.arange(1, 21)

if dataset == 'ali':
    textual_enable = False
else:
    textual_enable = True


# model test module
def test_one_user(x):
    u, rating = x[1], x[0]

    training_items = data_generator.train_items[u]

    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = rd.sample(list(all_items - set(training_items) - set(user_pos_test)), 99) + user_pos_test

    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    precision, recall, ndcg, hit_ratio = [], [], [], []

    def hit_at_k(r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def ndcg_at_k(r, k):
        r = np.array(r)[:k]

        if np.sum(r) > 0:
            return math.log(2) / math.log(np.where(r == 1)[0] + 2)
        else:
            return 0.

    for K in Ks:
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'k': (u, K_max_item_score[:20])}


def test(sess, model, users, items, batch_size, cores):
    result = {'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'k': []}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2

    n_test_users = len(users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size

        end = (u_batch_id + 1) * u_batch_size

        user_batch = users[start: end]

        item_batch = items

        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                    model.pos_items: item_batch})

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['k'].append(re['k'])

    assert count == n_test_users
    pool.close()
    return result

class Model(object):

    def __init__(self, data_config, img_feat, text_feat, d1, d2):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.d1 = d1
        self.d2 = d2
        self.n_fold = 10
        self.norm_adj = data_config['norm_adj']
        self.norm_adj_m = data_config['norm_adj_m']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = data_config['lr']
        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']
        self.n_layers = data_config['n_layers']
        self.decay = data_config['decay']

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()

        self.im_v = tf.matmul(img_feat, self.weights['w1_v'])
        self.um_v = self.weights['user_embedding_v']

        self.im_t = tf.matmul(text_feat, self.weights['w1_t'])
        self.um_t = self.weights['user_embedding_t']

        '''
        ######################################################################################
        generate interactive-dimension embeddings
        '''
        self.ua_embeddings, self.ia_embeddings = self._create_norm_embed()
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        '''
        ######################################################################################
        generate multimodal-dimension embeddings
        '''
        self.ua_embeddings_v, self.ia_embeddings_v = self._create_norm_embed_v()
        self.u_g_embeddings_v = tf.nn.embedding_lookup(self.ua_embeddings_v, self.users)
        self.pos_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.pos_items)
        self.neg_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.neg_items)

        self.u_g_embeddings_v_pre = tf.nn.embedding_lookup(self.um_v, self.users)
        self.pos_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.pos_items)
        self.neg_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.neg_items)

        self.ua_embeddings_t, self.ia_embeddings_t = self._create_norm_embed_t()
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)
        self.neg_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.neg_items)

        self.u_g_embeddings_t_pre = tf.nn.embedding_lookup(self.um_t, self.users)
        self.pos_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.pos_items)
        self.neg_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.neg_items)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings,
                                       transpose_a=False, transpose_b=True) + \
                             lambda_m * tf.matmul(self.u_g_embeddings_v, self.pos_i_g_embeddings_v,
                                       transpose_a=False, transpose_b=True) + \
                             tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t,
                                       transpose_a=False, transpose_b=True)

        self.mf_loss, self.mf_loss_m, self.emb_loss = self.create_bpr_loss()
        self.cl_loss, self.cl_loss_v, self.cl_loss_t = self.create_cl_loss_cf(), self.create_cl_loss_v(), self.create_cl_loss_t()

        self.opt_1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mf_loss + self.mf_loss_m + self.emb_loss)
        self.opt_2 = tf.train.AdamOptimizer(learning_rate=self.lr/10).minimize(self.cl_loss + self.cl_loss_v + self.cl_loss_t)

    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['user_embedding_v'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_v')
        all_weights['user_embedding_t'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_t')

        all_weights['w1_v'] = tf.Variable(initializer([self.d1, self.emb_dim]), name='w1_v')
        all_weights['w1_t'] = tf.Variable(initializer([self.d2, self.emb_dim]), name='w1_t')

        return all_weights

    def _split_A_hat(self, X):

        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_norm_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        for k in range(0, self.n_layers):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_norm_embed_v(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_m)

        ego_embeddings_v = tf.concat([self.um_v, self.im_v], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_v = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_t(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_m)

        ego_embeddings_t = tf.concat([self.um_t, self.im_t], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_t))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_t = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_t, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def create_bpr_loss(self):
        pos_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.pos_i_g_embeddings_v), axis=1)
        neg_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.neg_i_g_embeddings_v), axis=1)

        pos_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.pos_i_g_embeddings_t), axis=1)
        neg_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.neg_i_g_embeddings_t), axis=1)

        pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)

        regularizer_mf_v = tf.nn.l2_loss(self.u_g_embeddings_v_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_v_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_v_pre)
        regularizer_mf_t = tf.nn.l2_loss(self.u_g_embeddings_t_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_t_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_t_pre)
        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_pre)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        mf_loss_m = tf.reduce_mean(tf.nn.softplus(-(pos_scores_v - neg_scores_v))) \
                  + tf.reduce_mean(tf.nn.softplus(-(pos_scores_t - neg_scores_t)))

        emb_loss = self.decay * (regularizer_mf + 0.1 * regularizer_mf_t + 0.1 * regularizer_mf_v) / self.batch_size

        self.user_embed, self.user_embed_v, self.user_embed_t = tf.nn.l2_normalize(self.u_g_embeddings_pre, axis=1), \
                                                                tf.nn.l2_normalize(self.u_g_embeddings_v_pre, axis=1), \
                                                                tf.nn.l2_normalize(self.u_g_embeddings_t_pre, axis=1)
        self.item_embed, self.item_embed_v, self.item_embed_t = tf.nn.l2_normalize(self.pos_i_g_embeddings_pre, axis=1), \
                                                                tf.nn.l2_normalize(self.pos_i_g_embeddings_v_pre, axis=1), \
                                                                tf.nn.l2_normalize(self.pos_i_g_embeddings_t_pre, axis=1)

        return mf_loss, mf_loss_m, emb_loss

    def create_cl_loss_cf(self):
        pos_score_v = tf.reduce_sum(tf.multiply(self.user_embed, self.user_embed_v), axis=1)
        pos_score_t = tf.reduce_sum(tf.multiply(self.user_embed, self.user_embed_t), axis=1)

        neg_score_v_u = tf.matmul(self.user_embed, self.user_embed_v, transpose_a=False, transpose_b=True)
        neg_score_t_u = tf.matmul(self.user_embed, self.user_embed_t, transpose_a=False, transpose_b=True)

        cl_loss_v_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_v / temp) / tf.reduce_sum(tf.exp(neg_score_v_u / temp), axis=1)))
        cl_loss_t_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_t / temp) / tf.reduce_sum(tf.exp(neg_score_t_u / temp), axis=1)))

        pos_score_v = tf.reduce_sum(tf.multiply(self.item_embed, self.item_embed_v), axis=1)
        pos_score_t = tf.reduce_sum(tf.multiply(self.item_embed, self.item_embed_t), axis=1)

        neg_score_v_i = tf.matmul(self.item_embed, self.item_embed_v, transpose_a=False, transpose_b=True)
        neg_score_t_i = tf.matmul(self.item_embed, self.item_embed_t, transpose_a=False, transpose_b=True)

        cl_loss_v_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_v / temp) / tf.reduce_sum(tf.exp(neg_score_v_i / temp), axis=1)))
        cl_loss_t_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_t / temp) / tf.reduce_sum(tf.exp(neg_score_t_i / temp), axis=1)))

        cl_loss = cl_loss_v_u + cl_loss_t_u + cl_loss_v_i + cl_loss_t_i

        return cl_loss

    def create_cl_loss_v(self):
        pos_score_1 = tf.reduce_sum(tf.multiply(self.user_embed_v, self.user_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.user_embed_v, self.user_embed_t), axis=1)

        neg_score_1_u = tf.matmul(self.user_embed_v, self.user_embed, transpose_a=False, transpose_b=True)
        neg_score_2_u = tf.matmul(self.user_embed_v, self.user_embed_t, transpose_a=False, transpose_b=True)

        cl_loss_1_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_u / temp), axis=1)))
        cl_loss_2_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_u / temp), axis=1)))

        pos_score_1 = tf.reduce_sum(tf.multiply(self.item_embed_v, self.item_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.item_embed_v, self.item_embed_t), axis=1)

        neg_score_1_i = tf.matmul(self.item_embed_v, self.item_embed, transpose_a=False, transpose_b=True)
        neg_score_2_i = tf.matmul(self.item_embed_v, self.item_embed_t, transpose_a=False, transpose_b=True)

        cl_loss_1_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_i / temp), axis=1)))
        cl_loss_2_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_i / temp), axis=1)))

        cl_loss = cl_loss_1_u + cl_loss_2_u + cl_loss_1_i + cl_loss_2_i

        return cl_loss

    def create_cl_loss_t(self):
        pos_score_1 = tf.reduce_sum(tf.multiply(self.user_embed_t, self.user_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.user_embed_t, self.user_embed_v), axis=1)

        neg_score_1_u = tf.matmul(self.user_embed_t, self.user_embed, transpose_a=False, transpose_b=True)
        neg_score_2_u = tf.matmul(self.user_embed_t, self.user_embed_v, transpose_a=False, transpose_b=True)

        cl_loss_1_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_u / temp), axis=1)))
        cl_loss_2_u = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_u / temp), axis=1)))

        pos_score_1 = tf.reduce_sum(tf.multiply(self.item_embed_t, self.item_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.item_embed_t, self.item_embed_v), axis=1)

        neg_score_1_i = tf.matmul(self.item_embed_t, self.item_embed_v, transpose_a=False, transpose_b=True)
        neg_score_2_i = tf.matmul(self.item_embed_t, self.item_embed_t, transpose_a=False, transpose_b=True)

        cl_loss_1_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_i / temp), axis=1)))
        cl_loss_2_i = - tf.reduce_sum(
            tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_i / temp), axis=1)))

        cl_loss = cl_loss_1_u + cl_loss_2_u + cl_loss_1_i + cl_loss_2_i

        return cl_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    cores = multiprocessing.cpu_count() // 3
    Ks = np.arange(1, 21)

    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['decay'] = decay
    config['n_layers'] = n_layers
    config['embed_size'] = embed_size
    config['lr'] = lr
    config['batch_size'] = batch_size

    """
    ################################################################################
    Generate the Laplacian matrix.
    """
    norm_left, norm_3, norm_4, norm_5 = data_generator.get_adj_mat()

    if dataset == 'amazon-beauty' or dataset == 'Art':
        config['norm_adj'] = norm_4
        config['norm_adj_m'] = norm_4
    else:
        config['norm_adj'] = norm_3
        config['norm_adj_m'] = norm_3

    print('shape of adjacency', norm_left.shape)

    t0 = time.time()

    if textual_enable == True:
        model = Model(data_config=config,
                      img_feat=data_generator.imageFeaMatrix,
                      text_feat=data_generator.textFeatMatrix, d1=4096, d2=300)
    else:
        model = Model(data_config=config,
                      img_feat=data_generator.imageFeaMatrix,
                      text_feat=data_generator.imageFeaMatrix, d1=4096, d2=4096)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    ################################################################################
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
    max_epoch = 0
    early_stopping = 0

    best_score = 0
    best_result = {}
    all_result = {}

    for epoch in range(500):
        t1 = time.time()
        loss, mf_loss, emb_loss, cl_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_u()

            _, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt_1, model.mf_loss, model.emb_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items
                           })
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

            _, batch_cl_loss = sess.run(
                [model.opt_2, model.cl_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items
                           })
            cl_loss += batch_cl_loss

        if np.isnan(mf_loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % interval != 0:
            perf_str = 'Epoch {} [{:.1f}s]: train==[{:.5f} + {:.5f}]'.format(
                epoch, time.time() - t1,
                mf_loss + emb_loss, cl_loss / data_generator.n_train)
            print(perf_str)
            continue

        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())

        result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
        hr = result['hit_ratio']
        ndcg = result['ndcg']

        score = hr[4] + ndcg[4]
        if score > best_score:
            best_score = score
            best_result['hr'] = [str(i) for i in hr]
            best_result['ndcg'] = [str(i) for i in ndcg]
            print('best result until now: hr@5,10,20={:.4f},{:.4f},{:.4f},ndcg@5,10,20={:.4f},{:.4f},{:.4f}'.format(
                hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19]))
            early_stopping = 0
        else:
            early_stopping += 1

        t3 = time.time()

        perf_str = 'Epoch {} [{:1f}s + {:1f}s]: hit@5=[{:5f}],hit@10=[{:5f}],hit@20=[{:5f}],ndcg@5=[{:5f}],ndcg@10=[{:5f}],ndcg@20=[{:5f}]'.format(epoch, t2 - t1, t3 - t2,
                    hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19])
        print(perf_str)

        all_result[epoch + 1] = result
        if early_stopping == 10:
            break
    print('###########################################################################################################################')
    print('[{}], best result: hr@5,10,20={},{},{},ndcg@5,10,20={},{},{}'.format(dataset,
        best_result['hr'][4], best_result['hr'][9], best_result['hr'][19], best_result['ndcg'][4],
        best_result['ndcg'][9], best_result['ndcg'][19]))