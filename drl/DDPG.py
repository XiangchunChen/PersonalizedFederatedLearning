
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import logging

# tf.compat.v1.disable_eager_execution()
np.random.seed(1)

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.001  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 100
BATCH_SIZE = 64
OUTPUT_GRAPH = False



###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.a_cost = []
        self.c_cost = []
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def plot_cost(self):
        f1 = open("result/a_cost.csv", "w")
        for i in range(len(self.a_cost)):
            f1.write(str(i)+","+str(self.a_cost[i])+"\n")
        f1.close()

        f1 = open("result/c_cost.csv", "w")
        for i in range(len(self.c_cost)):
            f1.write(str(i)+","+str(self.c_cost[i])+"\n")
        f1.close()

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]


    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        a_cost = self.sess.run(self.atrain, {self.S: bs})
        self.a_cost.append(a_cost)
        c_cost = self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.c_cost.append(c_cost)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l5', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l4', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def save_net(self):
        saver = tf.train.Saver()
        now = "DDPGTO"
        # time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        fname="models/"+now+".ckpt"
        # fname = "baselinemodels/DRLTOmodel.ckpt"
        save_path = saver.save(self.sess, fname)
        print("Save to path: ", save_path)

    def restore_net(self):
        saver = tf.train.Saver()
        now = "DDPGTO"
        # time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        fname="models/"+now+".ckpt"
        # fname = "baselinemodels/DRLTOmodel.ckpt"
        saver.restore(self.sess, fname)
        print("Model restored.")
        print('Initialized')
