"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import time
import random
import numpy as np
import tensorflow as tf
from setup_inception import ImageNet, InceptionModel


class GenAttack2(object):
    def mutation_op(self,  cur_pop, idx, step_noise=0.01, p=0.005):
        perturb_noise = tf.random_uniform(cur_pop.get_shape(),
                                          minval=-step_noise, maxval=step_noise, dtype=tf.float32)
        mutated_pop = perturb_noise * \
            tf.cast(tf.random_uniform(cur_pop.get_shape())
                    < p, tf.float32) + cur_pop
        return mutated_pop

    def attack_step(self, idx, success, orig_copies, cur_noise, prev_elite, margin_log, best_win_margin, cur_plateau_count, num_plateaus):
        if self.resize_dim:
            noise_resized = tf.image.resize_bilinear(
                cur_noise, (self.image_dim, self.image_dim))
        else:
            noise_resized = cur_noise
        noise_dim = self.resize_dim or self.image_dim
        cur_pop = tf.clip_by_value(
            noise_resized + orig_copies, self.box_min, self.box_max)
        pop_preds = self.model.predict(cur_pop)
        all_preds = tf.argmax(pop_preds, axis=1)

        success_pop = tf.cast(tf.equal(all_preds, self.target), tf.int32)
        success = tf.reduce_max(success_pop, axis=0)

        target_scores = tf.reduce_sum(self.tlab * pop_preds, axis=1)
        sum_others = tf.reduce_sum((1-self.tlab) * pop_preds, axis=1)
        max_others = tf.reduce_max((1-self.tlab) * pop_preds, axis=1)

        # the goal is to maximize this loss
        loss = -(tf.log(sum_others+1e-30) - tf.log(target_scores+1e-30))

        win_margin = tf.reduce_max(
            pop_preds[:, self.target] - tf.reduce_max(pop_preds, axis=1))

        new_best_win_margin, new_cur_plateau_count = tf.cond(
            tf.greater(win_margin, best_win_margin),
            false_fn=lambda: (best_win_margin, cur_plateau_count+1),
            true_fn=lambda: (win_margin, 0)
        )
        plateau_threshold = tf.cond(tf.greater(win_margin, -0.40),
                                    true_fn=lambda: 100,
                                    false_fn=lambda: 300)
        new_num_plateaus, new_cur_plateau_count = tf.cond(
            tf.greater(new_cur_plateau_count, plateau_threshold),
            true_fn=lambda: (num_plateaus+1, 0),
            false_fn=lambda: (num_plateaus, new_cur_plateau_count)
        )

        if self.adaptive:
            step_noise = tf.maximum(self.alpha,
                                    0.4*tf.pow(0.9, tf.cast(new_num_plateaus, tf.float32)))
            step_p = tf.cond(tf.less(idx, 10),
                             true_fn=lambda: 1.0,
                             false_fn=lambda: tf.maximum(self.mutation_rate, 0.5*tf.pow(0.90, tf.cast(new_num_plateaus, tf.float32))))
        else:
            step_noise = self.alpha
            step_p = self.mutation_rate

        step_temp = self.temp

        elite_idx = tf.cond(
            tf.equal(success, 1),
            true_fn=lambda: tf.expand_dims(
                tf.cast(tf.argmax(success_pop), tf.int32), axis=0),
            false_fn=lambda: tf.expand_dims(tf.cast(tf.argmax(loss, axis=0), tf.int32), axis=0))

        elite = tf.gather(cur_noise, elite_idx)
        select_probs = tf.nn.softmax(tf.squeeze(loss) / step_temp)
        parents = tf.distributions.Categorical(
            probs=select_probs).sample(2*self.pop_size-2)
        parent1 = tf.gather(cur_noise, parents[:self.pop_size-1])
        parent2 = tf.gather(cur_noise, parents[self.pop_size-1:])
        pp1 = tf.gather(select_probs, parents[:self.pop_size-1])
        pp2 = tf.gather(select_probs, parents[self.pop_size-1:])
        pp2 = pp2 / (pp1+pp2)
        pp2 = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.expand_dims(pp2, 1), 2), self.image_channels), (1, noise_dim, noise_dim, self.image_channels))
        xover_prop = tf.cast(tf.random_uniform(
            shape=parent1.get_shape()) > pp2, tf.float32)
        childs = parent1 * xover_prop + parent2 * (1-xover_prop)
        idx = tf.Print(idx+1, [idx, tf.reduce_min(loss),
                               win_margin, step_p, step_noise, new_cur_plateau_count])
        margin_log = tf.concat([margin_log, [[win_margin]]], axis=0)
        mutated_childs = self.mutation_op(
            childs, idx=idx, step_noise=self.eps*step_noise, p=step_p)
        new_pop = tf.concat((mutated_childs, elite), axis=0)
        return idx, success, orig_copies, new_pop, tf.reshape(elite, (noise_dim, noise_dim, self.image_channels)), margin_log, new_best_win_margin, new_cur_plateau_count, new_num_plateaus

    def __init__(self, model, pop_size=6, mutation_rate=0.001,
                 eps=0.15, max_steps=10000, alpha=0.20,
                 image_dim=299,
                 image_channels=3,
                 num_labels=1001,
                 temp=0.3,
                 resize_dim=None, adaptive=False):
        self.eps = eps
        self.pop_size = pop_size
        self.model = model
        self.alpha = alpha
        self.temp = temp
        self.max_steps = max_steps
        self.mutation_rate = mutation_rate
        self.image_dim = image_dim
        self.resize_dim = resize_dim
        noise_dim = self.resize_dim or self.image_dim
        self.image_channels = image_channels
        self.num_labels = num_labels
        self.adaptive = adaptive
        self.writer = tf.summary.FileWriter(logdir='.')
        self.input_img = tf.Variable(
            np.zeros((1, self.image_dim, self.image_dim, self.image_channels), dtype=np.float32), name='x', dtype=tf.float32)
        # copies of original image
        self.pop_orig = tf.Variable(np.zeros(
            (self.pop_size, self.image_dim, self.image_dim, image_channels), dtype=np.float32), name='pop_orig', dtype=tf.float32)
        self.pop_noise = tf.Variable(np.zeros(
            (self.pop_size, noise_dim, noise_dim, self.image_channels), dtype=np.float32), name='pop_noise', dtype=tf.float32)

        self.target = tf.Variable(0, dtype=tf.int64, name='target')
        self.init_success = tf.Variable(0, dtype=tf.int32, name='success')
        self.box_min = tf.tile(tf.maximum(
            self.input_img-eps, -0.5), (self.pop_size, 1, 1, 1))
        self.box_max = tf.tile(tf.minimum(
            self.input_img+eps, 0.5), (self.pop_size, 1, 1, 1))
        self.margin_log = tf.Variable(initial_value=np.zeros(
            (1, 1), dtype=np.float32), validate_shape=False, name='margin_log', dtype=tf.float32)
        self.margin_log.set_shape((None, 1))
        self.tlab = tf.contrib.layers.one_hot_encoding(
            [self.target], num_classes=self.num_labels)
        self.i = tf.Variable(0, dtype=tf.int64, name='step')

        # Variables to detect plateau
        self.best_win_margin = tf.Variable(-1,
                                           dtype=tf.float32, name='cur_margin')
        self.cur_plateau_count = tf.Variable(0, dtype=tf.int32, name='plateau')
        self.num_plateaus = tf.Variable(0, dtype=tf.int32, name='num_plateaus')

        def cond(i, success, pop_orig, pop_noise, cur_elite, margin_log, best_win_margin, cur_plateau_count,
                 num_plateaus): return tf.logical_and(tf.less_equal(i, self.max_steps), tf.equal(success, 0))

        def attack_body(i, success, pop_orig, pop_noise, cur_elite, margin_log, best_win_margin, cur_plateau_count, num_plateaus): return self.attack_step(
            i, success, pop_orig, pop_noise, cur_elite, margin_log, best_win_margin, cur_plateau_count, num_plateaus)

        self.attack_main = tf.while_loop(cond, attack_body, [self.i, self.init_success,  self.pop_orig,
                                                             self.pop_noise, self.pop_noise[0], self.margin_log, self.best_win_margin, self.cur_plateau_count, self.num_plateaus])
        self.summary_op = tf.summary.merge_all()

    def initialize(self, sess, img, target):
        sess.run([x.initializer for x in
                  [self.i,
                   self.input_img,
                   self.target,
                   self.margin_log,
                   self.pop_noise,
                   self.best_win_margin,
                   self.cur_plateau_count,
                   self.num_plateaus,
                   self.init_success]])
        sess.run(tf.assign(self.input_img, np.expand_dims(img, axis=0)))
        sess.run(tf.assign(self.target, target))
        orig_copies = tf.tile(self.input_img, [self.pop_size, 1, 1, 1])
        sess.run(tf.assign(self.pop_orig, orig_copies))
        init_noise = self.mutation_op(
            self.pop_noise, idx=self.i, p=self.mutation_rate, step_noise=self.eps)
        sess.run(tf.assign(self.margin_log, np.zeros((1, 1), dtype=np.float32)))
        sess.run(tf.assign(self.pop_noise, init_noise))
        sess.run(tf.assign(self.best_win_margin,
                           np.array(-1.0, dtype=np.float32)))
        sess.run(tf.assign(self.cur_plateau_count, np.array(0, dtype=np.int32)))
        sess.run(tf.assign(self.num_plateaus, 0))
        print('Population initailized')

    def attack(self, sess, input_img, target_label):
        self.initialize(sess, input_img, target_label)
        (num_steps, success,  copies, final_pop, adv_noise,
         log_hist, _, _, _) = sess.run(self.attack_main)
        if success:
            if self.resize_dim:
                adv_img = sess.run(
                    tf.clip_by_value(
                        np.expand_dims(input_img, axis=0)+tf.image.resize_bilinear(
                            np.expand_dims(adv_noise, axis=0), (self.image_dim, self.image_dim)),
                        self.box_min[0:1], self.box_max[0:1]))
            else:
                adv_img = sess.run(
                    tf.clip_by_value(np.expand_dims(input_img, axis=0)+np.expand_dims(adv_noise, axis=0),
                                     self.box_min[0:1], self.box_max[0:1]))

            # Number of queries = NUM_STEPS * (POP_SIZE -1 ) + 1
            # We subtract 1 from pop_size, because we use elite mechanism, so one population
            # member is copied from previous generation and no need to re-evaluate it.
            # The first population is an exception, therefore we add 1 to have total sum.
            query_count = num_steps * (self.pop_size - 1) + 1
            return adv_img[0], query_count, log_hist[1:, :]
        else:
            return None
