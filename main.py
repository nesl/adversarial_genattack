"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import time
import os
import sys
import random
import numpy as np

import tensorflow as tf
from setup_inception import ImageNet, InceptionModel
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel

import utils
from genattack_tf2 import GenAttack2

flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path for input images.')
flags.DEFINE_string('output_dir', 'output', 'Path to save results.')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_bool('verbose', True, 'Print logs.')
flags.DEFINE_integer('test_example', default=None, help='Test only one image')

flags.DEFINE_float('mutation_rate', default=0.005, help='Mutation rate')
flags.DEFINE_float('eps', default=0.10,
                   help='maximum L_inf distance threshold')
flags.DEFINE_float('alpha', default=0.20, help='Step size')
flags.DEFINE_float('temp', default=0.1,
                   help='sampling temperature for selection')
flags.DEFINE_integer('pop_size', default=6, help='Population size')
flags.DEFINE_integer('max_steps', default=10000,
                     help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', None,
                     'Reduced dimension for dimensionality reduction')
flags.DEFINE_bool('adaptive', True,
                  'Turns on the dynamic scaling of mutation prameters')
flags.DEFINE_string('model', 'inception', 'model name')
flags.DEFINE_integer(
    'target', None, 'target class. if not provided will be random')
FLAGS = flags.FLAGS

if __name__ == '__main__':

    with tf.Session() as sess:
        if FLAGS.model == 'inception':
            assert FLAGS.input_dir is not None, 'You must provide input_dir.'
            dataset = ImageNet(FLAGS.input_dir)
            inputs, targets, reals = utils.generate_data(
                dataset, FLAGS.test_size)
            image_dim = 299
            image_channels = 3
            num_labels = 1001
            model = InceptionModel(sess, use_log=True)
        elif FLAGS.model == 'mnist':
            dataset = MNIST()
            model = MNISTModel('models/mnist', sess, use_log=True)
            image_dim = 28
            image_channels = 1
            num_labels = 10
            inputs, targets, reals = utils.generate_data(
                dataset, FLAGS.test_size)
            assert FLAGS.resize_dim is None, 'Dimensionality reduction of noise is used only for ImageNet models'
        elif FLAGS.model == 'cifar':
            dataset = CIFAR()
            model = CIFARModel('models/cifar', sess, use_log=True)
            image_dim = 32
            image_channels = 3
            num_labels = 10
            inputs, targets, reals = utils.generate_data(
                dataset, FLAGS.test_size)
            assert FLAGS.resize_dim is None, 'Dimensionality reduction of noise is used only for ImageNet models'
        else:
            raise ValueError(
                'Incorrect model name provided ({})'.format(FLAGS.model))
        test_in = tf.placeholder(
            tf.float32, (1, image_dim, image_dim, image_channels), 'x')
        test_pred = tf.argmax(model.predict(test_in), axis=1)

        attack = GenAttack2(model=model,
                            pop_size=FLAGS.pop_size,
                            mutation_rate=FLAGS.mutation_rate,
                            eps=FLAGS.eps,
                            max_steps=FLAGS.max_steps,
                            alpha=FLAGS.alpha,
                            resize_dim=FLAGS.resize_dim,
                            image_dim=image_dim,
                            image_channels=image_channels,
                            num_labels=num_labels,
                            temp=FLAGS.temp,
                            adaptive=FLAGS.adaptive)
        num_valid_images = len(inputs)
        total_count = 0  # Total number of images attempted
        success_count = 0
        logger = utils.ResultLogger(FLAGS.output_dir, FLAGS.flag_values_dict())
        for ii in range(num_valid_images):
            if (FLAGS.test_example and FLAGS.test_example != ii):
                continue
            input_img = inputs[ii]
            if FLAGS.target:
                target_label = FLAGS.target + 1
            else:
                target_label = np.argmax(targets[ii])
            real_label = reals[ii]
            orig_pred = sess.run(test_pred, feed_dict={
                                 test_in: [input_img]})[0]
            if FLAGS.verbose:
                print('Real = {}, Predicted = {}, Target = {}'.format(
                    real_label, orig_pred, target_label))
            if orig_pred != real_label:
                if FLAGS.verbose:
                    print('\t Skipping incorrectly classified image.')
                continue
            total_count += 1
            start_time = time.time()
            result = attack.attack(sess, input_img, target_label)
            end_time = time.time()
            attack_time = (end_time-start_time)
            if result is not None:
                adv_img, query_count, margin_log = result
                final_pred = sess.run(test_pred, feed_dict={
                                      test_in: [adv_img]})[0]
                if (final_pred == target_label):
                    success_count += 1
                    print('--- SUCCEEEED ----')
                    if image_channels == 1:
                        input_img = input_img[:, :, 0]
                        adv_img = adv_img[:, :, 0]
                    logger.add_result(ii, input_img, adv_img, real_label,
                                      target_label, query_count, attack_time, margin_log)
            else:
                print('Attack failed')
    logger.close(num_attempts=total_count)
    print('Number of success = {} / {}.'.format(success_count, total_count))
