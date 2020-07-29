import os
from CVAE import vae_decoder, vae_prior
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
from scipy import misc
import pdb
from scipy import misc
num_sample = 20

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

np.set_printoptions(threshold=np.inf)
# tf.reset_default_graph()
i = 152

# file = '../fu_image/input/' + str(i) + '_1.jpg'
file = "../test_fu_image/input/" + str(i) + '.jpg'
# file = '18.jpg'
ori = img.imread(file) / 255.0


clean_file = '../fu_image/label/' + str(i) + '.jpg'

clean = img.imread(clean_file) / 255.0

input_tensor = np.expand_dims(ori[:, :, :], axis=0)

input_tensor = np.tile(input_tensor, [num_sample,1,1,1])


num_channels = 3
images = tf.placeholder(tf.float32, shape=(num_sample, input_tensor.shape[1], input_tensor.shape[2], num_channels))
is_training = tf.placeholder(tf.bool)
images_r = images[:, :, :, :1]
images_g = images[:, :, :, 1:2]
images_b = images[:, :, :, 2:3]


with tf.variable_scope('r_out'):

    pr_sample, pr_mu, pr_var = vae_prior(images_r, is_training)
    r_out, all_r = vae_decoder(pr_sample, images_r, is_training)

with tf.variable_scope('g_out'):

    pg_sample, pg_mu, pg_var = vae_prior(images_g, is_training)
    g_out, all_g = vae_decoder(pg_sample, images_g, is_training)
with tf.variable_scope('b_out'):
    pb_sample, pb_mu, pb_var = vae_prior(images_b, is_training)
    b_out, all_b = vae_decoder(pb_sample, images_b, is_training)


with tf.variable_scope('final_out'):
    final_out = tf.concat([r_out, g_out, b_out], axis=3)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if tf.train.get_checkpoint_state('./cvae_model/'):
        ckpt = tf.train.latest_checkpoint('./cvae_model/')
        saver.restore(sess, ckpt)
        print("Loading model")

    else:
        saver.restore(sess, "./cvae_model/test-model/model")  # try a pre-trained model
        print("load pre-trained model")



    for i in range(1, 1001):
        print(i)
        file = "../test_fu_image/input/" + str(i) + '.jpg'
        # file = '18.jpg'
        ori = img.imread(file)
        ori = ori / 255.0

        input_tensor = np.expand_dims(ori[:, :, :], axis=0)
        input_tensor = np.tile(input_tensor, [num_sample, 1, 1, 1])
        num_channels = 3

        final_output, m_r, m_g, m_b = sess.run([final_out, pr_sample, pg_sample, pb_sample],
                                                feed_dict={images: input_tensor, is_training:False})
        final_output = np.mean(final_output, axis=0)
        final_output[np.where(final_output < 0.)] = 0.
        final_output[np.where(final_output > 1.)] = 1.

        derained = final_output
        plt.imsave('./test_prior_results/' + str(i) + '.jpg', derained)