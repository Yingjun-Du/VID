
import os
import h5py
import re
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import matplotlib.pyplot as plt
################ batch normalization setting ################
MOVING_AVERAGE_DECAY = 0.9997
BN_EPSILON = 0.001
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_OPS_COLLECTION = 'Derain_update_ops'
DERAIN_VARIABLES = 'Derain_variables'
#############################################################

os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # select GPU device

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_h5_file', 2000,
                            """number of training h5 files.""")
tf.app.flags.DEFINE_integer('num_patches', 500,
                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          """learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 3,
                            """epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_channels', 3,
                            """Number of the input's channels.""")
tf.app.flags.DEFINE_integer('image_size', 80,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 80,
                            """Size of the labels.""")
tf.app.flags.DEFINE_string("data_path", "../fu_h5data/", "The path of h5 files")

tf.app.flags.DEFINE_string("save_model_path", "./model/", "The path of saving model")


# read h5 files
def read_data(file):
    with h5py.File(file, 'r') as hf:
        data = hf.get('data')
        label = hf.get('label')
        return np.array(data), np.array(label)


# guided filter
def guided_filter(data, num_patches=FLAGS.num_patches, width=FLAGS.image_size, height=FLAGS.image_size,
                  channel=FLAGS.num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)  # 使用均值滤波
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :, j] = q
    return batch_q


# get variables for batch normalization
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, DERAIN_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


# batch normalization
def bn(x, c):
    x_shape = x.get_shape()  # tensor的shape
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x


# initialize weights
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


# network structure
def inference(images, detail, is_training):
    c = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')

    #  layer 1
    with tf.variable_scope('conv_1'):
        kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')

        conv1 = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.bias_add(conv1, biases)

        bias1 = bn(bias1, c)

        conv_shortcut = tf.nn.relu(bias1)

    # layers 2 to 25
    for i in range(12):
        with tf.variable_scope('conv_%s' % (i * 2 + 2)):
            kernel = create_kernel(name=('weights_%s' % (i * 2 + 2)), shape=[3, 3, 16, 16])
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True,
                                 name=('biases_%s' % (i * 2 + 2)))

            conv_tmp1 = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
            bias_tmp1 = tf.nn.bias_add(conv_tmp1, biases)

            bias_tmp1 = bn(bias_tmp1, c)

            out_tmp1 = tf.nn.relu(bias_tmp1)

        with tf.variable_scope('conv_%s' % (i * 2 + 3)):
            kernel = create_kernel(name=('weights_%s' % (i * 2 + 3)), shape=[3, 3, 16, 16])
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True,
                                 name=('biases_%s' % (i * 2 + 3)))

            conv_tmp2 = tf.nn.conv2d(out_tmp1, kernel, [1, 1, 1, 1], padding='SAME')
            bias_tmp2 = tf.nn.bias_add(conv_tmp2, biases)

            bias_tmp2 = bn(bias_tmp2, c)

            bias_tmp2 = tf.nn.relu(bias_tmp2)
            conv_shortcut = tf.add(conv_shortcut, bias_tmp2)

    # layer 26
    with tf.variable_scope('conv_26'):
        kernel = create_kernel(name='weights_26', shape=[3, 3, 16, FLAGS.num_channels])
        biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True,
                             name='biases_26')

        conv_final = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
        bias_final = tf.nn.bias_add(conv_final, biases)

        neg_residual = bn(bias_final, c)

        final_out = tf.nn.relu(tf.add(images, neg_residual))

    return final_out


if __name__ == '__main__':

    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # data
    details = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # label
    labels = tf.placeholder(tf.float32,
                            shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))  # detail layer
    is_training = tf.placeholder(tf.bool)
    outputs = inference(images, details, is_training=is_training)

    loss = tf.reduce_mean(tf.square(labels - outputs))  # MSE loss

    lr_ = FLAGS.learning_rate  # learning rate
    lr = tf.placeholder(tf.float32, shape=[])

    g_optim = tf.train.AdamOptimizer(lr).minimize(loss)  # Optimization method: Adam
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    train_op = tf.group(g_optim, batchnorm_updates_op)

    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU setting
    config.gpu_options.allow_growth = True

    data_path = FLAGS.data_path
    save_path = FLAGS.save_model_path
    epoch = int(FLAGS.epoch)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        validation_data_name = "validation.h5"
        validation_data, validation_label = read_data(data_path + validation_data_name)  # data for validation
        validation_detail = validation_data - guided_filter(validation_data)  # detail layer for validation

        if tf.train.get_checkpoint_state('./model/'):  # load previous trained models
            ckpt = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d", ckpt)
            if len(ckpt_num) == 2:
                start_point = 10 * int(ckpt_num[0]) + int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success")

        else:  # re-training if no previous trained models
            print("re-training")
            start_point = 0

        for j in range(start_point, epoch):  # the number of epoch

            if j + 1 > 1:  # reduce learning rate
                lr_ = FLAGS.learning_rate * 0.1
            if j + 1 > 2:
                lr_ = FLAGS.learning_rate * 0.01

            Training_Loss = 0.

            for h5_num in range(FLAGS.num_h5_file):  # the number of h5 files
                train_data_name = "train" + str(h5_num + 1) + ".h5"
                train_data, train_label = read_data(data_path + train_data_name)  # data for training
                detail_data = train_data - guided_filter(train_data)  # detail layer for training

                data_size = int(FLAGS.num_patches / FLAGS.batch_size)  # the number of batch
                train_loss = 0.0
                for batch_num in range(data_size):
                    rand_index = np.arange(int(batch_num * FLAGS.batch_size), int((batch_num + 1) * FLAGS.batch_size))
                    batch_data = train_data[rand_index, :, :, :]
                    batch_detail = detail_data[rand_index, :, :, :]
                    batch_label = train_label[rand_index, :, :, :]

                    _, lossvalue = sess.run([train_op, loss], feed_dict={images: batch_data, details: batch_detail,
                                                                         labels: batch_label, lr: lr_, is_training:True})
                    Training_Loss += lossvalue  # training loss
                    train_loss +=lossvalue
                print(
                    'training %d epoch, %d / %d h5 files are finished, learning rate = %.4f, Training_Loss = %.4f' %
                    (j + 1, h5_num + 1, FLAGS.num_h5_file, lr_, train_loss))
                model_name = 'model-epoch'  # save model
                save_path_full = os.path.join(save_path, model_name)
                saver.save(sess, save_path_full, global_step=h5_num)



            Training_Loss /= (data_size * FLAGS.num_h5_file)

            Validation_Loss = sess.run(loss, feed_dict={images: validation_data[0:FLAGS.batch_size, :, :, :],
                                                       details: validation_detail[0:FLAGS.batch_size, :, :, :],
                                                        labels: validation_label[0:FLAGS.batch_size, :, :,
                                                                :], is_training:False})  # validation loss

            print('%d epoch is finished, Training_Loss = %.4f, Validation_Loss = %.4f' % (
                j + 1, Training_Loss, Validation_Loss))


