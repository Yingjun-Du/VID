

import os
import DDN as Network
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

        
def guided_filter(data, height,width):
    r = 15
    eps = 1.0
    batch_size = 1
    channel = 3
    batch_q = np.zeros((batch_size, height, width, channel))
    for i in range(batch_size):
        for j in range(channel):
            I = data[i, :, :,j] 
            p = data[i, :, :,j] 
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q 
    return batch_q


# file ="../test_fu_image/input/1.jpg"
file = '../single_real_image/' + str(52) + '.jpg'
# file = '18.jpg'
ori = img.imread(file)
ori = ori/255.0

input_tensor = np.expand_dims(ori[:,:,:], axis = 0)
detail_layer = input_tensor - guided_filter(input_tensor, input_tensor.shape[1], input_tensor.shape[2])


num_channels = 3
image = tf.placeholder(tf.float32, shape=(1, input_tensor.shape[1], input_tensor.shape[2], num_channels))
detail = tf.placeholder(tf.float32, shape=(1, input_tensor.shape[1], input_tensor.shape[2], num_channels))

output = Network.inference(image, detail, is_training = False)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    
    if tf.train.get_checkpoint_state('./model/'):  
        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, ckpt)
        print ("Loading model")

    else:
        saver.restore(sess, "./model/test-model/model") # try a pre-trained model 
        print ("load pre-trained model")

    for i in range(0,1):
        print(i)
        # file = '../testing_our_our/' + str(i) + '.jpg'
        file = '../single_real_image/' + str(0) + '.jpg'
        # file = '18.jpg'
        ori = img.imread(file)
        ori = ori / 255.0

        input_tensor = np.expand_dims(ori[:, :, :], axis=0)
        detail_layer = input_tensor - guided_filter(input_tensor, input_tensor.shape[1], input_tensor.shape[2])

        num_channels = 3


        final_output  = sess.run(output, feed_dict={image: input_tensor, detail: detail_layer})

        final_output[np.where(final_output < 0. )] = 0.
        final_output[np.where(final_output > 1. )] = 1.
        derained = final_output[0,:,:,:]
        plt.imsave('./DDN_true_results/'+str(0)+'.png', derained)
