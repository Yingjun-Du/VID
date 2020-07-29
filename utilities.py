import tensorflow as tf
import numpy as np
import cv2

def sample_normal(mu, log_variance, num_samples):
    """
    Generate samples from a parameterized normal distribution.
    :param mu: tf tensor - mean parameter of the distribution.
    :param log_variance: tf tensor - log variance of the distribution.
    :param num_samples: np scalar - number of samples to generate.
    :return: tf tensor - samples from distribution of size num_samples x dim(mu).
    """
    shape = tf.concat([tf.constant([num_samples]), tf.shape(mu)], axis=-1)
    eps = tf.random_normal(shape, dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(log_variance))


# guided filter
def guided_filter(data, num_patches, width, height, channel):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
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
