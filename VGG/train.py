from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections

path = '/home/aodev/test/'
w = 224
h = 224
c = 3

# Training Parameters
learning_rate = 0.001
num_epoch = 1
batch_size = 16
display_step = 1
model_path = "model/model.ckpt"
logs_path = 'logs'

def mkfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkfolder('model')
mkfolder(logs_path)

def read_img(path):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    num_class = idx+1
    one_hot_tensor = tf.one_hot(labels,num_class)
    sess = tf.Session()
    one_hot_labels = sess.run(one_hot_tensor)
    sess.close()
    return np.asarray(imgs, np.float32), one_hot_labels, num_class

data, label, num_class = read_img(path)
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val   = data[s:]
y_val   = label[s:]

train_len = len(x_train)
test_len = len(x_val)


def build_network(x, y):

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(input, w, b, name = None):
        return tf.add(tf.matmul(input, w),b, name = name)

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 3, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)

    pool1 = pool_max(output_conv1_2)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)

    pool3 = pool_max(output_conv3_3)

    # conv4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 256, 512])
        biases = bias_variable([512])
        output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

    with tf.name_scope('conv4_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

    pool4 = pool_max(output_conv4_3)

    # conv5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)

    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

    with tf.name_scope('conv5_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

    pool5 = pool_max(output_conv5_3)

    #fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        kernel = weight_variable([shape, 4096])
        biases = bias_variable([4096])
        pool5_flat = tf.reshape(pool5, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    #fc7
    with tf.name_scope('fc7') as scope:
        kernel = weight_variable([4096, 4096])
        biases = bias_variable([4096])
        output_fc7 = tf.nn.relu(fc(output_fc6, kernel, biases), name=scope)

    #fc8
    with tf.name_scope('fc8') as scope:
        kernel = weight_variable([4096, num_classes])
        biases = bias_variable([num_classes])
        output_fc8 = (fc(output_fc7, kernel, biases, name = 'logit'))     ## removed relu
        # output_fc8 = tf.nn.relu(fc(output_fc7, kernel, biases), name=scope)

    return output_fc8


## Construct model
height = 224
width = 224
channel = 3
num_classes = num_class 
learning_rate = 1e-4

x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels_placeholder')

logits = build_network(x, y)
prediction = tf.nn.softmax(logits, name='softmax')


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

## Start training 
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for ep in range(num_epoch):
        for i in range(int(train_len/batch_size)):
            _, train_loss, train_accuracy = sess.run([train_op, loss_op, accuracy], feed_dict = {x:x_train[i*batch_size:(i+1)*batch_size],
                               y: y_train[i*batch_size:(i+1)*batch_size]})
            if i%display_step==0:
                print('Epoch: {}, Steps: {}, Training loss: {}, Training accuracy: {}'.format(ep, i, train_loss,train_accuracy))
                test_loss, test_accuracy = sess.run([loss_op, accuracy], feed_dict = {x:x_val, y:y_val})
                print('Epoch: {}, Steps: {}, Test loss: {}, Test accuracy: {}'.format(ep, i, test_loss,test_accuracy))
    
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    print('Optimization Finished')

    test_loss, test_accuracy = sess.run([loss_op, accuracy], feed_dict = {x:x_val, y:y_val})
    print('Final Test loss: {}, Final Test accuracy: {}'.format(test_loss,test_accuracy))
