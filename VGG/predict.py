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
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#latest_ckp = tf.train.latest_checkpoint('model/model.ckpt')
#print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')

path = '/home/aodev/test/'
w = 224
h = 224
c = 3

model_path = "model/model.ckpt"
logs_path = 'logs'


def read_img(path):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    # print('cate: ', cate)
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
    # print('one hot: ', sess.run(one_hot_labels).shape)
    sess.close()
    return np.asarray(imgs, np.float32), one_hot_labels, num_class

data, label, num_class = read_img(path)

with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    # Restore model weights from previously saved model
    saver.restore(sess, 'model/model.ckpt')
    print(saver)
    print("Model restored from file: %s" % model_path)
    #latest_ckp = tf.train.latest_checkpoint('model/model.ckpt')
    #print(latest_ckp)
    # For getting the names of every operations
    for op in tf.get_default_graph().get_operations():
        print str(op.name) 
    # print_tensors_in_checkpoint_file(saver, all_tensors=True, tensor_name='fc8')
    input_x = sess.graph.get_tensor_by_name("input:0")
    print input_x
    out_softmax = sess.graph.get_tensor_by_name("softmax:0")
    print out_softmax
    out_label = sess.graph.get_tensor_by_name("fc8/logit:0")
    print out_label

    img_out_softmax = sess.run(out_softmax, feed_dict={input_x:data})

    print "img_out_softmax:",img_out_softmax
    prediction_labels = np.argmax(img_out_softmax, axis=1)
    print "label:",prediction_labels
