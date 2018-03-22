def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import tensorflow as tf
import numpy as np
from utils import get_data, next_batch
import time

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True): 
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases
    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

train_x, train_y, v_x, v_y, test_x, test_y, _ = get_data()
test_img_y = np.argmax(test_y, axis=1)
v_img_y = np.argmax(v_y, axis=1)

filter_size1 = 7          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.
fc_size = 1024             # Number of neurons in fully-connected layer.
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
img_shape = train_x[0].shape
num_classes = 10

training = tf.placeholder(tf.bool, name='training')
x = tf.placeholder(tf.float32, shape=[None,img_size, img_size, num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
# print(layer_conv1)						# after Max-pooling(halved)
layer_conv_nm = tf.layers.batch_normalization(layer_conv1, training=training)
# print(layer_conv_nm)
layer_flat, num_features = flatten_layer(layer_conv_nm)
# print(layer_flat,num_features)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
# print(layer_fc1)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()

train_batch_size = 256
total_iterations = 0
test_batch_size = 256
v_batch_size = 256

def print_valid_accuracy():
    num_test = len(v_x)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + v_batch_size, num_test)
        images = np.array(v_x)[i:j, :]
        labels = np.array(v_y)[i:j, :]
        feed_dict = {x: images,y_true: labels, training: True}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = v_img_y

    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

def optimize(num_epochs):
    epoch = num_epochs
    iteration = (int)(len(train_x) / train_batch_size)

    for j in range(epoch):
            for i in range(iteration):
                        x_batch,y_true_batch = next_batch(train_batch_size, train_x, train_y)   
                        feed_dict_train = {x: x_batch,y_true: y_true_batch, training: True}             #, training=True
                        session.run(optimizer, feed_dict=feed_dict_train)
                        # print('Iteration: ',i,', Epoch: ',j)
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print('Optimization Epoch: ',j+1,', Training Accuracy: ', round(acc*100,1))
            print_valid_accuracy()
            # if j%20 == 0:
            #     saver.save(session, "model_1",global_step=j)


def print_test_accuracy():
    num_test = len(test_x)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = np.array(test_x)[i:j, :]
        labels = np.array(test_y)[i:j, :]
        feed_dict = {x: images,y_true: labels, training: True}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = test_img_y

    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

epochs = int(input('Number of Epochs: '))
optimize(epochs)
print_test_accuracy()

save_path = saver.save(session, "model-30")
print("Model saved in path: %s" % save_path)

session.close()

