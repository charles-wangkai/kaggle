import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 105000
BATCH_SIZE = 16
EPSILON = 1e-9

sess = tf.InteractiveSession()

data = pd.read_csv('../data/train.csv')

images = data.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

labels_flat = data[[0]].values.ravel()
print('labels_flat({0})'.format(len(labels_flat)))

labels_count = np.unique(labels_flat).shape[0]
print('labels_count => {0}'.format(labels_count))

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('labels({0[0]},{0[1]})'.format(labels.shape))

train_images = images
train_labels = labels
print('train_images({0[0]},{0[1]})'.format(train_images.shape))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, image_size])
y_ = tf.placeholder(tf.float32, shape=[None, labels_count])

keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
keep_prob5 = tf.placeholder(tf.float32)

image = tf.reshape(x, [-1, image_width , image_height, 1])

# stage 1 : Convolution // 1x28x28 -> 64x24x24 -> 64x12x12
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
h_drop1 = tf.nn.dropout(h_pool1, keep_prob1)

# stage 2 : Convolution // 64x12x12 -> 64x12x12 -> 64x6x6
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_drop1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
h_drop2 = tf.nn.dropout(h_pool2, keep_prob2)

# stage 3 : Convolution // 64x6x6 -> 256x6x6 -> 256x3x3
W_conv3 = weight_variable([3, 3, 64, 256])
b_conv3 = bias_variable([256])

h_conv3 = tf.nn.relu(tf.nn.conv2d(h_drop2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
h_drop3 = tf.nn.dropout(h_pool3, keep_prob3)

# stage 4 : Convolution // 256x3x3 -> 1024x1x1
W_conv4 = weight_variable([3, 3, 256, 1024])
b_conv4 = bias_variable([1024])

h_conv4 = tf.nn.relu(tf.nn.conv2d(h_drop3, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
h_drop4 = tf.nn.dropout(h_conv4, keep_prob4)

# stage 5 : FC // 1024 -> 256
h_drop4_flat = tf.reshape(h_drop4, [-1, 1024])

W_fc5 = weight_variable([1024, 256])
b_fc5 = bias_variable([256])

h_fc5 = tf.nn.relu(tf.matmul(h_drop4_flat, W_fc5) + b_fc5)
h_drop5 = tf.nn.dropout(h_fc5, keep_prob5)

# stage 6 : FC // 256 -> 10
W_fc6 = weight_variable([256, labels_count])
b_fc6 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_drop5, W_fc6) + b_fc6)

# cost function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y + EPSILON))

# optimization function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prediction function
predict = tf.argmax(y, 1)

index_in_epoch = 0
num_examples = train_images.shape[0]
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > num_examples:
        # finished epoch
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

tf.initialize_all_variables().run()

display_step = 1
for i in range(TRAINING_ITERATIONS):
    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, keep_prob4: 1.0, keep_prob5: 1.0})
        print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        
        # increase display_step
        if i % (display_step * 10) == 0 and i and display_step < 1000:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: 0.9, keep_prob2: 0.8, keep_prob3: 0.7, keep_prob4: 0.6, keep_prob5: 0.5})

all_train_accuracy = accuracy.eval(feed_dict={x: train_images, y_: train_labels, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, keep_prob4: 1.0, keep_prob5: 1.0})
print('all_train_accuracy => %.4f' % all_train_accuracy)

test_images = pd.read_csv('../data/test.csv').values
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

predicted_labels = predict.eval(feed_dict={x: test_images, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, keep_prob4: 1.0, keep_prob5: 1.0})
print('predicted_labels({0})'.format(len(predicted_labels)))

np.savetxt('submission.csv',
           np.c_[range(1, len(test_images) + 1), predicted_labels],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

sess.close()
