import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

TRAINING_ITERATIONS = 100000
BATCH_SIZE = 100

sess = tf.InteractiveSession()

# read training data from CSV file 
data = pd.read_csv('../data/train.csv')
print('data({0[0]},{0[1]})'.format(data.shape))

train_images = data.iloc[:, 1:].values
train_images = train_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
train_images = np.multiply(train_images, 1.0 / 255.0)
print('train_images({0[0]},{0[1]})'.format(train_images.shape))

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

train_labels = dense_to_one_hot(labels_flat, labels_count)
train_labels = train_labels.astype(np.uint8)
print('train_labels({0[0]},{0[1]})'.format(train_labels.shape))

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# prediction function
# [0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y, 1)

# Test trained model
correct_prediction = tf.equal(predict, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
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

# Train
tf.initialize_all_variables().run()

display_step = 1
for i in range(TRAINING_ITERATIONS):
    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})       
        print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        
        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    
    # train on batch
    train_step.run({x: batch_xs, y_: batch_ys})

print(accuracy.eval({x: train_images, y_: train_labels}))

# read test data from CSV file 
test_images = pd.read_csv('../data/test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

# predict test set
predicted_labels = predict.eval(feed_dict={x: test_images})
print('predicted_labels({0})'.format(len(predicted_labels)))

# save results
np.savetxt('submission.csv',
           np.c_[range(1, len(test_images) + 1), predicted_labels],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

sess.close()
