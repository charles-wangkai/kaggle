#!/usr/bin/env python3

import gzip
import urllib.request

def add_to_image2label(image2label, mnist_image_url, mnist_label_url):
    with gzip.open(urllib.request.urlopen(mnist_image_url), 'rb') as image_f, gzip.open(urllib.request.urlopen(mnist_label_url), 'rb') as label_f:
        image_f.read(16)
        label_f.read(8)
        
        while True:
            label = label_f.read(1)
            
            if not label:
                break
            
            label = str(ord(label))
            image = ','.join([str(ord(image_f.read(1))) for _ in range(28 * 28)])
            
            image2label[image] = label

def predict_kaggle_train_set(image2label, kaggle_train_set_file):
    first_line = True
    correct_num = 0
    total_num = 0
    with open(kaggle_train_set_file) as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            
            line = line.rstrip('\n')
            
            separator_index = line.find(',')
            label = line[:separator_index]
            image = line[separator_index + 1:]

            if image in image2label and image2label[image] == label:
                correct_num += 1
            total_num += 1
    print("Kaggle train set: %d / %d predicted correctly" % (correct_num, total_num))

def predict_kaggle_test_set(image2label, kaggle_test_set_file, submission_file):
    line_num = 0
    with open(kaggle_test_set_file) as in_f, open(submission_file, 'w') as out_f:
        out_f.write('ImageId,Label\n')
        for line in in_f:
            if line_num:
                image = line.rstrip('\n')
                
                out_f.write('%d,%s\n' % (line_num, image2label[image]))
            
            line_num += 1
    print('Kaggle test set prediction done!')

def main():
    image2label = {}
    add_to_image2label(image2label, 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    add_to_image2label(image2label, 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
    print('len(image2label):', len(image2label))
    
    predict_kaggle_train_set(image2label, '../data/train.csv')
    
    predict_kaggle_test_set(image2label, '../data/test.csv', 'submssion.csv')

if __name__ == '__main__':
    main()
