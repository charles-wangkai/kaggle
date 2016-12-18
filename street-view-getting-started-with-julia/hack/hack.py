#!/usr/bin/env python3

import csv
import os
import tarfile
import tempfile
import urllib.request

CHARS74K_URL = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz'
EXTRACTED_DIR = tempfile.gettempdir()

ORIGINAL_IMAGE_SUFFIX = '.png'
KAGGLE_IMAGE_SUFFIX = '.Bmp'

ORIGINAL_NAME_2_CORRECT_LABEL = {'img015-00376.png': '1'}

def convert_to_label(dir_name):
    sequence = int(dir_name[len('Sample'):])
    if sequence <= 10:
        return chr(sequence - 1 + ord('0'))
    elif sequence <= 36:
        return chr(sequence - 11 + ord('A'))
    else:
        return chr(sequence - 37 + ord('a'))

def read_file_content(file_path):
    return open(file_path, 'rb').read()

def add_to_image2label(image2label, folder):
    for entry in os.scandir(folder):
        if entry.is_dir():
            label = convert_to_label(entry.name)
            for file in os.scandir(entry.path):
                if file.name.endswith(ORIGINAL_IMAGE_SUFFIX):
                    image2label[read_file_content(file.path)] = label if file.name not in ORIGINAL_NAME_2_CORRECT_LABEL else ORIGINAL_NAME_2_CORRECT_LABEL[file.name]

def build_image2label():
    image2label = {}
    
    local_filename = urllib.request.urlretrieve(CHARS74K_URL)[0]
    print('Finished file download:', local_filename)
    
    tar = tarfile.open(local_filename, "r:gz")
    tar.extractall(EXTRACTED_DIR)
    tar.close()
    print('Finished file extraction:', EXTRACTED_DIR)
    
    add_to_image2label(image2label, EXTRACTED_DIR + '/English/Img/BadImag/Bmp')
    add_to_image2label(image2label, EXTRACTED_DIR + '/English/Img/GoodImg/Bmp')
    
    return image2label

def build_id2label(kaggle_train_label_file):
    id2label = {}
    
    csv_file_object = csv.reader(open(kaggle_train_label_file))
    next(csv_file_object)
    for row in csv_file_object:
        id, label = row
        id2label[id] = label
    
    return id2label

def get_id(kaggle_image_file):
    return kaggle_image_file[:-len(KAGGLE_IMAGE_SUFFIX)]

def predict_kaggle_train_set(image2label, kaggle_train_set_folder, kaggle_train_label_file):
    id2label = build_id2label(kaggle_train_label_file)
    
    correct_num = 0
    total_num = 0
    for entry in os.scandir(kaggle_train_set_folder):
        if entry.name.endswith(KAGGLE_IMAGE_SUFFIX):
            image = read_file_content(entry.path)
            id = get_id(entry.name)
            
            if image in image2label and image2label[image] == id2label[id]:
                correct_num += 1
            else:
                print('%s: original = %s, kaggle = %s' % (id, image2label[image], id2label[id]))
            total_num += 1

    print("Kaggle train set: %d / %d predicted correctly" % (correct_num, total_num))

def predict_kaggle_test_set(image2label, kaggle_test_set_folder, submission_file):
    with open(submission_file, 'w') as out_f:
        out_f.write('ID,Class\n')
        
        for entry in os.scandir(kaggle_test_set_folder):
            if entry.name.endswith(KAGGLE_IMAGE_SUFFIX):
                id = get_id(entry.name)
                image = read_file_content(entry.path)
                
                out_f.write('%s,%s\n' % (id, image2label[image]))
    
    print('Kaggle test set prediction done!')

def main():
    image2label = build_image2label()
    print('len(image2label):', len(image2label))
        
    predict_kaggle_train_set(image2label, '../data/train', '../data/trainLabels.csv')
     
    predict_kaggle_test_set(image2label, '../data/test', 'submission.csv')
    
if __name__ == '__main__':
    main()
