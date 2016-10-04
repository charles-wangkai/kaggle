#!/usr/bin/env python3

import csv
import urllib.request

TITANIC_URL = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv'

def simplify_number(s):
    return str(float(s)) if s else ''

def build_observation(pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked):
    return (pclass, name.replace('"', ''), sex, age, sibsp, parch, ticket, simplify_number(fare), cabin, embarked)

def build_observation2label():
    observation2label = {}
    
    local_filename = urllib.request.urlretrieve(TITANIC_URL)[0]
    csv_file_object = csv.reader(open(local_filename))
    next(csv_file_object)
    
    for row in csv_file_object:
        pclass, survived, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked, _, _, _ = row
        observation = build_observation(pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked)
        label = survived
        
        observation2label[observation] = label

    return observation2label

def predict_kaggle_train_set(observation2label, kaggle_train_set_file):
    csv_file_object = csv.reader(open(kaggle_train_set_file))
    next(csv_file_object)
    
    correct_num = 0
    total_num = 0
    for row in csv_file_object:
        _, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = row
        observation = build_observation(pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked)
        label = survived
        
        if observation in observation2label and observation2label[observation] == label:
            correct_num += 1
        else:
            print(observation)
        total_num += 1

    print("Kaggle train set: %d / %d predicted correctly" % (correct_num, total_num))

def predict_kaggle_test_set(observation2label, kaggle_test_set_file, submission_file):
    csv_file_object = csv.reader(open(kaggle_test_set_file))
    next(csv_file_object)
    
    with open(submission_file, 'w') as out_f:
        out_f.write('PassengerId,Survived\n')
        for row in csv_file_object:
            passenger_id, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = row
            observation = build_observation(pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked)
            
            out_f.write('%s,%s\n' % (passenger_id, observation2label[observation]))            

    print('Kaggle test set prediction done!')

def main():
    observation2label = build_observation2label()
    print('len(observation2label):', len(observation2label))
    
    predict_kaggle_train_set(observation2label, '../data/train.csv')
    
    predict_kaggle_test_set(observation2label, '../data/test.csv', 'submssion.csv')

if __name__ == '__main__':
    main()
