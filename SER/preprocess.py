import json
import numpy as np
from string import punctuation


def read_dataset(s_path, t_path):
    # Initialization
    s_dict, t_dict, w_embed = dict(), dict(), dict()
    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nProcessing Source & Target Data ... \n')

    f = open(s_path, 'r')

    # Read source data and generate user & item's review dict
    while True:
        line = f.readline()
        if not line:
            break

        # Convert str to json format
        line = json.loads(line)

        try:
            user, item, review, rating = line['reviewerID'], line['asin'], line['reviewText'], line['overall']

            review = review.lower()
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        s_data.append([user, item, rating])

        if user in s_dict:
            s_dict[user].append([item, review])
        else:
            s_dict[user] = [[item, review]]

        if item in s_dict:
            s_dict[item].append([user, review])
        else:
            s_dict[item] = [[user, review]]
    f.close()

    # For the separation of train / valid / test data in a target domain
    f = open(t_path, 'r')
    while True:
        len_t_data += 1
        line = f.readline()
        if not line:
            break

    len_train_data = int(len_t_data * 0.8)
    len_t_data = int(len_t_data * 0.2)
    f.close()

    # Read target domain's data
    f = open(t_path, 'r')
    while True:
        line = f.readline()
        if not line:
            break

        line = json.loads(line)

        try:
            user, item, review, rating = line['reviewerID'], line['asin'], line['reviewText'], line['overall']

            review = review.lower()
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        if user in t_dict and item in t_dict and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])
        else:
            if len(t_train) > len_train_data:
                break

            t_train.append([user, item, rating])

            if user in t_dict:
                t_dict[user].append([item, review])
            else:
                t_dict[user] = [[item, review]]
            if item in t_dict:
                t_dict[item].append([user, review])
            else:
                t_dict[item] = [[user, review]]

    f.close()

    # Split valid / test data
    t_test, t_valid = t_valid[int(
        len_t_data/2):len_t_data], t_valid[0:int(len_t_data/2)]

    print('Size of Train / Valid / Test data  : %d / %d / %d' %
          (len(t_train), len(t_valid), len(t_test)))

    # Dictionary for word embedding
    f = open('./glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    return s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed
