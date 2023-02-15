import numpy as np
import math
from collections import Counter


def get_information_entropy(dataset):
    dataset = np.array(dataset)
    dataset_labels = Counter(dataset[:, -1])
    dataset_len = len(dataset)
    information_entropy = 0
    for _, num in dataset_labels.items():
        prob = num / dataset_len
        information_entropy -= prob * math.log(prob, 2)
    return information_entropy


def get_information_gain(dataset, idx, threshold):
    dataset = np.array(dataset)
    dataset_len = len(dataset)
    entropy = get_information_entropy(dataset)

    split_data = split_dataset(dataset, idx, threshold)
    ## print("length of split_data is ", len(split_data[0]),len(split_data[1]))
    conditional_entropy = 0
    for data in split_data:
        if len(data) == 0:
            continue
        p = len(data) / dataset_len
        conditional_entropy += p * get_information_entropy(data)
    return entropy - conditional_entropy, conditional_entropy


def get_information_gain_ratio(dataset, idx, threshold=None):
    split_information = 0
    dataset_len = len(dataset)
    split_data = split_dataset(dataset, idx, threshold)
    information_gain, conditional_entropy = get_information_gain(dataset, idx, threshold)
    for data in split_data:
        p = len(data) / dataset_len
        if p == 0:
            continue
        split_information -= p * math.log(p, 2)
    return information_gain / (split_information+1e-5), conditional_entropy, information_gain, split_information


def split_dataset(dataset, idx, threshold=None):
    bigger_set = []
    smaller_set = []
    for data in dataset:
        if data[idx] >= threshold:
            # print("now data is ", data)
            bigger_set.append(data)
        else:
            smaller_set.append(data)
    return list([bigger_set, smaller_set])


def find_best_split(dataset):
    best_threshold = None
    best_idx = -1
    best_gain_ratio = 0.0
    dimension_num = len(dataset[0]) - 1
    best_conditional_entropy = 0
    for_collection_data = 0
    data_type = ''
    collection = []
    for i in range(dimension_num):
        thresholds = list(set(dataset[:, i].tolist()))
        for threshold in thresholds:
            gain_ratio, conditional_entropy, information_gain, split_information = get_information_gain_ratio(dataset, i, threshold)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio, best_idx, best_threshold = gain_ratio, i, threshold
            if conditional_entropy > best_conditional_entropy:
                best_conditional_entropy = conditional_entropy
            if split_information == 0:
                data_type = 'information gain'
                for_collection_data=information_gain
            else:
                data_type = 'information gain ratio'
                for_collection_data=gain_ratio
            collection.append([i+1,threshold,data_type,for_collection_data])
    return best_idx, best_threshold, best_gain_ratio, best_conditional_entropy, collection


def make_subtree(dataset):
    dataset = np.array(dataset)
    # print(dataset.shape)

    counter = Counter(dataset[:, -1])
    if len(counter) == 1:
        return dataset[0, -1]

    best_idx, best_threshold, best_gain_ratio, best_conditional_entropy,collection = find_best_split(dataset)
    for c in collection:
        print("When idx is", c[0], ", threshold is ", c[1], ", we get the ", c[2], ", which is ", c[3])






    if best_gain_ratio <= 1e-6:
        return counter.most_common(1)[0][0]
    if best_conditional_entropy == 0:
        return counter.most_common(1)[0][0]

    sub_tree = {best_idx: {'threshold is ': best_threshold}}
    split_data = split_dataset(dataset, best_idx, best_threshold)

    # print("now shape is ")
    # print(np.array(split_data[0]).shape)
    # print(np.array(split_data[1]).shape)
    # print("")
    #sub_tree[best_idx][True] = make_subtree(split_data[0])
    #sub_tree[best_idx][False] = make_subtree(split_data[1])
    return sub_tree


def predict_label(subtree, data):
    key = list(subtree.keys())[0]
    subtree = subtree[key]
    idx = int(key)

    result_label = None
    threshold = subtree['threshold is ']

    if isinstance(subtree[data[idx] >= threshold], dict):
        # print("data[idx] is ", data[idx])
        # print("threshold is ", threshold)
        # print(subtree[data[idx] >= threshold])
        result_label = predict_label(subtree[data[idx] >= threshold], data)

    else:
        result_label = subtree[data[idx] >= threshold]
    return result_label


file1 = open('data/Druns.txt', 'r')
Lines = file1.readlines()
data = []
for line in Lines:
    data_line = [float(line.split()[0]),float(line.split()[1]),float(line.split()[2])]
    data.append(data_line)
#print(data)
datasets = data
decision_tree = make_subtree(datasets)

