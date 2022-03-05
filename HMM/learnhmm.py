import sys
import csv
import math
import numpy as np

def adjust_pseudo(a):
    a = a+1
    a = a/np.sum(a)
    return a

def learn(train_input, index_to_word_input, index_to_tag_input):
    with open(train_input, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        train_data = []
        tmp = []
        for row in reader:
            if (row == []):
                train_data.append(tmp)
                tmp = []
            else:
                tmp.append((row[0], row[1]))
        train_data.append(tmp)

    with open(index_to_word_input, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            index_word = []
            for row in reader:
                index_word.append(row[0])

    with open(index_to_tag_input, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            index_tag = []
            for row in reader:
                index_tag.append(row[0])

    # for i in range(len(train_data)):
    #     for j in range(len(train_data[i])):
    #         word, tag = train_data[i][j]
    #         word_i = index_word.index(word)
    #         tag_i = index_tag.index(tag)
    #         train_data[i][j] = (word_i, tag_i)

    pi = np.zeros(len(index_tag))
    for i in range(len(train_data)):
        word, label = train_data[i][0]
        pi[index_tag.index(label)] += 1
    pi = adjust_pseudo(pi)

    B = np.zeros((len(index_tag), len(index_tag)))
    for i in range(len(train_data)):
        for j in range(len(train_data[i])-1):
            word_t, tag_t = train_data[i][j]
            word_t1, tag_t1 = train_data[i][j+1]
            B[index_tag.index(tag_t)][index_tag.index(tag_t1)] += 1
    B = np.apply_along_axis(adjust_pseudo, 1, B)

    A = np.zeros((len(index_tag), len(index_word)))
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            word_t, tag_t = train_data[i][j]
            A[index_tag.index(tag_t)][index_word.index(word_t)] += 1
    A = np.apply_along_axis(adjust_pseudo, 1, A)

    return pi, B, A

if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word_input = sys.argv[2]
    index_to_tag_input = sys.argv[3]
    init_out = sys.argv[4]
    emit_out = sys.argv[5]
    trans_out = sys.argv[6]

    pi, B, A = learn(train_input, index_to_word_input, index_to_tag_input)

    init_out_file = open(init_out, "w")
    emit_out_file = open(emit_out, "w")
    trans_out_file = open(trans_out, "w")

    for i in pi:
        init_out_file.write(str(i)+"\n")
    for line in B:
        for prob in line:
            trans_out_file.write(str(prob)+" ")
        trans_out_file.write("\n")
    for line in A:
        for prob in line:
            emit_out_file.write(str(prob)+" ")
        emit_out_file.write("\n")

