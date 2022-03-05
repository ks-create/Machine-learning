import sys
import csv
import math
import numpy as np

# val_input = "fr_data/validation.txt"
# index_to_word_input = "fr_data/index_to_word.txt"
# index_to_tag_input = "fr_data/index_to_tag.txt"
# init_input = "fr_output/hmminit.txt"
# emit_input = "fr_output/hmmemit.txt"
# trans_input = "fr_output/hmmtrans.txt"

def process_input(val_input, index_to_word_input, index_to_tag_input, init_input, emit_input, trans_input):
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

    with open(val_input, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        val_data = []
        sequence = []
        for row in reader:
            if (row == []):
                val_data.append(sequence)
                sequence = []
            else:
                word_i = index_word.index(row[0])
                tag_i = index_tag.index(row[1])
                sequence.append((word_i, tag_i))
        val_data.append(sequence)

    pi = np.genfromtxt(init_input, delimiter=' ')
    A = np.genfromtxt(emit_input, delimiter=' ')
    B = np.genfromtxt(trans_input, delimiter=' ')

    return index_word, index_tag, val_data, pi, A, B

def log_sum_exp(l):
    # l is a np 1-d array
    m = np.amax(l)
    return m+np.log(np.sum(np.exp(l-m)))


def predict(sequence, num_tag, pi, A, B):
    # sequence = [(word, tag), (word, tag)...]

    # FORWARD
    # initialization
    T = len(sequence)
    alpha = np.zeros((T, num_tag))
    x1, tag1 = sequence[0]
    alpha[0] = np.log(pi) + np.log(A[:, x1])

    # recursively define alpha
    for i in range(1, T):
        xi, tagi = sequence[i]
        for j in range(num_tag):
            alpha[i][j] = np.log(A[j][xi]) + log_sum_exp(alpha[i-1] + np.log(B[:,j]))

    # BACKWARD
    # initialization
    beta = np.zeros((T, num_tag))
    beta[-1] = np.ones(num_tag)
    
    # recursively define beta
    for i in range(T-2, -1, -1):
        xi, tagi = sequence[i+1]
        for j in range(num_tag):
            beta[i][j] = log_sum_exp(np.log(A[:, xi]) + beta[i+1] + np.log(B[j,:]))
    
    # PREDICT
    y_hat = np.zeros(T)
    for i in range(T):
        p = alpha[i]+beta[i]
        argmax = np.where(p == np.amax(p))[0][0]
        y_hat[i] = argmax

    # error rate
    total_count = T
    correct_count = 0
    for i in range(0, T):
        xi, tagi = sequence[i]
        if (tagi == y_hat[i]):
            correct_count += 1

    log_likelihood = log_sum_exp(alpha[-1])

    return y_hat.tolist(), total_count, correct_count, log_likelihood
    

if __name__ == "__main__":
    val_input = sys.argv[1]
    index_to_word_input = sys.argv[2]
    index_to_tag_input = sys.argv[3]
    init_input = sys.argv[4]
    emit_input = sys.argv[5]
    trans_input = sys.argv[6]
    predict_out = sys.argv[7]
    metric_out = sys.argv[8]

    index_word, index_tag, val_data, pi, A, B = process_input(val_input, index_to_word_input, index_to_tag_input, init_input, emit_input, trans_input)
    num_tag = len(index_tag)

    sum_total = 0
    sum_correct = 0
    sum_log_likelihood = 0
    num_log_likelihood = 0
    predict_tags = []

    for i in range(len(val_data)):
        y_hat, total_count, correct_count, log_likelihood = predict(val_data[i], num_tag, pi, A, B)
        sum_total += total_count
        sum_correct += correct_count
        sum_log_likelihood += log_likelihood
        num_log_likelihood += 1
        predict_tags.append(y_hat)
    
    avg_log = sum_log_likelihood/num_log_likelihood
    accuracy = sum_correct/sum_total

    predict_out_file = open(predict_out, "w")
    metric_out_file = open(metric_out, "w")

    for i in range(len(val_data)):
        for j in range(len(val_data[i])):
            word, tag = val_data[i][j]
            predict_tag = predict_tags[i][j]
            predict_out_file.write(index_word[int(word)]+"\t")
            predict_out_file.write(index_tag[int(predict_tag)]+"\n")
        predict_out_file.write("\n")
    
    metric_out_file.write("Average Log-Likelihood: "+str(avg_log)+"\n")
    metric_out_file.write("Accuracy: "+str(accuracy)+"\n")










