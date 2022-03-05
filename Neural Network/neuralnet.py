import sys
import csv
import math
import numpy as np

class Object:
    def __init__(self, x, a, z, b, y_hat, J):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J

def linear_forward(x, alpha):
    # x is M vector, alpha is D x M+1 matrix
    # we add ones here
    x = np.insert(x, 0, 1)
    return np.matmul(alpha, x)

def sigmoid_forward(x):
    # apply sigmoid to all elements in x
    return (1/(1+np.exp(-x)))

def softmax_forward(x):
    # apply softmax to all elements in x
    return np.exp(x)/sum(np.exp(x))

def cross_entropy_forward(y, y_hat):
    # y is a integer(label), y_hat is a vector of probabilities
    y_vector = np.zeros(y_hat.shape[0])
    y_vector[y] = 1
    return -1*(y_vector.dot(np.log(y_hat)))

def nn_forward(x, y, alpha, beta):
    # x: vector of length M 
    # y: integer
    # alpha: matrix of size (D x M+1)
    # beta: matrix of size (K x D+1)
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(int(y), y_hat)
    o = Object(x, a, z, b, y_hat, J)
    return o

def softmax_backward(y, y_hat):
    # y is a integer(label), y_hat is a vector of probabilities
    y_vector = np.zeros(y_hat.shape[0])
    y_vector[y] = 1
    return -1*y_vector + y_hat

def linear_backward(z, beta, g_b):
    z = np.insert(z, 0, 1)
    g_beta = np.outer(g_b, z)
    g_z = np.matmul(np.transpose(beta[:,1:]), g_b)
    return g_beta, g_z

def sigmoid_backward(a, z, g_z):
    return g_z*z*(1-z)

def nn_backward(x, y, alpha, beta, o):
    a = o.a
    z = o.z
    b = o.b
    y_hat = o.y_hat
    J = o.J
    g_b = softmax_backward(int(y), y_hat)
    g_beta, g_z = linear_backward(z, beta, g_b)
    g_a = sigmoid_backward(a, z, g_z)
    g_alpha, g_x = linear_backward(x, alpha, g_a)
    return g_alpha, g_beta

# data = np.genfromtxt ('tinyTrain.csv', delimiter=",")
# y = data[:, 0]
# x = data[:, 1:]
# m = x.shape[1]
# K = 4
# epsilon = 1*(10**-5)
# learning_rate = 0.1
# alpha = np.zeros((4, m+1))
# beta = np.zeros((4, 4+1))
# s_alpha = np.zeros((4, m+1))
# s_beta = np.zeros((4, 4+1))
# x_1 = x[0]
# y_1 = y[0]
# o = nn_forward(x_1, y_1, alpha, beta)
# g_alpha, g_beta = nn_backward(x_1, y_1, alpha, beta, o)
# s_alpha = s_alpha + g_alpha*g_alpha
# s_beta = s_beta + g_beta*g_beta
# alpha = alpha - (learning_rate/np.sqrt(s_alpha+epsilon))*g_alpha
# beta = beta - (learning_rate/np.sqrt(s_beta+epsilon))*g_beta

# x_2 = x[1]
# y_2 = y[1]
# o = nn_forward(x_2, y_2, alpha, beta)
# g_alpha, g_beta = nn_backward(x_2, y_2, alpha, beta, o)
# print(o.J)

# s_alpha = s_alpha + g_alpha*g_alpha
# s_beta = s_beta + g_beta*g_beta
# alpha = alpha - (learning_rate/np.sqrt(s_alpha+epsilon))*g_alpha
# beta = beta - (learning_rate/np.sqrt(s_beta+epsilon))*g_beta

# print(alpha)
# print(beta)

def mean_cross_entropy(x, y, alpha, beta):
    K = 4 # given
    sum = 0
    count = 0
    for i in range(x.shape[0]):
        o = nn_forward(x[i], y[i], alpha, beta)
        sum += o.J
        count += 1
    return sum/count

def sgd(train_input, val_input, epoch, hidden_units, init_flag, learning_rate):
    train_data = np.genfromtxt (train_input, delimiter=",")
    train_y = train_data[:, 0]
    train_x = train_data[:, 1:]

    val_data = np.genfromtxt (val_input, delimiter=",")
    val_y = val_data[:, 0]
    val_x = val_data[:, 1:]

    K = 4 # given
    M = train_x.shape[1]
    D = hidden_units
    epsilon = 1*(10**-5)

    if (init_flag == 1):
        # random initialization
        alpha = np.random.uniform(-0.1, 0.1, (D,M+1))
        beta = np.random.uniform(-0.1, 0.1, (K,D+1))
    else:
        alpha = np.zeros((D,M+1))
        beta = np.zeros((K,D+1))
    
    s_alpha = np.zeros((D,M+1))
    s_beta = np.zeros((K,D+1))

    train_mce = []
    val_mce = []

    for e in range(epoch):
        print(e)
        for i in range(train_x.shape[0]):
            x = train_x[i]
            y = train_y[i]
            o = nn_forward(x, y, alpha, beta)
            g_alpha, g_beta = nn_backward(x, y, alpha, beta, o)
            s_alpha = s_alpha + g_alpha*g_alpha
            s_beta = s_beta + g_beta*g_beta
            alpha = alpha - (learning_rate/np.sqrt(s_alpha+epsilon))*g_alpha
            beta = beta - (learning_rate/np.sqrt(s_beta+epsilon))*g_beta
            print("alpha", alpha, "\n")
            print("beta", beta, "\n")
        train_mce.append(mean_cross_entropy(train_x, train_y, alpha, beta))
        val_mce.append(mean_cross_entropy(val_x, val_y, alpha, beta))
    return alpha, beta, train_mce, val_mce

def error_rate(true_data, predict_data):
    mis_count = 0
    for i in range(len(true_data)):
        if (true_data[i] != predict_data[i]):
            mis_count += 1
    return mis_count/len(true_data)

def predict(input_file, alpha, beta):
    data = np.genfromtxt (input_file, delimiter=",")
    y = data[:, 0]
    x = data[:, 1:]
    y_predict = []
    for i in range(x.shape[0]):
        # print(x[i])
        a = linear_forward(x[i], alpha)
        # print(a)
        z = sigmoid_forward(a)
        # print(z)
        b = linear_forward(z, beta)
        # print(b)
        y_hat = softmax_forward(b)
        # print(y_hat)
        l = (np.where(y_hat == np.amax(y_hat)))[0][0]
        y_predict.append(l)
    rate = error_rate(y.tolist(), y_predict)
    return y_predict, rate

if __name__ == "__main__":
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    train_out = sys.argv[3]
    val_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    alpha, beta, train_mce, val_mce = sgd(train_input, val_input, num_epoch, hidden_units, init_flag, learning_rate)
    train_predict, train_rate = predict(train_input, alpha, beta)
    val_predict, val_rate = predict(val_input, alpha, beta)

    # print("train:", train_mce)
    # print("validation: ", val_mce)
    
    train_out_file = open(train_out, "w")
    val_out_file = open(val_out, "w")
    metrics_out_file = open(metrics_out, "w")

    for i in train_predict:
        train_out_file.write(str(i)+"\n")
    for i in val_predict:
        val_out_file.write(str(i)+"\n")
    for i in range(len(train_mce)):
        metrics_out_file.write("epoch="+str(i+1)+" crossentropy(train): "+str(train_mce[i])+"\n")
        metrics_out_file.write("epoch="+str(i+1)+" crossentropy(validation): "+str(val_mce[i])+"\n")
    metrics_out_file.write("error(train): "+str(train_rate)+"\n")
    metrics_out_file.write("error(validation): "+str(val_rate))

    train_out_file.close()
    val_out_file.close()
    metrics_out_file.close()


