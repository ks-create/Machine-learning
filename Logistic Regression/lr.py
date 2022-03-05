import sys
import csv
import math
import numpy as np

train_input = "smalloutput/formatted_train.tsv"
test_input = "smalloutput/formatted_test.tsv"

def theta_reg(input_file, val_file, epochs):
    # text file format: [label, v1, v2, .., b=1]
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        text = []
        for row in reader:
            text.append(row[0])
    for i in range(len(text)):
        text[i] = text[i].split("\t")
        text[i].append(1)
        for j in range(len(text[i])):
            text[i][j] = float(text[i][j])
    
    with open(val_file, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        val_text = []
        for row in reader:
            val_text.append(row[0])
    for i in range(len(val_text)):
        val_text[i] = val_text[i].split("\t")
        val_text[i].append(1)
        for j in range(len(val_text[i])):
            val_text[i][j] = float(val_text[i][j])

    alpha = 0.01
    theta = np.zeros(len(text[1])-1)
    n = len(text)

    for i in range(epochs):
        for line in text:
            theta_dot_x = np.dot(theta, line[1:])
            y_minus_term = line[0]-(math.exp(theta_dot_x)/(1+math.exp(theta_dot_x)))
            graident_trem_np = np.array(line[1:])*alpha*y_minus_term/n
            theta = np.add(theta, graident_trem_np)
            # theta = (np.array(theta) * )
            # for j in range(len(line)-1):
            #     theta[j] += alpha*float(line[j])*y_minus_term/n
        # x = text[(i % len(text))]
        # theta_dot_x = np.dot(theta, x[1:])
        # # for j in range(len(x)-1):
        # #     theta_dot_x += theta[j]*float(x[j+1])
        # y_minus_term = float(x[0])-math.exp(theta_dot_x)/(1+math.exp(theta_dot_x))
        # for j in range(len(x)-1):
        #     theta[j] += alpha*float(x[j])*y_minus_term
        if (i%100 == 0):
            res = 0
            val_res = 0
            for line in text:
                res += -1 * line[0] * np.dot(theta, line[1:]) + np.log(1+math.exp(np.dot(theta, line[1:])))
            for line in val_text:
                val_res += -1 * line[0] * np.dot(theta, line[1:]) + np.log(1+math.exp(np.dot(theta, line[1:])))
            print(i, res/len(text), "validation", val_res/len(val_text))

    return theta.tolist()

def error_rate(true_data, predict_data):
    mis_count = 0
    for i in range(len(true_data)):
        if (true_data[i] != predict_data[i]):
            mis_count += 1
    return mis_count/len(true_data)

def predict(input_file, theta):
   # text file format: [label, v1, v2, .., b=1]
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        text = []
        for row in reader:
            text.append(row[0])

    for i in range(len(text)):
        text[i] = text[i].split("\t")
        text[i].append(1) 
        for j in range(len(text[i])):
            text[i][j] = float(text[i][j])

    output = []
    for i in range(len(text)):
        theta_dot_x = np.dot(theta, text[i][1:])
        # sum = 0
        # for j in range(len(text[i])-1):
        #     sum += theta[j]*float(text[i][j+1])
        val = (math.exp(theta_dot_x))/(1+math.exp(theta_dot_x))
        # print(theta_dot_x)
        if (val >= 0.5):
            output.append(1)
        else:
            output.append(0)
    
    true_label = []
    for line in text:
        true_label.append(line[0])
    error = error_rate(true_label, output)

    return output, error

# theta = theta_reg(train_input, 30)
# predict(test_input, theta)
if __name__ == "__main__":
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    theta = theta_reg(formatted_train_input, formatted_validation_input, num_epoch)
    train_predicted, train_error = predict(formatted_train_input, theta)
    test_predicted, test_error = predict(formatted_test_input, theta)

    train_out_file = open(train_out, "w")
    test_out_file = open(test_out, "w")
    metrics_out_file = open(metrics_out, "w")

    for i in train_predicted:
        train_out_file.write(str(i)+"\n")
    for i in test_predicted:
        test_out_file.write(str(i)+"\n")
    metrics_out_file.write("error(train): "+str(train_error)+"\n")
    metrics_out_file.write("error(test): "+str(test_error))

    train_out_file.close()
    test_out_file.close()
    metrics_out_file.close()
