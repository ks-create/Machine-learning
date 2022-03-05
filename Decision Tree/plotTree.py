import sys
import csv
import math

class Node:
    def __init__(self, isLeaf):
        self.attribute = None
        self.isLeaf = isLeaf
        self.leafValue = None
        self.leftBranch = None
        self.leftNode = None
        self.rightBranch = None
        self.rightNode = None

def entropy(input_data):
    data = []
    for row in input_data:
        data.append(row[-1])
    
    d = dict()
    for i in range(len(data)):
        value = data[i]
        if (value in d):
            d[value] += 1
        else:
            d[value] = 1
    
    sum = 0
    for key in d:
        p = d[key]/len(data)
        sum += p*math.log2(p)

    entropy = -1*sum

    return entropy

# attribute X: int representing index, value x
def specific_cond_entropy(input_data, attribute, value):
    d_a = dict()
    d_sum = 0
    for i in range(len(input_data)):
        if (input_data[i][attribute] == value):
            d_sum += 1
            val = input_data[i][-1]
            if (val in d_a):
                d_a[val] += 1
            else:
                d_a[val] = 1
    
    sum = 0
    for key in d_a:
        p = d_a[key]/d_sum
        sum += p*math.log2(p)
    
    return -1*sum

def cond_entropy(input_data, attribute):
    d = dict()
    d_sum = 0
    for i in range(len(input_data)):
        d_sum += 1
        value = input_data[i][attribute]
        if (value in d):
            d[value] += 1
        else:
            d[value] = 1
    sum = 0
    for key in d:
        sum += (d[key]/d_sum)*specific_cond_entropy(input_data, attribute, key)
    return sum

def mutual_info(input_data, attribute):
    return entropy(input_data)-cond_entropy(input_data, attribute)

def major_vote(input_data):
    label = []
    for row in input_data:
        label.append(row[-1])
    
    d = dict()
    for i in range(len(label)):
        value = label[i]
        if (value in d):
            d[value] += 1
        else:
            d[value] = 1
    
    max_list = []
    max_val = -1
    for key in d:
        if (d[key] > max_val):
            max_list = [key]
            max_val = d[key]
        elif (d[key] == max_val):
            max_list.append(key)
        else:
            continue
    max_list.sort()
    return max_list[-1]
    # return max(d, key=d.get)

def print_label_count(input_data):
    label = [row[-1] for row in input_data]
    
    d = dict()
    for i in range(len(label)):
        value = label[i]
        if (value in d):
            d[value] += 1
        else:
            d[value] = 1
    
    print(d)

def train_true(input_data, depth, col_names, max_depth):
    print_label_count(input_data)
    # if at maxium depth
    if (depth == 0):
        node = Node(True)
        node.leafValue = major_vote(input_data)
        # print("depth:", input_data, node.leafValue)
        return node

    # find the best attribute x_m
    d_attr = dict()
    for i in range(len(input_data[0])-1):
        d_attr[i] = mutual_info(input_data, i)
    x_m = max(d_attr, key=d_attr.get)

    if (d_attr[x_m] <= 0):
        node = Node(True)
        node.leafValue = major_vote(input_data)
        # print("classified: ", node.leafValue)
        return node

    # find binary values in x_m
    x_m_list = []
    for i in range(len(input_data)):
        x_m_list.append(input_data[i][x_m])
    x_m_set = set(x_m_list)
    x_m_unique = list(x_m_set)
    v = x_m_unique[0]
    notv = x_m_unique[1]

    v_data = []
    notv_data = []
    for row in input_data:
        if (row[x_m] == v):
            v_data.append(row)
        else:
            notv_data.append(row)

    root = Node(False)
    root.attribute = x_m
    root.leftBranch = v
    root.rightBranch = notv

    print("| "*(max_depth-depth+1), end="")
    print(col_names[x_m], " = ", v, ": ", end="")
    root.leftNode = train_true(v_data, depth-1, col_names, max_depth)
    print("| "*(max_depth-depth+1), end="")
    print(col_names[x_m], " = ", notv, ": ", end="")
    root.rightNode = train_true(notv_data, depth-1, col_names, max_depth)
    # print(root.attribute, root.leftBranch, root.rightBranch)

    return root

def predict(input_data, root):
    output = []
    for row in input_data:
        node = root
        while (node.isLeaf == False):
            if (row[node.attribute] == node.leftBranch):
                node = node.leftNode
            else:
                node = node.rightNode
        output.append(node.leafValue)
    return output

def error_rate(true_data, predict_data):
    mis_count = 0
    for i in range(len(true_data)):
        if (true_data[i] != predict_data[i]):
            mis_count += 1
    return mis_count/len(true_data)

def train(train_name, test_name, max_depth):
    with open(train_name, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        train_data = []
        for row in reader:
            train_data.append(row)
    col_names = train_data[0]
    train_data = train_data[1:]
    train_true_label = [row[-1] for row in train_data]

    with open(test_name, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        test_data = []
        for row in reader:
            test_data.append(row)
    test_data = test_data[1:]
    test_true_label = [row[-1] for row in test_data]

    decision_tree = train_true(train_data, max_depth, col_names, max_depth)
    train_predict = predict(train_data, decision_tree)
    test_predict = predict(test_data, decision_tree)

    train_error = error_rate(train_true_label, train_predict)
    test_error = error_rate(test_true_label, test_predict)

    return train_predict, test_predict, train_error, test_error

train_input = "politicians_train.tsv" 
test_input = "politicians_test.tsv"
with open(train_input, "r") as file:
    reader = csv.reader(file, delimiter="\t")
    train_data = []
    for row in reader:
        train_data.append(row)
max_depth = len(train_data[0])
x = range(max_depth+1)
y_train = []
y_test = []

for i in x:
    train_pre, test_pre, train_err, test_err = train(train_input, test_input, i)
    y_train.append(train_err)
    y_test.append(test_err)

print(x)
print(y_train)
print(y_test)