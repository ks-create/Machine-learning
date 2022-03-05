import sys
import csv
import math

def entropy(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        labels = []
        for row in reader:
            labels.append(row[-1])
    labels = labels[1:]

    dict_l = dict()
    for i in range(len(labels)):
        value = labels[i]
        if (value in dict_l):
            dict_l[value] += 1
        else:
            dict_l[value] = 1
    
    sum = 0
    for key in dict_l:
        p = dict_l[key]/len(labels)
        sum += p*math.log2(p)

    entropy = -1*sum

    return entropy

def error(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        labels = []
        for row in reader:
            labels.append(row[-1])
    labels = labels[1:]

    dict_l = dict()
    for i in range(len(labels)):
        value = labels[i]
        if (value in dict_l):
            dict_l[value] += 1
        else:
            dict_l[value] = 1
    
    major_vote = max(dict_l, key=dict_l.get)

    mismatch_count = 0
    for i in range(len(labels)):
        if (labels[i] != major_vote):
            mismatch_count += 1
    
    return mismatch_count/len(labels)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    entropy = entropy(input_file)
    error = error(input_file)

    out_file = open(output_file, "w")

    out_file.write("entropy: "+str(entropy)+"\n")
    out_file.write("error: "+str(error)+"\n")

    out_file.close()

     
