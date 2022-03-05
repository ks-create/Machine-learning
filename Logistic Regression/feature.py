import sys
import csv
import math

train_input = "smalldata/test_data.tsv"
dict_input = "dict.txt"
word2vec_input = "word2vec.txt"

def model_1(input_file, dict_file):
    # format input_file as text[i]=[label, word1, word2, ..]
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        text = []
        for row in reader:
            text.append(row[0])

    for i in range(len(text)):
        text[i] = text[i].split("\t")
        text[i][1] = text[i][1].split(" ")
        text[i] = [text[i][0]] + text[i][1]
    
    # format dict_file as dict[i] = [word, index]
    with open(dict_file) as f:
        dict = f.readlines()
    
    for i in range(len(dict)):
        dict[i] = dict[i].split(" ")
        dict[i][1] = int(dict[i][1][:-1])
    
    output = []
    for i in range(len(text)):
        output_i = []
        for j in range(len(dict)):
            if (dict[j][0] in text[i][1:]):
                output_i.append(1)
            else:
                output_i.append(0)
        output_i = [int(text[i][0])] + output_i
        output.append(output_i)
    
    return output

def model_2(input_file, dict_file):
    # format input_file as text[i]=[label, word1, word2, ..]
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        text = []
        for row in reader:
            text.append(row[0])

    for i in range(len(text)):
        text[i] = text[i].split("\t")
        text[i][1] = text[i][1].split(" ")
        text[i] = [text[i][0]] + text[i][1]
    
    # format dict_file as dict[i] = [word, v1, v2, ..]
    with open(dict_file) as f:
        dict = f.readlines()
    
    dict_word = []
    for i in range(len(dict)):
        dict[i] = dict[i].split("\t")
        dict[i][-1] = dict[i][-1][:-1]
        dict_word.append(dict[i][0])
    
    output = []
    for i in range(len(text)):
        count = 0
        output_i = [0]*(len(dict[0])-1)
        for j in range(1,len(text[i])):
            if (text[i][j] in dict_word):
                count += 1
                index = dict_word.index(text[i][j])
                for k in range(1,len(dict[index])):
                    output_i[k-1] += float(dict[index][k])
        for l in range(len(output_i)):
            output_i[l] = format(output_i[l]/count, ".6f")
            # output_i[l] = round(output_i[l]/count, 6)
        output_i = [format(int(text[i][0]), ".6f")] + output_i
        output.append(output_i)
    
    return output

if __name__ == "__main__":
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    feature_dictionary_input = sys.argv[9]

    if (feature_flag == 1):
        train_output = model_1(train_input, dict_input)
        val_output = model_1(validation_input, dict_input)
        test_output = model_1(test_input, dict_input)
    else:
        train_output = model_2(train_input, feature_dictionary_input)
        val_output = model_2(validation_input, feature_dictionary_input)
        test_output = model_2(test_input, feature_dictionary_input)
    
    train_out_file = open(formatted_train_out, "w")
    val_out_file = open(formatted_validation_out, "w")
    test_out_file = open(formatted_test_out, "w")

    for line in train_output:
        for i in range(len(line)-1):
            train_out_file.write(str(line[i])+"\t")
        train_out_file.write(str(line[len(line)-1])+"\n")
    for line in val_output:
        for i in range(len(line)-1):
            val_out_file.write(str(line[i])+"\t")
        val_out_file.write(str(line[len(line)-1])+"\n")
    for line in test_output:
        for i in range(len(line)-1):
            test_out_file.write(str(line[i])+"\t")
        test_out_file.write(str(line[len(line)-1])+"\n")

    train_out_file.close()
    val_out_file.close()
    test_out_file.close()