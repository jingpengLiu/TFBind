import json
import torch
import torch.nn.functional as F
import os, shutil, random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
# 制作dataset
def make_train_dataset():
    label_list = []
    sequence_list = []
    dir_0 = 'train_test/train/0/'
    dir_1 = 'train_test/train/1/'
    for i in os.listdir(dir_0):
        dir = dir_0 + i
        tf = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
        target = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
        with open(f"{dir}/{tf}", "r") as tf_file:
            protein_sequence = ""
            tf_file = tf_file.readlines()
            for str in tf_file:
                if str[0] != ">":
                    str = str.replace("\n", "")
                    str = str.upper()
                    protein_sequence += str

        with open(f"{dir}/{target}", "r") as target_file:
            target_file = target_file.readlines()
            for str in target_file:
                if str[0] != ">":
                    str = str.lower()
                    str = str.replace("\n", "")
                    str = str.replace("n", "")
                    all_sequence = protein_sequence + str
                    sequence_list.append(all_sequence)
                    label_list.append(0)

    for i in os.listdir(dir_1):
        dir = dir_1 + i
        tf = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
        target = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
        with open(f"{dir}/{tf}", "r") as tf_file:
            protein_sequence = ""
            tf_file = tf_file.readlines()
            for str in tf_file:
                if str[0] != ">":
                    str = str.replace("\n", "")
                    str = str.upper()
                    protein_sequence += str

        with open(f"{dir}/{target}", "r") as target_file:
            target_file = target_file.readlines()
            for str in target_file:
                if str[0] != ">":
                    str = str.lower()
                    str = str.replace("\n", "")
                    str = str.replace("n", "")
                    all_sequence = protein_sequence + str
                    sequence_list.append(all_sequence)
                    label_list.append(1)

    return sequence_list, label_list

# 制作test dataset
def make_val_dataset():
    label_list = []
    sequence_list = []
    dir_0 = 'train_test/test/0/'
    dir_1 = 'train_test/test/1/'
    for i in os.listdir(dir_0):
        dir = dir_0 + i
        tf = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
        target = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
        with open(f"{dir}/{tf}", "r") as tf_file:
            protein_sequence = ""
            tf_file = tf_file.readlines()
            for str in tf_file:
                if str[0] != ">":
                    str = str.replace("\n", "")
                    str = str.upper()
                    protein_sequence += str

        with open(f"{dir}/{target}", "r") as target_file:
            target_file = target_file.readlines()
            for str in target_file:
                if str[0] != ">":
                    str = str.lower()
                    str = str.replace("\n", "")
                    str = str.replace("n", "")
                    all_sequence = protein_sequence + str
                    sequence_list.append(all_sequence)
                    label_list.append(0)

    for i in os.listdir(dir_1):
        dir = dir_1 + i
        tf = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
        target = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
        with open(f"{dir}/{tf}", "r") as tf_file:
            protein_sequence = ""
            tf_file = tf_file.readlines()
            for str in tf_file:
                if str[0] != ">":
                    str = str.replace("\n", "")
                    str = str.upper()
                    protein_sequence += str

        with open(f"{dir}/{target}", "r") as target_file:
            target_file = target_file.readlines()
            for str in target_file:
                if str[0] != ">":
                    str = str.lower()
                    str = str.replace("\n", "")
                    str = str.replace("n", "")
                    all_sequence = protein_sequence + str
                    sequence_list.append(all_sequence)
                    label_list.append(1)

    return sequence_list, label_list

# datas转tensor
def datas_totensor(datas, max_length):
    dict_file = open("dict.txt")
    table = dict_file.read()
    table = eval(table)
    nuc_length = table.__len__()
    max_length = max_length # 氨基酸和核苷酸合并后的最大长度
    all_tensor = torch.empty((0, 1, max_length, nuc_length))
    all_tensor = torch.reshape(all_tensor, (0, 8, 94, 65))
    for str in datas:
        try:
            int_str = list(map(lambda x: table[x], list(str)))
        except Exception:
            print(str)
        int_str += [0 for i in range(max_length - len(int_str))]
        tensor_str = F.one_hot(torch.tensor(int_str), nuc_length)
        tensor_str = torch.reshape(tensor_str, (1, 1, max_length, nuc_length))
        tensor_str = torch.reshape(tensor_str, (1, 8, 94, 65))
        all_tensor = torch.cat((all_tensor, tensor_str), 0)
    return all_tensor

def datas_totensor2(datas, max_length):
    dict_file = open("dict.txt")
    table = dict_file.read()
    table = eval(table)
    nuc_length = table.__len__()
    max_length = max_length # 氨基酸和核苷酸合并后的最大长度
    all_tensor = torch.empty((0, 1, max_length, nuc_length))
    all_tensor = torch.reshape(all_tensor, (0, 8, 94, 65))
    for str in datas:
        try:
            int_str = list(map(lambda x: table[x], list(str)))
        except Exception:
            print(str)
        int_str += [0 for i in range(max_length - len(int_str))]
        tensor_str = F.one_hot(torch.tensor(int_str), nuc_length)
        tensor_str = torch.reshape(tensor_str, (1, 1, max_length, nuc_length))
        tensor_str = torch.reshape(tensor_str, (1, 8, 94, 65))
        all_tensor = torch.cat((all_tensor, tensor_str), 0)
    return all_tensor



# 数据集分类并且打标签
def datas_label():
    list_dir = os.listdir("./jasper_data")
    data_len = list_dir.__len__()
    number = int(data_len * 0.9)
    for dir in list_dir[0:number]:
        # 训练数据集中，转录因子可以结合DNA的数据
        if not os.path.exists(f"train_test/train/1/{dir}"):
            shutil.copytree(f"./jasper_data/{dir}/", f"train_test/train/1/{dir}")
        # 训练数据集中，转录因子不可以结合DNA的数据
        tf_dir_list = os.listdir(f'./jasper_data/{dir}')
        tf_dir = tf_dir_list[1] if tf_dir_list[0][0:2] == 'MA' else tf_dir_list[0]
        target_dir_list = list_dir.copy()
        target_dir_list.remove(dir)
        target_dir_list = random.choice(target_dir_list)
        target_dir_number = target_dir_list
        target_dir_list = os.listdir(f'./jasper_data/{target_dir_list}')
        target_dir = target_dir_list[0] if target_dir_list[0][0:2] == 'MA' else target_dir_list[1]
        if not os.path.exists(f"train_test/train/0/{dir}"):
            os.makedirs(f"train_test/train/0/{dir}")
        shutil.copy(f"./jasper_data/{dir}/{tf_dir}", f"train_test/train/0/{dir}/{tf_dir}")
        shutil.copy(f"./jasper_data/{target_dir_number}/{target_dir}", f"train_test/train/0/{dir}/")

    for dir in list_dir[number:]:
        # 测试数据集中，转录因子可以结合DNA的数据
        if not os.path.exists(f"train_test/test/1/{dir}"):
            shutil.copytree(f"./jasper_data/{dir}/", f"train_test/test/1/{dir}")
        # 测试数据集中，转录因子不可以结合DNA的数据
        tf_dir_list = os.listdir(f'./jasper_data/{dir}')
        tf_dir = tf_dir_list[1] if tf_dir_list[0][0:2] == 'MA' else tf_dir_list[0]
        target_dir_list = list_dir.copy()
        target_dir_list.remove(dir)
        target_dir_list = random.choice(target_dir_list)
        target_dir_number = target_dir_list
        target_dir_list = os.listdir(f'./jasper_data/{target_dir_list}')
        target_dir = target_dir_list[0] if target_dir_list[0][0:2] == 'MA' else target_dir_list[1]
        if not os.path.exists(f"train_test/test/0/{dir}"):
            os.makedirs(f"train_test/test/0/{dir}")
        shutil.copy(f"./jasper_data/{dir}/{tf_dir}", f"train_test/test/0/{dir}/{tf_dir}")
        shutil.copy(f"./jasper_data/{target_dir_number}/{target_dir}", f"train_test/test/0/{dir}/")


# 清洗数据集
def clean_data():
    dir = './jasper_data/'
    dir_list = os.listdir(dir)
    rm_list = []
    for i in dir_list:
        data_dir = dir + i
        data_dir_list = os.listdir(data_dir)
        tf = data_dir_list[1] if data_dir_list[0][0:2] == 'MA' else data_dir_list[0]
        target = data_dir_list[0] if data_dir_list[0][0:2] == 'MA' else data_dir_list[1]

        # 替换数据
        shutil.copy(f"D:/idm_download/sites/{target}", f"{data_dir}")
        with open(f"{data_dir}/{tf}", "r") as infile:
            if "Error" in infile.read():
                print(data_dir)
                rm_list.append(data_dir)

    # 删除Error文件
    for dir in rm_list:
        shutil.rmtree(dir)

    print(True)


# 得到list中最大长度的字符串
def get_max_length(nuc_data):
    max_length = 0
    for i in nuc_data:
        if i.__len__() > max_length:
            max_length = i.__len__()

    return max_length


# 记录数据集中的蛋白数据的最大长度，dna数据的最大长度，总最大长度
def record_max_length():
    protein_list = []
    dna_list = []
    dir_list = ["./train_test/train/0/", "./train_test/train/1/", "./train_test/test/0/", "./train_test/test/1/"]
    for h in dir_list:
        for i in os.listdir(h):
            dir = h + i
            tf = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
            target = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
            with open(f"{dir}/{tf}", "r") as tf_file:
                protein_sequence = ""
                tf_file = tf_file.readlines()
                for str in tf_file:
                    if str[0] != ">":
                        str = str.replace("\n", "")
                        str = str.upper()
                        protein_sequence += str
                protein_list.append(protein_sequence)

            with open(f"{dir}/{target}", "r") as target_file:
                target_file = target_file.readlines()
                for str in target_file:
                    if str[0] != ">":
                        str = str.lower()
                        str = str.replace("\n", "")
                        dna_list.append(str)
    protein_length_max = get_max_length(protein_list)
    dna_length_max = get_max_length(dna_list)
    max_length = protein_length_max + dna_length_max
    length_dict = {
        "protein_length_max": protein_length_max,
        "dna_length_max": dna_length_max,
        "max_length": max_length
    }
    with open("train_test/length.txt", "w+") as outfile:
        outfile.write(json.dumps(length_dict))


def shuffle_data(nucdata, nuclabel):
    datas = {
        "nucdata": nucdata,
        "nuclabel": nuclabel
    }
    df = pd.DataFrame(datas)
    df = df.iloc[np.random.permutation(len(df)), ]
    df = df.iloc[np.random.permutation(len(df)), ]
    df = df.iloc[np.random.permutation(len(df)), ]
    nucdata = df['nucdata'].to_list()
    nuclabel = df['nuclabel'].to_list()
    return nucdata, nuclabel


# 制作数据集总表csv
def make_csvdata():
    dir_list = os.listdir("./jasper_data")
    # dir_list = dir_list[0:10]
    with open(f"./csv_data/tf_target_all.csv", "w+") as outfile:
        # 蛋白和dna可以结合的
        for one_dir in dir_list:
            dir = f"./jasper_data/{one_dir}"
            tf_dir = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
            target_dir = os.listdir(dir)[0] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[1]
            with open(f"{dir}/{tf_dir}") as tf_file:
                with open(f"{dir}/{target_dir}") as target_file:
                    tf_list = tf_file.readlines()
                    target_list = target_file.readlines()
                    protein = ""
                    for tf_data in tf_list:
                        if tf_data[0] != ">":
                            tf_data = tf_data.replace("\n", "")
                            protein += tf_data
                    for target_data in target_list:
                        if target_data[0] != ">":
                            target_data = target_data.replace("\n", "")
                            outfile.write(f"{protein},{target_data},1\n")

        # 蛋白和dna不可以结合的
        for one_dir in dir_list:
            dir = f"./jasper_data/{one_dir}"
            new_list = dir_list.copy()
            new_list.remove(one_dir)
            dir1 = random.choice(new_list)
            dir1 = f"./jasper_data/{dir1}"
            tf_dir = os.listdir(dir)[1] if os.listdir(dir)[0][0:2] == 'MA' else os.listdir(dir)[0]
            target_dir = os.listdir(dir1)[0] if os.listdir(dir1)[0][0:2] == 'MA' else os.listdir(dir1)[1]
            with open(f"{dir}/{tf_dir}") as tf_file:
                with open(f"{dir1}/{target_dir}") as target_file:
                    tf_list = tf_file.readlines()
                    target_list = target_file.readlines()
                    protein = ""
                    for tf_data in tf_list:
                        if tf_data[0] != ">":
                            tf_data = tf_data.replace("\n", "")
                            protein += tf_data
                    for target_data in target_list:
                        if target_data[0] != ">":
                            target_data = target_data.replace("\n", "")
                            outfile.write(f"{protein},{target_data},0\n")

# csv文件打乱
def shuffle_csv():
    data = pd.read_csv("./csv_data/tf_target_all.csv", header=None)
    data = shuffle(data)
    data = shuffle(data)
    data.to_csv('./csv_data/shuffle_tf_target_all.csv', index=False)

# 制作训练集和测试集的csv
def make_train_test_csv():
    data = pd.read_csv("./csv_data/shuffle_tf_target_all.csv")
    duandian = int(data.shape[0] * 0.8)
    protein_length_max = get_max_length(data['0']) + 10
    dna_length_max = get_max_length(data['1']) + 10
    max_length = protein_length_max + dna_length_max
    json_data = {
        "protein_length_max": protein_length_max,
        "dna_length_max": dna_length_max,
        "max_length": max_length
    }
    with open("./train_test/length.txt", "w+") as outfile:
        outfile.write(json.dumps(json_data))
    data.iloc[:duandian, :].to_csv("./train_test/train.csv", index=False)
    data.iloc[duandian:, :].to_csv("./train_test/test.csv", index=False)

# 训练和测试数据集
def make_dataset1(dir):
    sequence_list = []
    label_list = []
    with open(dir) as infile:
        data_list = infile.readlines()
        data_list = data_list[1:]
        for data in data_list:
            data = data.replace("\n", "")
            protein = data.split(",")[0]
            protein = protein.upper()
            dna = data.split(",")[1]
            dna = dna.lower()
            dna = dna.replace("n", "")
            sequence = protein + dna
            label = data.split(",")[2]
            label = int(label)
            sequence_list.append(sequence)
            label_list.append(label)
    return sequence_list, label_list


if __name__ == '__main__':
   print("OK")


