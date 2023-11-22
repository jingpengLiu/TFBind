import json
import time
import torch.cuda
import nuc_dataset
import nuc_utils
from torch.utils.data import DataLoader
from nuc_model import *
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':

    # 数据集最大长度
    with open("train_test/length.txt", "r") as infile:
        js = infile.read()
        dic = json.loads(js)
        dna_length_max = dic['dna_length_max']
        protein_length_max = dic['protein_length_max']
        max_length = dic['max_length']

    nuc_data, nuc_label = nuc_utils.make_dataset1(dir="./train_test/train.csv")
    nuc_data, nuc_label = nuc_utils.shuffle_data(nuc_data, nuc_label)


    # 训练数据集
    train_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 测试数据集
    val_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    print("#################################################")

    # 数据集长度
    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    print(f"训练数据集的长度:{train_data_size}")
    print(f"验证数据集的长度:{val_data_size}")


    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    nuc = Nuc()
    nuc.to(device)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(nuc.parameters(), lr=learning_rate)

    # 记录训练的参数
    epochs = 1000
    total_train_number = 0
    total_test_number = 0

    # 添加tensorboard
    writer = SummaryWriter("./logs")
    start_time = time.time()

    # 神经网络训练步骤和测试步骤
    for epoch in range(epochs):
        one_epoch_start_time = time.time()
        print(f"###################第{epoch}轮训练######################")

        # 训练步骤开始
        nuc.train()
        for data in train_dataloader:
            datas, targets = data
            datas = nuc_utils.datas_totensor(datas, max_length)
            datas = datas.to(device)
            targets = targets.to(device)
            outputs = nuc(datas)

            # 计算损失
            loss = loss_fn(outputs, targets)

            # 优化器优化(梯度清零，反向传播，优化梯度)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 训练次数
            total_train_number += 1

            if total_train_number % 100 == 0:
                print(f"----训练次数:{total_train_number},loss:{loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_number)

        # 测试步骤开始
        total_loss = 0
        total_accuracy = 0
        nuc.eval()
        with torch.no_grad():
            for data in val_dataloader:
                datas, targets = data
                datas = nuc_utils.datas_totensor(datas, max_length)
                datas = datas.to(device)
                targets = targets.to(device)
                outputs = nuc(datas)

                # 计算损失
                loss = loss_fn(outputs, targets)

                # 计算总损失
                total_loss += loss.item()

                # 计算正确率
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

            print(f"----第{epoch}次训练，总损失{total_loss}")
            print(f"验证集上的总体正确率:{(total_accuracy / val_data_size) * 100}%")
            writer.add_scalar("val_loss", total_loss, total_test_number)
            writer.add_scalar("total_accuracy", total_accuracy / val_data_size, total_test_number)
            total_test_number += 1

            # 保存模型
            # torch.save(nuc, f"net_model/nuc_{epoch}.pth")
            torch.save(nuc.state_dict(), f"net_model/nuc_{epoch}.pth")
            print(f"训练一轮的时间:{time.time() - one_epoch_start_time}")
            print("模型已保存!")

    end_time = time.time()
    print(f"训练总时间:{end_time - start_time}")
    writer.close()









