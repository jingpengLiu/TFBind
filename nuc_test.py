# 测试数据集
import torch, json, time, nuc_utils, nuc_dataset, os
from torch.utils.data import DataLoader
from nuc_model import *
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # 数据集最大长度
    with open("train_test/length.txt", "r") as infile:
        js = infile.read()
        dic = json.loads(js)
        dna_length_max = dic['dna_length_max']
        protein_length_max = dic['protein_length_max']
        max_length = dic['max_length']

    nuc_data, nuc_label = nuc_utils.make_dataset1(dir="./train_test/test.csv")
    nuc_data, nuc_label = nuc_utils.shuffle_data(nuc_data, nuc_label)


    # 测试
    test_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("#################################################")

    # 数据集长度
    test_data_size = len(test_dataset)
    print(f"测试数据集的长度:{test_data_size}")

    # 定义验证设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 记录训练的参数
    epochs = 1000
    total_val_number = 0

    # 添加tensorboard
    writer = SummaryWriter("./test_logs")


    # 读取模型并且验证数据集
    val_model = Nuc()
    for model in range(0, 59):
        start_time = time.time()
        val_model.load_state_dict(torch.load(f"./net_model/nuc_{model}.pth"))
        val_model.to(device)

        # 验证步骤开始
        total_loss = 0
        total_accuracy = 0
        val_model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                datas, targets = data
                datas = nuc_utils.datas_totensor(datas, max_length)
                datas = datas.to(device)
                targets = targets.to(device)
                outputs = val_model(datas)

                # 计算损失
                loss = loss_fn(outputs, targets)

                # 计算总损失
                total_loss += loss.item()

                # 计算正确率
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

            print(f"----第{model}个测试，总损失{total_loss}")
            print(f"测试集上的总体正确率:{(total_accuracy / test_data_size) * 100}%")
            writer.add_scalar("test_loss", total_loss, total_val_number)
            writer.add_scalar("total_accuracy", total_accuracy / test_data_size, total_val_number)
            total_val_number += 1

        end_time = time.time()
        print(f"-----测试模型共花费的时间:{end_time - start_time}")



