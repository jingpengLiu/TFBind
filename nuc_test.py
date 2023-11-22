# test dataset
import torch, json, time, nuc_utils, nuc_dataset, os
from torch.utils.data import DataLoader
from nuc_model import *
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # dataset max
    with open("train_test/length.txt", "r") as infile:
        js = infile.read()
        dic = json.loads(js)
        dna_length_max = dic['dna_length_max']
        protein_length_max = dic['protein_length_max']
        max_length = dic['max_length']

    nuc_data, nuc_label = nuc_utils.make_dataset1(dir="./train_test/test.csv")
    nuc_data, nuc_label = nuc_utils.shuffle_data(nuc_data, nuc_label)


    # test
    test_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("#################################################")

    # dataset length
    test_data_size = len(test_dataset)
    print(f"测试数据集的长度:{test_data_size}")

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # record parameter
    epochs = 1000
    total_val_number = 0

    # add tensorboard
    writer = SummaryWriter("./test_logs")


    # read model and valid dataset
    val_model = Nuc()
    for model in range(0, 59):
        start_time = time.time()
        val_model.load_state_dict(torch.load(f"./net_model/nuc_{model}.pth"))
        val_model.to(device)

        # start valid
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

                # calculate loss
                loss = loss_fn(outputs, targets)

                # calculate all loss
                total_loss += loss.item()

                # calculate acc
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

            print(f"----{model}test，all loss:{total_loss}")
            print(f"test all acc:{(total_accuracy / test_data_size) * 100}%")
            writer.add_scalar("test_loss", total_loss, total_val_number)
            writer.add_scalar("total_accuracy", total_accuracy / test_data_size, total_val_number)
            total_val_number += 1

        end_time = time.time()
        print(f"-----test all time:{end_time - start_time}")



