import json
import time
import torch.cuda
import nuc_dataset
import nuc_utils
from torch.utils.data import DataLoader
from nuc_model import *
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':

    # data_MAX
    with open("train_test/length.txt", "r") as infile:
        js = infile.read()
        dic = json.loads(js)
        dna_length_max = dic['dna_length_max']
        protein_length_max = dic['protein_length_max']
        max_length = dic['max_length']

    nuc_data, nuc_label = nuc_utils.make_dataset1(dir="./train_test/train.csv")
    nuc_data, nuc_label = nuc_utils.shuffle_data(nuc_data, nuc_label)


    # test
    train_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # data_length
    val_dataset = nuc_dataset.MyDataSet(nuc_data, nuc_label, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    print("#################################################")

    # dataset length
    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    print(f"train_data_size:{train_data_size}")
    print(f"val_data_size:{val_data_size}")


    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    nuc = Nuc()
    nuc.to(device)

    # define loss
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # define opti
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(nuc.parameters(), lr=learning_rate)

    # record parameter
    epochs = 1000
    total_train_number = 0
    total_test_number = 0

    # add tensorboard
    writer = SummaryWriter("./logs")
    start_time = time.time()

    # start train and vaild
    for epoch in range(epochs):
        one_epoch_start_time = time.time()
        print(f"###################epoch:{epoch}######################")

        # start traub
        nuc.train()
        for data in train_dataloader:
            datas, targets = data
            datas = nuc_utils.datas_totensor(datas, max_length)
            datas = datas.to(device)
            targets = targets.to(device)
            outputs = nuc(datas)

            # calculate loss
            loss = loss_fn(outputs, targets)

            # Optimizer optimization (gradient zeroing, backpropagation, optimization gradient)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train number
            total_train_number += 1

            if total_train_number % 100 == 0:
                print(f"----train time:{total_train_number},loss:{loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_number)

        # start test
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

                # calculate loss
                loss = loss_fn(outputs, targets)

                # calculate all loss
                total_loss += loss.item()

                # calculate acc
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

            print(f"----{epoch} train，all loss{total_loss}")
            print(f"valid all acc:{(total_accuracy / val_data_size) * 100}%")
            writer.add_scalar("val_loss", total_loss, total_test_number)
            writer.add_scalar("total_accuracy", total_accuracy / val_data_size, total_test_number)
            total_test_number += 1

            # torch.save(nuc, f"net_model/nuc_{epoch}.pth")
            torch.save(nuc.state_dict(), f"net_model/nuc_{epoch}.pth")
            print(f"one train time:{time.time() - one_epoch_start_time}")
            print("save model!")

    end_time = time.time()
    print(f"train all time:{end_time - start_time}")
    writer.close()









