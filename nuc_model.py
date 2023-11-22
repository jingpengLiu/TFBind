from torch import nn
import torch
# class Nuc(nn.Module):
#     def __init__(self):
#         super(Nuc, self).__init__()
#         self.conv1 = nn.Conv2d(8, 32, 5, padding=2)
#         self.max1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
#         self.max2 = nn.MaxPool2d(2)
#         self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
#         self.max3 = nn.MaxPool2d(2)
#         self.flat = nn.Flatten()
#         self.line1 = nn.Linear(64*11*8, 64)
#         self.line2 = nn.Linear(64, 10)
#         self.line3 = nn.Linear(10, 2)
#
#
#     def forward(self, input):
#         output = self.conv1(input)
#         print(output.shape)
#         output = self.max1(output)
#         print(output.shape)
#         output = self.conv2(output)
#         print(output.shape)
#         output = self.max2(output)
#         print(output.shape)
#         output = self.conv3(output)
#         print(output.shape)
#         output = self.max3(output)
#         print(output.shape)
#         output = self.flat(output)
#         print(output.shape)
#         output = self.line1(output)
#         print(output.shape)
#         output = self.line2(output)
#         print(output.shape)
#         output = self.line3(output)
#         print(output.shape)
#         print(output)
#         return output

class Nuc(nn.Module):
    def __init__(self):
        super(Nuc, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 11 * 8, 64),
            nn.Linear(64, 10),
            nn.Linear(10, 2)
        )


    def forward(self, input):
        output = self.model(input)
        return output

if __name__ == '__main__':
    test_tensor = torch.ones((64, 8, 94, 65))
    nuc = Nuc()
    output = nuc(test_tensor)
    print(output)