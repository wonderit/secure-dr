import time
import os
import datetime
import math
import numpy as np
import torch
from torch.nn.functional import softmax
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

# device setting, nvidia='cuda:0' | m1='mps' | cpu='cpu'
DEVICE = 'cuda:0'  # "cuda:0" or "mps" or "cpu"
DEVICE = 'mps'

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.0001


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.num_classes = 10
        self.ch = 1

        self.conv1 = nn.Conv2d(self.ch, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.actv = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

        self.flatten = nn.Flatten()

        self.fn1 = nn.Linear(7 * 7 * 256, 256)
        self.fn2 = nn.Linear(256, 100)
        self.fn3 = nn.Linear(100, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actv(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv(x)

        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.fn1(x)
        x = self.fn2(x)
        x = self.fn3(x)

        return softmax(x)


train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

test_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

model = Model()
n_params = sum(p.numel() for p in model.parameters())

print("\n===== Model Architecture =====")
print(model, "\n")

print("\n===== Model Parameters =====")
print(" - {}".format(n_params), "\n\n")

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

total_train_iter = math.ceil(len(train_set) / BATCH_SIZE)
total_valid_iter = math.ceil(len(test_set) / BATCH_SIZE)

device = torch.device(DEVICE)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.to(device)

train_start = time.time()
epochs_times = []

print("\n\nTraining start time : {}\n\n".format(datetime.datetime.now()))

for epoch in range(EPOCHS):
    epoch_start = time.time()
    train_loss, train_acc = 0.0, 0.0

    for step, data in enumerate(train_loader):
        iter_start = time.time()
        model.train()
        image, target = data

        image = image.to(device)
        target = target.to(device)

        out = model(image)

        acc = (torch.max(out, 1)[1].cpu().numpy() == target.cpu().numpy())
        acc = float(np.count_nonzero(acc) / BATCH_SIZE)

        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc

        if step % int(total_train_iter / 2) == 0:
            print("[train %5s/%5s] Epoch: %4s | Time: %6.2fs | loss: %10.4f | Acc: %10.4f" % (
                step + 1, total_train_iter, epoch + 1, time.time() - iter_start, round(loss.item(), 4), float(acc)))

    train_loss = train_loss / total_train_iter
    train_acc = train_acc / total_train_iter
    epoch_runtime = time.time() - epoch_start
    print("[Epoch {} training Ended] > Time: {:.2}s/epoch | Loss: {:.4f} | Acc: {:g}\n".format(
        epoch + 1, epoch_runtime, np.mean(train_loss), train_acc))

    epochs_times.append(epoch_runtime)

program_runtime = time.time() - train_start

print("\n\nTraining running time : {:.2}\n\n".format(program_runtime))

epochs_times = list(map(str, epochs_times))
epochs_times = list(map(lambda x: str(x) + "\n", epochs_times))

filename = os.path.basename(__file__).split('.')[0] + ".txt"
with open(filename, 'w') as f:
    f.writelines(epochs_times)
print(f'save success epoch times! -> {filename}')