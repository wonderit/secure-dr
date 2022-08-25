#!/usr/bin/env python
# coding: utf-8
import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Plotting

# Import Image Libraries - Pillow and OpenCV
from PIL import Image
import cv2

# Import PyTorch and useful functions
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim

# Import useful sklearn functions
from sklearn.metrics import cohen_kappa_score, accuracy_score

from tqdm import tqdm
import alexnet
import lenet
import smallnet
import os


print(os.listdir("."))

import argparse
from comet_ml import Experiment
import torchsummary

parser = argparse.ArgumentParser()
parser.add_argument("-el", "--epoch_limit", help="Set epoch limit", type=int, default=15)
parser.add_argument("-li", "--log_interval", help="Set batch interval for log", type=int, default=5)
parser.add_argument("-b", "--batch_size", help="Set batch size for log", type=int, default=32)
parser.add_argument("-v", "--valid_size", help="Set validation size for log", type=float, default=0.2)
parser.add_argument("-c", "--is_comet", help="Set isTest", action='store_true')
parser.add_argument("-p", "--comet_project", help="Set project name", type=str, default='secure-dr-plaintext')
parser.add_argument("-seed", "--seed", help="Set random seed", type=int, default=1234)
parser.add_argument("-m", "--model_name", help="model name(alex, lenet, resnet, vgg)", type=str, default='alexnet')
parser.add_argument("-w", "--image_width", help="image width, height", type=int, default=32)

args = parser.parse_args()

# Add the following code anywhere in your machine learning file
if args.is_comet:
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="eIskxE43gdgwOiTV27APVUQtB",
        project_name="secure-dr-plaintext",
        workspace="wonderit",
    )
else:
    experiment = None

# model_name = "lenet"
base_dir = "./aptos2019-blindness-detection/"
# model_type = f"{model_name}_{args.image_width}"

model_type = f"{args.model_name}_{args.image_width}"

# Loading Data + EDA
train_path = f"{base_dir}/train_images/"
test_path = f"{base_dir}/test_images/"
train_csv = pd.read_csv(f"{base_dir}/train.csv")
test_csv = pd.read_csv(f"{base_dir}/test.csv")

print(f"Train Size = {len(train_csv)}")
print(f"Public Test Size = {len(test_csv)}")

counts = train_csv['diagnosis'].value_counts()
class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
for i, x in enumerate(class_list):
    counts[x] = counts.pop(i)

plt.clf()
plt.figure(figsize=(10, 5))
sns.barplot(counts.index, counts.values, alpha=0.8, palette='bright')
plt.title('Distribution of Output Classes')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Target Classes', fontsize=12)
if args.is_comet:
    experiment.log_figure('dist_class', figure=plt)
else:
    plt.savefig('dist_class.png')


# Data Processing
# Our own custom class for datasets
class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name + '.png')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.image_width, args.image_width)),
    transforms.RandomHorizontalFlip(p=0.4),
    # transforms.ColorJitter(brightness=2, contrast=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_data = CreateDataset(df_data=train_csv, data_dir=train_path, transform=transforms)

# Percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Create Samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define Model Architecture
# model = vgg.__dict__['vgg16']()
# model_name = 'alexnet'
# model = alexnet.AlexNet().to(device)

if args.model_name == "alexnet":
    model = alexnet.AlexNet().to(device)
elif args.model_name == "lenet":
    model = lenet.LeNet().to(device)
elif args.model_name == "smallnet":
    model = smallnet.SmallNet().to(device)

print('model', model)
torchsummary.summary(model, (3, args.image_width, args.image_width))

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    model.cuda()

# Trainable Parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))

# Training (Fine Tuning) and Validation
# specify loss function (categorical cross-entropy loss)
criterion = nn.MSELoss()

# specify optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.00015)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
# n_epochs = 15

valid_loss_min = np.Inf

# keeping track of losses as it happen
train_losses = []
valid_losses = []
val_kappa = []
test_accuracies = []
valid_accuracies = []
kappa_epoch = []
batch = 0

for epoch in range(1, args.epoch_limit + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float()
        target = target.view(-1, 1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output.to(torch.float), target.to(torch.float))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
        # Update Train loss and accuracies
        train_loss += loss.item() * data.size(0)

    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in tqdm(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float()
        # forward pass: compute predicted outputs by passing inputs to the model
        target = target.view(-1, 1)
        with torch.set_grad_enabled(True):
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item() * data.size(0)
        # output = output.cohen_kappa_score_kappa_score)
        y_actual = target.data.cpu().numpy()
        y_pred = output[:, -1].detach().cpu().numpy()
        val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))

        # calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    valid_kappa = np.mean(val_kappa)
    kappa_epoch.append(np.mean(val_kappa))
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print training/validation statistics 
    print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f}'.format(
        epoch, train_loss, valid_loss, valid_kappa))

    if args.is_comet:
        experiment.log_metric("train_loss", train_loss, epoch=epoch)
        experiment.log_metric("valid_loss", valid_loss, epoch=epoch)
        experiment.log_metric("valid_kappa", valid_kappa, epoch=epoch)

    ##################
    # Early Stopping #
    ##################
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), f'best_model_{model_type}.pt')
        valid_loss_min = valid_loss


plt.cla()
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)

if args.is_comet:
    experiment.log_figure(f'learning-curve-{model_type}', figure=plt)
else:
    plt.savefig(f"learning-curve-{model_name}.png")

plt.cla()
plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
plt.legend("")
plt.xlabel("Epochs")
plt.ylabel("Kappa Score")
plt.legend(frameon=False)


if args.is_comet:
    experiment.log_figure(f"learning-curve-kappa-{model_type}", figure=plt)
else:
    plt.savefig(f"learning-curve-kappa-{model_name}.png")


model.load_state_dict(torch.load(f'best_model_{model_type}.pt'))


# Inference
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((args.image_width, args.image_width)),
    # torchvision.transforms.ColorJitter(brightness=2, contrast=2),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


test_csv['diagnosis'] = -1


test_data = CreateDataset(df_data=test_csv, data_dir=test_path, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


def round_off_preds(preds, coef=[0.5, 1.5, 2.5, 3.5]):
    for i, pred in enumerate(preds):
        if pred < coef[0]:
            preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            preds[i] = 3
        else:
            preds[i] = 4
    return preds


def predict(testloader):
    '''Function used to make predictions on the test set'''
    model.eval()
    preds = []
    for batch_i, (data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pr = output.detach().cpu().numpy()
        # print(pr.shape)
        for i in pr:
            preds.append(i.item())

    return preds


# ### TTA (Test Time Augmentation)
# preds1 = np.array(predict(testloader=test_loader))
# preds2 = np.array(predict(testloader=test_loader))
# preds3 = np.array(predict(testloader=test_loader))
# preds4 = np.array(predict(testloader=test_loader))
# preds5 = np.array(predict(testloader=test_loader))
# preds6 = np.array(predict(testloader=test_loader))
# preds7 = np.array(predict(testloader=test_loader))
# preds8 = np.array(predict(testloader=test_loader))
#
# preds = (preds1 + preds2 + preds3 + preds4 +
#          preds5 + preds6 + preds7 + preds8)/8.0

# preds = round_off_preds(preds)
# preds = np.array(predict(testloader=test_loader))

# # # Generating Submission File
# sample_sub = pd.read_csv('./aptos2019-blindness-detection/sample_submission.csv')
#
# sample_sub.diagnosis = preds
# sample_sub.diagnosis = sample_sub['diagnosis'].astype(int)
#
# sample_sub.head()
#
# sample_sub.to_csv(f'submission_{model_name}.csv', index=False)